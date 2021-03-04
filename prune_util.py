import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SDP_solver import compute_mask

import copy
import types
import pdb


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def __NTK_Prune(net, keep_ratio, train_dataloader, device, random=False):
    '''
    Outputs masks for P NxN matrix. (p,q) element of i th matrix denotes the product to i th weight gradients of pth data and qth data.
    '''
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            # layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    N = 10
    P = 450
    NTK_matrices = [np.ones((N,N)) for _ in range(P)]
    for idx_n in range(N):
        # Grab a single batch from the training dataset
        inputs, targets = next(iter(train_dataloader))
        inputs = inputs.to(device)

        # Compute gradients w.r.t. outputs
        net.zero_grad()
        outputs = net.forward(inputs)
        outputs.backward(torch.ones_like(outputs))

        # grads = []
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if name == 'conv1': #Choose layer by name
                    grads = layer.weight.grad
                    layer_shape = grads.shape
                    break
        # pdb.set_trace()
        flat_grads = torch.flatten(grads)
        if random:
            num_retain = int(len(flat_grads) * keep_ratio)
            idx = np.random.choice(range(len(flat_grads)), num_retain)
            mask = torch.zeros(torch.numel(flat_grads))
            mask[idx] = 1
            print("Pruned Ratio: {:.3f}".format(sum(mask) / len(mask)))
            mask = mask.view(layer_shape)
            break

        for idx_p, g in enumerate(flat_grads):
            NTK_matrices[idx_p][idx_n,:] *= g.data.item()
            NTK_matrices[idx_p][:,idx_n] *= g.data.item()
            NTK_matrices[idx_p][idx_n,idx_n] /= g.data.item()

    if not random:
        soft_mask = compute_mask(NTK_matrices, keep_ratio)
        num_retain = int(len(soft_mask) * keep_ratio)
        retain_idx = soft_mask.argsort()[-num_retain]
        mask = soft_mask > soft_mask[retain_idx]
        mask = mask.astype(np.float32)
        print("Pruned Ratio: {:.3f}".format(1 - sum(mask)/len(mask)))
        mask = torch.Tensor(mask).view(layer_shape)

    keep_masks = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if name == 'conv1':  # Choose layer by name or order?
                keep_masks.append(mask.to(device))
            else:
                keep_masks.append(torch.ones_like(layer.weight.grad).to(device))

    return(keep_masks)


def NTK_Prune(net, keep_ratio, train_dataloader, device):
    # Hyper-parameters
    N = 10  #Number of mini-batches used to create NTK mtx
    num_prune_layer = 2
    num_layer = 0
    total = 0
    pruned = 0
    # Loop through layers
    for name, m in net.named_modules():
        if num_layer == num_prune_layer :
            break
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            num_layer += 1
            # Init. values
            P = torch.numel(m.weight)
            m_shape = m.weight.shape
            NTK_matrices = [np.ones((N, N)) for _ in range(P)]
            num_retain = int(P * keep_ratio)

            print('\nlayer: {} : {}'.format(name, P))
            # Loop through samples to create empirical NTK
            for idx_n in range(N):
                inputs, targets = next(iter(train_dataloader))
                inputs = inputs.to(device)

                # Compute gradients w.r.t. outputs
                net.zero_grad()
                outputs = net.forward(inputs)
                outputs.backward(torch.ones_like(outputs))
                flat_grads = torch.flatten(m.weight.grad.clone())

                # Compute values into NTK mtx.
                for idx_p, g in enumerate(flat_grads):
                    NTK_matrices[idx_p][idx_n, :] *= g.item()
                    NTK_matrices[idx_p][:, idx_n] *= g.item()
                    if g.item() != 0.0:
                        NTK_matrices[idx_p][idx_n, idx_n] /= g.item()

            # Run CVX to compute mask
            soft_mask = compute_mask(NTK_matrices, keep_ratio)
            retain_idx = soft_mask.argsort()[::-1][-num_retain:]
            # mask = (soft_mask > soft_mask[retain_idx]).astype(np.float32)
            mask = torch.zeros_like(torch.Tensor(soft_mask))
            mask[retain_idx] = 1
            mask = torch.Tensor(mask).view(m_shape).to(device)
            # Multiply mask to prune out weights
            m.weight.data.mul_(mask)

            total += torch.numel(m.weight.data)
            pruned += torch.sum(m.weight.data==0.0)
            print('\t total params: {:d} \t remaining params: {:d}'.
                format(mask.numel(), int(torch.sum(mask))))

    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {:.3f}'.format(total, pruned, float(pruned/total)))

def Random_Prune(net, keep_ratio, train_dataloader, device):
    # Hyper-parameters
    total = 0
    pruned = 0
    num_prune_layer = 2
    num_layer = 0
    total = 0
    pruned = 0
    # Loop through layers
    for name, m in net.named_modules():
        if num_layer == num_prune_layer :
            break
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            num_layer += 1
            # Init. values
            P = torch.numel(m.weight)
            m_shape = m.weight.shape
            num_retain = int(P * keep_ratio)

            # Randomly select indices
            idx = np.random.choice(range(P), num_retain, replace=False)
            mask = torch.zeros(P)
            mask[idx] = 1
            mask = mask.view(m_shape).to(device)

            # Multiply mask to prune out weights
            m.weight.data.mul_(mask)

            total += torch.numel(m.weight.data)
            pruned += torch.sum(m.weight.data == 0)
            print('layer: {} \t total params: {:d} \t remaining params: {:d}'.
                format(name, mask.numel(), int(torch.sum(mask))))

    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {:.3f}'.format(total, pruned, float(pruned/total)))