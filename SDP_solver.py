import cvxpy as cp
import numpy as np
from time import time
import scipy
import pdb


def compute_mask(K, retain_ratio):
    start_time = time()
    N = K[0].shape[0] # Num. of training samples
    P = len(K) # Num. of parameters
    C = cp.Variable(P, pos=True)

    K_x = cp.sum([C[i] * K[i] for i in range(P)])


    # Minimum eigenvalue constraint
    t = cp.Variable(1, pos=True)
    constraints = [K_x >> t]
    # constraints += [t >= 0]  #not necessary due to constraint of C ?

    # Pruned Ratio constraint
    constraints += [(cp.norm(C,1) / P) <= retain_ratio]
    # saliency criterion is constrained to 0~1
    for i in range(P):
        constraints.append(C[i] <= 1)

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    # print("Pruned ratio", np.linalg.norm(C.value, 1)/P)
    # print("A solution X is")
    # print(C.value)
    print("Elapsed time : {}".format(time() - start_time))

    # Sanity Check
    # print("SANITY CHECK")
    # print(scipy.linalg.eigvals(K_x.value))
    # value = 0
    # for idx, c in enumerate(C):
    #     rand_num = np.random.rand() > 0.95
    #     value += K[idx] * rand_num
    # print(scipy.linalg.eigvals(value))
    return C.value.astype(np.float32)


if __name__ == '__main__':
    K = []
    P=100
    N=10
    for _ in range(P):
        a = np.random.rand(N, N)
        K.append(a.T @ a)
    compute_mask(K, 0.05)