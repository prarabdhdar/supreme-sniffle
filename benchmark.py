from Environment import model
import cvxpy as cp
import numpy as np


fd = 10
Ts = 20e-3
M = 4
K = 6
Ns = 11
max_p = 38. #dBm
min_p = 1
p_n = -114. #dBm
state_num = 2
action_num = 6  #action_num

weight_file = 'dqn_6.mat'


mod = model(fd, Ts, M, K, Ns, min_p, max_p, p_n, action_num)
H, _ = mod.state_space()
H = H[:, :, 0]
y = np.zeros(M*K)
for i in range(K):
    for j in range(M):
        y[i*M+j] = H[j, i]

sigma2 = 1e-3*pow(10., p_n/10.)

X = cp.Variable(M*(K+1))

for i in range(K):
    R_k=0
    r_c = (cp.quad_over_lin(y[i*M : i*M+M-1].T@X[M*K : M*K+M-1], sigma2) + cp.quad_over_lin(y[i*M : i*M+M-1].T@X[i*M : i*M+M-1], sigma2))#/np.float32(sigma2)
    R_c = cp.log(1 + r_c)
    for j in range(K):
        if(i==j):
            continue
        r_k = cp.quad_over_lin(y[j*M : j*M+M-1].T@X[j*M : j*M+M-1], sigma2)#/np.float32(sigma2)
        R_k = R_k + cp.log(1 + r_k)
    problem = cp.Minimize(-R_c - R_k)
    constraints = [cp.sum_squares(X)<=max_p]
    prob = cp.Problem(problem, constraints)
    prob.solve()