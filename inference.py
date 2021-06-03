"""
first_order to train a group of keys
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from neurodiffeq import diff  # the differentiation operation
from neurodiffeq.conditions import IVP  # the initial condition
from neurodiffeq.networks import FCNN  # fully-connect neural network
from neurodiffeq.solvers import Solver1D
from neurodiffeq.callbacks import WeightCallback
from neurodiffeq.callbacks import WeightCallback1, SolutionCallback, SaddleCallback
from neurodiffeq.callbacks import PeriodLocal
from sklearn.metrics import mean_squared_error
# from sklearn.metrics.pairwise import cosine_similarity
import copy


MAX_EPOCHS = 30
BURNIN = 0
SUB_RATE = 1

# fixed initial conditions
INIT_VAL = [IVP(t_0=0.0, u_0=0.5)]
T_MIN = -2
T_MAX = 2
ts = torch.linspace(T_MIN, T_MAX, 100).reshape(-1, 1)

#corpus of training diffeqs
DIFFEQS_TRAIN = {
    'exp': lambda u, t: [diff(u, t) + u],
    'exp1': lambda u, t: [diff(u, t) - u],
    'tanh': lambda u, t: [diff(u, t) + u ** 2 - 1],
    'psig': lambda u, t: [diff(u, t) - 3 * u + u ** 2],
    'r1': lambda u, t: [diff(u, t) - u + u ** 2 + u ** 3],
    'r2': lambda u, t: [diff(u, t) + u + u ** 2],
    'r3': lambda u, t: [diff(u, t) + u ** 2],
    'r4': lambda u, t: [diff(u, t) - u ** 2],
    'q1': lambda u, t: [diff(u, t) - u + u ** 2],
    'q2': lambda u, t: [diff(u, t) - u + u ** 2 - u ** 3],
    'q3': lambda u, t: [diff(u, t) + u ** 2 + u ** 4],
    'q4': lambda u, t: [diff(u, t) - u ** 2 - u ** 4],
    'high_order1': lambda u, t: [diff(u, t) + u - u ** 2 + u ** 3 - u ** 4 + u ** 5],
    'high_order2': lambda u, t: [diff(u, t) - u + u ** 2 - u ** 3 + u ** 4 - u ** 5],
    'baseline': lambda u, t: [diff(u,t)]
}
DIFFEQS_TEST = {
    'q1': lambda u, t: [diff(u, t) - u + u ** 2],
    'q2': lambda u, t: [diff(u, t) - u + u ** 2 - u ** 3],
}



BURNIN = 1
eps = [i for i in range(BURNIN,MAX_EPOCHS) if (i%SUB_RATE==0)]
equations = list(DIFFEQS_TRAIN.keys())

dydx = np.zeros((len(eps),len(equations)))
dydw= np.zeros((len(eps),len(equations)))
wdots= np.zeros((len(eps),len(equations)))
sols = np.zeros((len(eps),len(equations)))

for eqs_test in list(DIFFEQS_TEST.keys()):
    for meth_dex in range(len(equations)):
        for ep_index,ep in enumerate(eps):
            param_a = np.load(f'data/{equations[meth_dex]}_train_w/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_gt_test_w/{ep}.npy')
            wdots[ep_index,meth_dex]=np.dot(param_a,param_b)/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))

            param_a = np.load(f'data/{equations[meth_dex]}_train_solution/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_gt_test_solution/{ep}.npy')
            sols[ep_index,meth_dex]=np.dot(param_a.ravel(),param_b.ravel())/(np.linalg.norm(param_a.ravel(),ord=2)*np.linalg.norm(param_b.ravel(),ord=2))

            param_a = np.load(f'data/{equations[meth_dex]}_train_dydw/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_gt_test_dydw/{ep}.npy')
            dydw[ep_index,meth_dex]=np.dot(param_a,param_b)/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))

            param_a = np.load(f'data/{equations[meth_dex]}_train_dydx/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_gt_test_dydx/{ep}.npy')
            dydx[ep_index, meth_dex] = np.dot(param_a.ravel(), param_b.ravel()) / (np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))

    fig,ax = plt.subplots(1,4,figsize=(15,10))
    fig.suptitle(eqs_test)
    for i in range(len(equations)):
        ax[0].plot(eps,wdots[:,i],label=equations[i])
        ax[1].plot(eps,dydw[:,i],label=equations[i])
        ax[2].plot(eps,dydx[:,i],label=equations[i])
        ax[3].plot(eps, sols[:, i], label=equations[i])
    ax[0].set_title('Weights cosine similarity')
    ax[1].set_title(r'$\frac{dy}{dw}$ Cosine Similarity')
    ax[2].set_title(r'$\frac{dy}{dx}$ Cosine Similarity')
    ax[3].set_title('Solutions cosine similarity')
    plt.legend()
    plt.show()
