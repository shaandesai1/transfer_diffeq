
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
import matplotlib.pyplot as plt

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


solsa = np.load('data/q3_train_solution/3000.npy')
solsb = np.load('data/baseline_train_solution/3000.npy')
analytical =np.load('data/q3_gt_test_solution/3000.npy')
# pre1 =np.load('data/q2_q2_pretrain_500_solution/500.npy')
# pre2 =np.load('data/baseline_q2_pretrain_500_solution/500.npy')

plt.figure()
plt.plot(solsa,label='q2')
plt.plot(solsb,label='high_order_2')
plt.plot(analytical,label='analytical_q2')
# plt.plot(pre1,label='pre_q2_q2')
# plt.plot(pre2,label='pre_baseline_q2')
plt.legend()
plt.show()