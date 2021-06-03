"""
first_order to train a group of keys
"""

# from sklearn.metrics.pairwise import cosine_similarity
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from neurodiffeq import diff  # the differentiation operation
from neurodiffeq.conditions import IVP  # the initial condition
from neurodiffeq.networks import FCNN  # fully-connect neural network
from neurodiffeq.solvers import Solver1D,Solver2D
from sklearn.metrics import mean_squared_error
from neurodiffeq.solvers_utils import SolverConfig
from sklearn.decomposition import PCA, TruncatedSVD
import os
import torch
import torch.nn as nn
from neurodiffeq.conditions import NoCondition

MAX_EPOCHS = 3000
BURNIN = 0
SUB_RATE = 100

# fixed initial conditions
# INIT_VAL = [IVP(t_0=0.0, u_0=0.5)]
T_MIN = -2
T_MAX = 2
ts = torch.linspace(T_MIN, T_MAX, 100).reshape(-1, 1)


def train_methods(systems_dict, system_type, epochs):
    """
    take the systems dictionary and return the final solution and final weights for each
    args:
        systems_dict: the dictionary of diffeqs e.g. exp:[lambda u,t:diff(u,t) -u]
    """

    SOLUTIONS = {}
    DYDX = {}
    DYDW = {}
    WEIGHTS = {}

    for system_name, (system,INIT_VAL) in systems_dict.items():

        class NetWithIC(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, init_params):
                t,u_0 = init_params[:,0].reshape(-1,1),init_params[:,1].reshape(-1,1)
                t_0 = t[0]
                u_0_prime = 0
                network_output = self.net(t)
                resy = u_0 + (t - t_0) * u_0_prime + ((1 - torch.exp(-t + t_0)) ** 2) * network_output
                return_list = [resyv for resyv in resy]#((resyv) for resyv in resy)
                # print(return_list)
                return return_list
        raw_net = FCNN(1, 1)
        net_with_condition = NetWithIC(raw_net)

        def pde_system(u, t,u_0):
            print(len(u))
            print(t.shape)
            return diff(u, t[:,0].reshape(-1,1), order=2) + u

        # note that we use Solver2D instead of Solver1D because u0 is also considered an input.
        solver = Solver2D(
            pde_system=pde_system,
            conditions=[NoCondition()],
            nets=[net_with_condition],
            xy_min=(0., 0.),
            xy_max=(1., 1.),
        )


        solver.fit(max_epochs=epochs)

        # Saving locally
        models_dir = f"models/{system_type}"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        solver.save(f'./{models_dir}/{system_name}')

        solution = solver.get_solution(best=False)(ts.reshape(-1, 1), to_numpy=True)
        SOLUTIONS[system_name] = solution.ravel()

        solution = solver.get_solution(best=False)
        ts.requires_grad = True
        u = solution(ts.reshape(-1, 1))
        dydx = torch.autograd.grad(u.sum(), ts)[0]
        DYDX[system_name] = dydx.ravel()

        solution = solver.get_solution(best=False)
        model_weights = []
        fin_weights = []
        for tval in ts:
            u = solution(tval.reshape(-1, 1))
            u.backward(torch.ones_like(u))
            for net in solution.nets:
                for param in net.parameters():
                    model_weights.append(param.grad.detach().numpy().ravel())
            fin_weights.append(np.concatenate(model_weights))
            solution.nets[0].zero_grad()

        DYDW[system_name] = (np.concatenate(fin_weights)).ravel()

        model_weights = []
        solution = solver.get_solution(best=False)
        for net in solution.nets:
            for param in net.parameters():
                model_weights.append(param.detach().numpy().ravel())
        fin_weights = np.concatenate(model_weights)
        WEIGHTS[system_name] = fin_weights.ravel()

        # record final model and final solution and return it
        print(f"***{system_type}_{system_name}_trained***")

    return SOLUTIONS, WEIGHTS, DYDW, DYDX


# corpus of training diffeqs
DIFFEQS_TRAIN = {
    # 'exp_a': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=0.5)]],
    # 'exp_b': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=1.)]],
    # 'exp_c': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=0.21)]],
    'exp_d': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=3.5)]],
    'exp_e': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=.8)]],

    'exp1_a': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=0.5)]],
    'exp1_b': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=1.)]],
    # 'exp1_c': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=0.21)]],
    # 'exp1_d': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=3.5)]],
    # 'exp1_e': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=.8)]],

    'tanh_a': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=0.5)]],
    'tanh_b': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=1.)]],
    'tanh_c': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=0.21)]],
    # 'tanh_d': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=3.5)]],
    # 'tanh_e': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=.8)]],

    # 'sigmoid_a': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=0.5)]],
    'sigmoid_b': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=1.)]],
    # 'sigmoid_c': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=0.21)]],
    'sigmoid_d': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=3.5)]],
    # 'sigmoid_e': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=.8)]],

    # 'newt_cool_a': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=0.5)]],
    # 'newt_cool_b': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=1.)]],
    # 'newt_cool_c': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=0.21)]],
    # 'newt_cool_d': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=3.5)]],
    'newt_cool_e': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=.8)]],

    # 'cubic_a': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=0.5)]],
    # 'cubic_b': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=1.)]],
    # 'cubic_c': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=0.21)]],
    'cubic_d': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=3.5)]],
    # 'cubic_e': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=.8)]],

    'baseline_a': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=0.5)]],
    # 'baseline_b': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=1.)]],
    # 'baseline_c': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=0.21)]],
    # 'baseline_d': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=3.5)]],
    'baseline_e': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=.8)]],

}

# train all the keys (MAXEPS)
SOLUTIONS_TRAIN, WEIGHTS_TRAIN, DYDW_TRAIN, DYDX_TRAIN = train_methods(DIFFEQS_TRAIN, 'train', MAX_EPOCHS)

sol_collector = []
for val in SOLUTIONS_TRAIN.keys():
    sol_collector.append(SOLUTIONS_TRAIN[val])
sol_collector = np.concatenate(sol_collector)
sol_collector = sol_collector.reshape(-1, len(ts))
solution_pca = PCA(n_components=5)
solution_pca.fit(sol_collector)
PCA_SOLUTIONS_TRAIN = solution_pca.transform(sol_collector)

DIFFEQS_TEST = {
    'exp_d': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=3.5)]],
    'exp1_b': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=1.)]],
    'tanh_a': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=0.5)]],
    'sigmoid_d': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=3.5)]],
    'newt_cool_e': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=.8)]],
    'cubic_d': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=3.5)]],
    'baseline_a': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=0.5)]],
}

# train all the queries (500 epochs)
SOLUTIONS_TEST, WEIGHTS_TEST, DYDW_TEST, DYDX_TEST = train_methods(DIFFEQS_TEST, 'test', 500)

sol_collector1 = []
for val in SOLUTIONS_TEST.keys():
    sol_collector1.append(SOLUTIONS_TEST[val])
sol_collector1 = np.concatenate(sol_collector1)
sol_collector1 = sol_collector1.reshape(-1, len(ts))
PCA_SOLUTIONS_TEST = solution_pca.transform(sol_collector1)
SOLUTIONS_GT_TEST, WEIGHTS_GT_TEST, DYDW_GT_TEST, DYDX_GT_TEST = train_methods(DIFFEQS_TEST, 'gt_test', MAX_EPOCHS)
ts_np = ts.detach().numpy().ravel()

def pretrain_methods(train_dict, test_dict, epochs):
    PRETRAINED_SOLUTIONS = {}
    for system_name, system in train_dict.items():
        for system_name_test, system_test in test_dict.items():
            solverconfig = SolverConfig()
            solverconfig.diff_eqs = system_test
            solver = Solver1D.load(f'./models/train/{system_name}', solverconfig)
            solver.fit(max_epochs=epochs)
            solution = solver.get_solution(best=False)(ts, to_numpy=True)
            PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'] = solution
    return PRETRAINED_SOLUTIONS


def metric_evaluation(PRETRAINED_SOLUTIONS, metric):
    errors_pre_gt = np.zeros((len(DIFFEQS_TEST), len(DIFFEQS_TRAIN)))
    errors_train_test = np.zeros((4, len(DIFFEQS_TEST), len(DIFFEQS_TRAIN)))
    for j, (system_name, system) in enumerate(DIFFEQS_TRAIN.items()):
        for i, (system_name_test, system_test) in enumerate(DIFFEQS_TEST.items()):
            errors_pre_gt[i, j] = metrics[metric](PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'],
                                                  SOLUTIONS_GT_TEST[system_name_test])
            errors_train_test[0, i, j] = metrics[metric](PCA_SOLUTIONS_TEST[i], PCA_SOLUTIONS_TRAIN[j])
            errors_train_test[1, i, j] = metrics[metric](SOLUTIONS_TEST[system_name_test], SOLUTIONS_TRAIN[system_name])

    return errors_pre_gt, errors_train_test


cosine_similarity = lambda a, b: np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b))
metrics = {'L2': mean_squared_error,
           'cosine_similarity': cosine_similarity
           }

for epoch_values in [100, 200, 500]:
    PRETRAINED_SOLUTIONS = pretrain_methods(DIFFEQS_TRAIN, DIFFEQS_TEST, epoch_values)
    errors_pre_gt, errors_train_test = metric_evaluation(PRETRAINED_SOLUTIONS, 'L2')

    fig, ax = plt.subplots(3, len(DIFFEQS_TEST.keys()), figsize=(15, 25), sharex=True)
    fig.suptitle(f'L2_{epoch_values}')
    for i in range(len(DIFFEQS_TEST.keys())):
        ax[0, i].set_title(list(DIFFEQS_TEST.keys())[i])
        plt.setp(ax[2, i].xaxis.get_majorticklabels(), rotation=90)

        ax[0, i].bar(DIFFEQS_TRAIN.keys(), errors_pre_gt[i, :])
        ax[1, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[0, i, :])
        ax[2, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[1, i, :])

        ax[0, i].set_ylabel('pretrained model errors on quer diffeq')
        ax[1, i].set_ylabel('train-test PCA (5 components) error')
        ax[2, i].set_ylabel('train-test solution error')

        ax[0, i].set_yscale('log')
        ax[1, i].set_yscale('log')
        ax[2, i].set_yscale('log')

        # plt.show()
    plt.tight_layout()
    plt.savefig(f'L2_pca_ensemble_{epoch_values}.pdf')
