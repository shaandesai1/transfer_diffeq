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
from neurodiffeq.solvers import Solver1D
from sklearn.metrics import mean_squared_error
from neurodiffeq.solvers_utils import SolverConfig

import os
import torch.optim as optim

MAX_EPOCHS = 3000
BURNIN = 0
SUB_RATE = 100

# fixed initial conditions
INIT_VAL = [IVP(t_0=0.0, u_0=0.5)]
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

    # MODELS = {}
    for system_name, (system,INIT_VAL) in systems_dict.items():
        # define the baseline network
        NETS = [FCNN(n_input_units=1, n_output_units=1,hidden_units=[30,30])]
        # call the dy/dweights (integrated over the inputs)
        # cb = PeriodLocal(period=SUB_RATE)
        # wcb = WeightCallback(f'{system_name}_{system_type}', ts)
        # cb.set_action_callback(wcb)
        # # call the weights
        # cb0 = PeriodLocal(period=SUB_RATE)
        # wcb0 = WeightCallback1(f'{system_name}_{system_type}', ts)
        # cb0.set_action_callback(wcb0)
        # # call the solutions
        # cb1 = PeriodLocal(period=SUB_RATE)
        # wcb1 = SolutionCallback(f'{system_name}_{system_type}', ts)
        # cb1.set_action_callback(wcb1)
        # # call dydx
        # cb2 = PeriodLocal(period=SUB_RATE)
        # wcb2 = SaddleCallback(f'{system_name}_{system_type}', ts)
        # cb2.set_action_callback(wcb2)

        solver = Solver1D(
            ode_system=system,
            conditions=INIT_VAL,
            t_min=T_MIN,
            t_max=T_MAX,
            nets=NETS
        )
        solver.fit(max_epochs=epochs)

        # Saving locally
        models_dir = f"models/{system_type}"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        solver.save(f'./{models_dir}/{system_name}')

        # print(solver.lowest_loss)
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
        fin_weights = []
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
    'tanh': [lambda u, t: [diff(u, t) + u ** 2 - 1],[IVP(t_0=0.0, u_0=0.5)]],
    'sigmoid': [lambda u, t: [diff(u, t) + u ** 2 - u],[IVP(t_0=0.0, u_0=0.5)]],
    'sigmoid1': [lambda u, t: [diff(u, t) + u ** 2 - u],[IVP(t_0=0.0, u_0=0.45)]],
}

# train all the keys (MAXEPS)
SOLUTIONS_TRAIN, WEIGHTS_TRAIN, DYDW_TRAIN, DYDX_TRAIN = train_methods(DIFFEQS_TRAIN, 'train', MAX_EPOCHS)

DIFFEQS_TEST = {
    'tanh': [lambda u, t: [diff(u, t) + u ** 2 - 1],[IVP(t_0=0.0, u_0=0.5)]],
    'sigmoid': [lambda u, t: [diff(u, t) + u ** 2 - u],[IVP(t_0=0.0, u_0=0.5)]],
    'sigmoid1': [lambda u, t: [diff(u, t) + u ** 2 - u],[IVP(t_0=0.0, u_0=0.45)]],
}

# train all the queries (500 epochs)
SOLUTIONS_TEST, WEIGHTS_TEST, DYDW_TEST, DYDX_TEST = train_methods(DIFFEQS_TEST, 'test', 500)

# train all the queries (MAXEPS)
SOLUTIONS_GT_TEST,WEIGHTS_GT_TEST, DYDW_GT_TEST, DYDX_GT_TEST  = train_methods(DIFFEQS_TEST, 'gt_test', MAX_EPOCHS)
ts_np = ts.detach().numpy().ravel()


# PRE_MAX_EPOCHS = 500

def pretrain_methods(train_dict, test_dict, epochs):
    PRETRAINED_SOLUTIONS = {}
    for system_name, system in train_dict.items():
        for system_name_test, system_test in test_dict.items():
            # cb0 = PeriodLocal(period=SUB_RATE)
            # wcb0 = WeightCallback1(f'{system_name}_{system_name_test}_pretrain_{epochs}', ts)
            # cb0.set_action_callback(wcb0)
            # # call the solutions
            # cb1 = PeriodLocal(period=SUB_RATE)
            # wcb1 = SolutionCallback(f'{system_name}_{system_name_test}_pretrain_{epochs}', ts)
            # cb1.set_action_callback(wcb1)
            # # call dydx
            # cb2 = PeriodLocal(period=SUB_RATE)
            # wcb2 = SaddleCallback(f'{system_name}_{system_name_test}_pretrain_{epochs}', ts)
            # cb2.set_action_callback(wcb2)
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
            print(system_name, system_name_test)
            errors_pre_gt[i, j] = metrics[metric](PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'],
                                                  SOLUTIONS_GT_TEST[system_name_test])
            errors_train_test[0, i, j] = metrics[metric](SOLUTIONS_TEST[system_name_test], SOLUTIONS_TRAIN[system_name])
            errors_train_test[1, i, j] = metrics[metric](WEIGHTS_TEST[system_name_test], WEIGHTS_TRAIN[system_name])
            errors_train_test[2, i, j] = metrics[metric](DYDX_TEST[system_name_test], DYDX_TRAIN[system_name])
            errors_train_test[3, i, j] = metrics[metric](DYDW_TEST[system_name_test], DYDW_TRAIN[system_name])
    return errors_pre_gt, errors_train_test


cosine_similarity = lambda a, b: np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b))
metrics = {'L2': mean_squared_error,
           'cosine_similarity': cosine_similarity
           }

for epoch_values in [100, 200, 500]:
    PRETRAINED_SOLUTIONS = pretrain_methods(DIFFEQS_TRAIN, DIFFEQS_TEST, epoch_values)
    errors_pre_gt, errors_train_test = metric_evaluation(PRETRAINED_SOLUTIONS, 'L2')

    fig, ax = plt.subplots(5, len(DIFFEQS_TEST.keys()), figsize=(25, 25),sharex=True)
    fig.suptitle(f'L2_{epoch_values}')
    for i in range(len(DIFFEQS_TEST.keys())):
        ax[0, i].set_title(list(DIFFEQS_TEST.keys())[i])
        plt.setp(ax[4, i].xaxis.get_majorticklabels(), rotation=90)
        # plt.setp(ax[1, i].xaxis.get_majorticklabels(), rotation=45)

        ax[0, i].bar(DIFFEQS_TRAIN.keys(), errors_pre_gt[i, :])
        ax[1, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[0, i, :])
        ax[2, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[1, i, :])
        ax[3, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[2, i, :])
        ax[4, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[3, i, :])
        ax[0, i].set_yscale('log')
        ax[1, i].set_yscale('log')
        ax[2, i].set_yscale('log')
        ax[3, i].set_yscale('log')
        ax[4, i].set_yscale('log')

        ax[0, i].set_ylabel('pretrained model errors on quer diffeq')
        ax[1, i].set_ylabel('train-test solution error')
        ax[2, i].set_ylabel('train-test weights error')
        ax[3, i].set_ylabel('train-test dydx error')
        ax[4, i].set_ylabel('train-test dydw error')

        # plt.show()
    plt.tight_layout()
    plt.savefig(f'L2_phase_{epoch_values}.pdf')

    errors_pre_gt, errors_train_test = metric_evaluation(PRETRAINED_SOLUTIONS, 'cosine_similarity')

    fig, ax = plt.subplots(5, len(DIFFEQS_TEST.keys()), figsize=(25, 25), sharex=True)
    fig.suptitle(f'cosine_sim_{epoch_values}')
    for i in range(len(DIFFEQS_TEST.keys())):
        ax[0, i].set_title(list(DIFFEQS_TEST.keys())[i])
        plt.setp(ax[4, i].xaxis.get_majorticklabels(), rotation=90)

        ax[0, i].bar(DIFFEQS_TRAIN.keys(), errors_pre_gt[i, :])
        ax[1, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[0, i, :])
        ax[2, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[1, i, :])
        ax[3, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[2, i, :])
        ax[4, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[3, i, :])
        ax[0, i].set_yscale('log')
        # ax[1, i].set_yscale('log')
        # ax[2, i].set_yscale('log')
        # ax[3, i].set_yscale('log')
        # ax[4, i].set_yscale('log')

        ax[0, i].set_ylabel('pretrained model errors on quer diffeq')
        ax[1, i].set_ylabel('train-test solution error')
        ax[2, i].set_ylabel('train-test weights error')
        ax[3, i].set_ylabel('train-test dydx error')
        ax[4, i].set_ylabel('train-test dydw error')

    plt.tight_layout()
    plt.savefig(f'cosine_phase_sim_{epoch_values}.pdf')

#
#
# BURNIN = 1
# eps = [i for i in range(BURNIN,MAX_EPOCHS) if (i%SUB_RATE==0)]
# equations = list(DIFFEQS_TRAIN.keys())
#
#
# dydx = np.zeros((len(eps),len(equations)))
# dydw= np.zeros((len(eps),len(equations)))
# wdots= np.zeros((len(eps),len(equations)))
# sols = np.zeros((len(eps),len(equations)))
#
# for eqs_test in list(DIFFEQS_TEST.keys()):
#     for meth_dex in range(len(equations)):
#         for ep_index,ep in enumerate(eps):
#             param_a = np.load(f'data/{equations[meth_dex]}_train_w/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_gt_test_w/{ep}.npy')
#             wdots[ep_index,meth_dex]=np.dot(np.sort(param_a),np.sort(param_b))/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))
#
#             param_a = np.load(f'data/{equations[meth_dex]}_train_solution/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_gt_test_solution/{ep}.npy')
#             sols[ep_index,meth_dex]=np.dot(param_a.ravel(),param_b.ravel())/(np.linalg.norm(param_a.ravel(),ord=2)*np.linalg.norm(param_b.ravel(),ord=2))
#
#             param_a = np.load(f'data/{equations[meth_dex]}_train_dydw/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_gt_test_dydw/{ep}.npy')
#             dydw[ep_index,meth_dex]=np.dot(np.sort(param_a),np.sort(param_b))/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))
#
#             param_a = np.load(f'data/{equations[meth_dex]}_train_dydx/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_gt_test_dydx/{ep}.npy')
#             dydx[ep_index, meth_dex] = np.dot(param_a.ravel(), param_b.ravel()) / (np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))
#
#     fig,ax = plt.subplots(1,4,figsize=(15,10))
#     fig.suptitle(eqs_test)
#     for i in range(len(equations)):
#         ax[0].plot(eps,wdots[:,i],label=equations[i])
#         ax[1].plot(eps,dydw[:,i],label=equations[i])
#         ax[2].plot(eps,dydx[:,i],label=equations[i])
#         ax[3].plot(eps, sols[:, i], label=equations[i])
#     ax[0].set_title('Weights cosine similarity')
#     ax[1].set_title(r'$\frac{dy}{dw}$ Cosine Similarity')
#     ax[2].set_title(r'$\frac{dy}{dx}$ Cosine Similarity')
#     ax[3].set_title('Solutions cosine similarity')
#     plt.legend()
#     # plt.show()
#     plt.savefig(f'{eqs_test}_test_v_keys.pdf')
