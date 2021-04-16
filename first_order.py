"""
first_order to train a group of keys
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy


from neurodiffeq import diff  # the differentiation operation
from neurodiffeq.conditions import IVP  # the initial condition
from neurodiffeq.networks import FCNN  # fully-connect neural network
from neurodiffeq.networks import SinActv  # sin activation
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.solvers import Solver1D
from neurodiffeq.callbacks import MonitorCallback
from neurodiffeq.callbacks import WeightCallback
from neurodiffeq.callbacks import WeightCallback1, WeightCallback2, SolutionCallback
from neurodiffeq.callbacks import PeriodLocal
import copy

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

MAX_EPOCHS = 3000
BURNIN = 0
SUB_RATE = 1

mdl_weight_collector = []

# fixed initial conditions
INIT_VAL = [IVP(t_0=0.0, u_0=0.5)]
T_MIN = -2
T_MAX = 2
ts = torch.linspace(T_MIN, T_MAX, 100).reshape(-1, 1)

def train_methods(systems_dict,system_type,epochs):
    """
    take the systems dictionary and return the final solution and final weights for each
    args:
        systems_dict: the dictionary of diffeqs e.g. exp:[lambda u,t:diff(u,t) -u]
    """

    SOLUTIONS = {}
    MODELS = {}
    for system_name, system in systems_dict.items():
        NETS = [FCNN(n_input_units=1, n_output_units=1, hidden_units=[5])]
        # call the dy/dweights (integrated over the inputs)
        cb = PeriodLocal(period=SUB_RATE)
        wcb = WeightCallback(f'{system_name}_{system_type}', ts)
        cb.set_action_callback(wcb)
        # call the weights
        cb0 = PeriodLocal(period=SUB_RATE)
        wcb0 = WeightCallback1(f'{system_name}_{system_type}', ts)
        cb0.set_action_callback(wcb0)
        # call the solutions
        cb1 = PeriodLocal(period=SUB_RATE)
        wcb1 = SolutionCallback(f'{system_name}_{system_type}', ts)
        cb1.set_action_callback(wcb1)

        solver = Solver1D(
            ode_system=system,
            conditions=INIT_VAL,
            t_min=T_MIN,
            t_max=T_MAX,
            nets = NETS
        )
        solver.fit(max_epochs=epochs)
        solution = solver.get_solution(best=False)(ts.reshape(-1,1), to_numpy=True)
        #record final model and final solution and return it
        SOLUTIONS[system_name] = solution
        MODELS[system_name] = solver.nets[0]

    return SOLUTIONS,MODELS

#corpus of training diffeqs
DIFFEQS_TRAIN = {
    'exp': lambda u, t: [diff(u, t) + u],
    # 'exp1': lambda u, t: [diff(u, t) - u],
    # 'tanh': lambda u, t: [diff(u, t) + u ** 2 - 1],
    # 'psig': lambda u, t: [diff(u, t) - 3 * u + u ** 2],
    # 'r1': lambda u, t: [diff(u, t) - u + u ** 2 + u ** 3],
    # 'r2': lambda u, t: [diff(u, t) + u + u ** 2],
    # 'r3': lambda u, t: [diff(u, t) + u ** 2],
    # 'r4': lambda u, t: [diff(u, t) - u ** 2],
    # 'q1': lambda u, t: [diff(u, t) - u + u ** 2],
    # 'q2': lambda u, t: [diff(u, t) - u + u ** 2 - u ** 3],
    # 'q3': lambda u, t: [diff(u, t) + u ** 2 + u ** 4],
    # 'q4': lambda u, t: [diff(u, t) - u ** 2 - u ** 4],
    'high_order1': lambda u, t: [diff(u, t) + u - u ** 2 + u ** 3 - u ** 4 + u ** 5],
    'high_order2': lambda u, t: [diff(u, t) - u + u ** 2 - u ** 3 + u ** 4 - u ** 5],
    'baseline': lambda u, t: [diff(u,t)]
}

SOLUTIONS_TRAIN,MODELS_TRAIN = train_methods(DIFFEQS_TRAIN,'train',MAX_EPOCHS)

DIFFEQS_TEST = {
    'q1': lambda u, t: [diff(u, t) - u + u ** 2],
    # 'q2': lambda u, t: [diff(u, t) - u + u ** 2 - u ** 3],
    # 'q3': lambda u, t: [diff(u, t) + u ** 2 + u ** 4],
    # 'q4': lambda u, t: [diff(u, t) - u ** 2 - u ** 4],
}

SOLUTIONS_TEST, MODEL_TEST  = train_methods(DIFFEQS_TEST,'test',500)

SOLUTIONS_GT_TEST, MODEL_GT_TEST  = train_methods(DIFFEQS_TEST,'gt_test',MAX_EPOCHS)


PRE_MAX_EPOCHS = 500
PRETRAINED_SOLUTIONS = {}
for system_name, system in DIFFEQS_TRAIN.items():
    for system_name_test, system_test in DIFFEQS_TEST.items():
        solver = Solver1D(
            ode_system=system_test,
            conditions=INIT_VAL,
            t_min=T_MIN,
            t_max=T_MAX,
            nets=[copy.deepcopy(MODEL_TRAIN[system_name])]
        )
        solver.fit(max_epochs=PRE_MAX_EPOCHS)
        solution = solver.get_solution(best=False)(ts, to_numpy=True)
        PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'] = solution

# %%
def metric_evaluation(metric):
    metrics = {'L2':mean_squared_error,
               'cosine_similarity':cosine_similarity
              }
    if metric in metrics.keys():

        errors_pre_gt = np.zeros((len(DIFFEQS_TEST), len(DIFFEQS_TRAIN)))
        errors_train_test = np.zeros((len(DIFFEQS_TEST), len(DIFFEQS_TRAIN)))
        for j, (system_name, system) in enumerate(DIFFEQS_TRAIN.items()):
            for i, (system_name_test, system_test) in enumerate(DIFFEQS_TEST.items()):
                errors_pre_gt[i, j] = metrics[metric](PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'], SOLUTIONS_GT_TEST[
                    system_name_test])
                errors_train_test[i, j] = metrics[metric](SOLUTIONS_TEST[system_name_test], SOLUTIONS_TRAIN[system_name])
        return errors_pre_gt,errors_train_test
    else:
        raise ValueError('metric not defined')

errors_pre_gt,errors_train_test = metric_evaluation('L2')

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
for i in range(len(DIFFEQS_TEST.keys())):
    fig.suptitle(list(DIFFEQS_TEST.keys())[i])
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].bar(DIFFEQS_TRAIN.keys(), errors_pre_gt[i, :])
    ax[1].bar(DIFFEQS_TRAIN.keys(), errors_train_test[i, :])
    ax[0].set_title('pretrained model errors on quer diffeq')
    ax[1].set_title('train-test error')


dydw= np.zeros((len(eps),len(equations)))
wdots= np.zeros((len(eps),len(equations)))
sols = np.zeros((len(eps),len(equations)))

# MAX_EPOCHS = 3000
BURNIN = 1
equations = list(DIFFEQS_TRAIN.keys())
eps = [i for i in range(BURNIN,MAX_EPOCHS) if (i%SUB_RATE==0)]

for eqs_test in list(DIFFEQS_TEST.keys()):
    for meth_dex in range(len(equations)):
        for ep_index,ep in enumerate(eps):
            param_a = np.load(f'data/{equations[meth_dex]}_w/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_w/{ep}.npy')
            wdots[ep_index,meth_dex]=np.dot(param_a,param_b)/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))

            param_a = np.load(f'data/{equations[meth_dex]}_solution/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_solution/{ep}.npy')
            sols[ep_index,meth_dex]=np.dot(param_a,param_b)/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))

            param_a = np.load(f'data/{equations[meth_dex]}_dydw/{ep}.npy')
            param_b = np.load(f'data/{eqs_test}_dydw/{ep}.npy')
            dydw[ep_index,meth_dex]=np.dot(param_a,param_b)/(np.linalg.norm(param_a,ord=2)*np.linalg.norm(param_b,ord=2))

    fig,ax = plt.subplots(1,3,figsize=(15,10))
    fig.suptitle(eqs_test)
    for i in range(len(equations)):
        ax[0].plot(eps,dydw[:,i],label=equations[i])
        ax[1].plot(eps,wdots[:,i],label=equations[i])
        ax[2].plot(eps,sols[:,i],label=equations[i])
    ax[0].set_title(r'$\frac{dy}{dw}$ Cosine Similarity')
    ax[1].set_title('Weights Cosine Similarity')
    ax[2].set_title('Solutions Cosine Similarity')
    plt.legend()
