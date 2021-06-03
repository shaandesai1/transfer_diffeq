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
from sklearn.metrics.pairwise import cosine_similarity
import copy


MAX_EPOCHS = 3000
BURNIN = 0
SUB_RATE = 1

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
        # define the baseline network
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
        # call dydx
        cb2 = PeriodLocal(period=SUB_RATE)
        wcb2 = SaddleCallback(f'{system_name}_{system_type}', ts)
        cb2.set_action_callback(wcb2)

        solver = Solver1D(
            ode_system=system,
            conditions=INIT_VAL,
            t_min=T_MIN,
            t_max=T_MAX,
            nets = NETS
        )
        solver.fit(max_epochs=epochs,callbacks=[cb,cb0,cb1,cb2])
        solution = solver.get_solution(best=False)(ts.reshape(-1,1), to_numpy=True)
        #record final model and final solution and return it
        SOLUTIONS[system_name] = solution
        MODELS[system_name] = solver.nets[0]
        print(f"***{system_type}_{system_name}_trained***")

    return SOLUTIONS,MODELS

#corpus of training diffeqs
DIFFEQS_TRAIN = {
    't1': lambda u, t: [diff(u, t,ord=2) + u],
    't2': lambda u, t: [diff(u, t,ord=2) - u],
    't3': lambda u, t: [diff(u, t,ord=2) + u ** 2 - 1],
    't4': lambda u, t: [diff(u, t, ord=2) -u + u ** 2 - 1],
    'baseline': lambda u, t: [diff(u,t,ord=2)]
}

#train all the keys (MAXEPS)
SOLUTIONS_TRAIN,MODELS_TRAIN = train_methods(DIFFEQS_TRAIN,'train',MAX_EPOCHS)

DIFFEQS_TEST = {
    'q1': lambda u, t: [diff(u, t,ord=2) - u + u ** 2],
    'q2': lambda u, t: [diff(u, t,ord=2) - u + u ** 2 - u ** 3],
    # 'q3': lambda u, t: [diff(u, t) + u ** 2 + u ** 4],
    # 'q4': lambda u, t: [diff(u, t) - u ** 2 - u ** 4],
}

#train all the queries (500 epochs)
SOLUTIONS_TEST, _  = train_methods(DIFFEQS_TEST,'test',500)

#train all the queries (MAXEPS)
SOLUTIONS_GT_TEST, _  = train_methods(DIFFEQS_TEST,'gt_test',MAX_EPOCHS)


# PRE_MAX_EPOCHS = 500

def pretrain_methods(train_dict,test_dict,epochs):
    PRETRAINED_SOLUTIONS = {}
    for system_name, system in train_dict.items():
        for system_name_test, system_test in test_dict.items():
            solver = Solver1D(
                ode_system=system_test,
                conditions=INIT_VAL,
                t_min=T_MIN,
                t_max=T_MAX,
                nets=[copy.deepcopy(MODELS_TRAIN[system_name])]
            )
            solver.fit(max_epochs=epochs)
            solution = solver.get_solution(best=False)(ts, to_numpy=True)
            PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'] = solution

    return PRETRAINED_SOLUTIONS

def metric_evaluation(PRETRAINED_SOLUTIONS, metric):

    errors_pre_gt = np.zeros((len(DIFFEQS_TEST), len(DIFFEQS_TRAIN)))
    errors_train_test = np.zeros((len(DIFFEQS_TEST), len(DIFFEQS_TRAIN)))
    for j, (system_name, system) in enumerate(DIFFEQS_TRAIN.items()):
        for i, (system_name_test, system_test) in enumerate(DIFFEQS_TEST.items()):
            errors_pre_gt[i, j] = metrics[metric](PRETRAINED_SOLUTIONS[f'{system_name_test}_{system_name}'], SOLUTIONS_GT_TEST[
                system_name_test])
            errors_train_test[i, j] = metrics[metric](SOLUTIONS_TEST[system_name_test], SOLUTIONS_TRAIN[system_name])
    return errors_pre_gt,errors_train_test


metrics = {'L2':mean_squared_error,
           'cosine_similarity':cosine_similarity
           }

for epoch_values in [10,50,100,500,1000]:
    PRETRAINED_SOLUTIONS = pretrain_methods(DIFFEQS_TRAIN,DIFFEQS_TEST,epoch_values)
    errors_pre_gt,errors_train_test = metric_evaluation(PRETRAINED_SOLUTIONS,'L2')


    fig, ax = plt.subplots(2,len(DIFFEQS_TEST), figsize=(16, 10))
    fig.suptitle('L2')
    for i in range(len(DIFFEQS_TEST.keys())):
        ax[0,i].set_title(list(DIFFEQS_TEST.keys())[i])
        plt.setp(ax[0,i].xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax[1,i].xaxis.get_majorticklabels(), rotation=45)

        ax[0,i].set_yscale('log')
        ax[1,i].set_yscale('log')
        ax[0,i].bar(DIFFEQS_TRAIN.keys(), errors_pre_gt[i, :])
        ax[1,i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[i, :])
        ax[0,i].set_ylabel('pretrained model errors on quer diffeq')
        ax[1,i].set_ylabel('train-test error')
        plt.show()

    errors_pre_gt,errors_train_test = metric_evaluation(PRETRAINED_SOLUTIONS,'cosine_similarity')

    fig, ax = plt.subplots(2, len(DIFFEQS_TEST), figsize=(16, 10))
    fig.suptitle('cosine_sim')
    for i in range(len(DIFFEQS_TEST.keys())):
        ax[0, i].set_title(list(DIFFEQS_TEST.keys())[i])
        plt.setp(ax[0, i].xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax[1, i].xaxis.get_majorticklabels(), rotation=45)

        ax[0, i].set_yscale('log')
        ax[1, i].set_yscale('log')
        ax[0, i].bar(DIFFEQS_TRAIN.keys(), errors_pre_gt[i, :])
        ax[1, i].bar(DIFFEQS_TRAIN.keys(), errors_train_test[i, :])
        ax[0, i].set_ylabel('pretrained model errors on quer diffeq')
        ax[1, i].set_ylabel('train-test error')
        plt.show()



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
            dydx[ep_index, meth_dex] = np.dot(param_a, param_b) / (np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))

    fig,ax = plt.subplots(2,4,figsize=(15,10))
    fig.suptitle(eqs_test)
    for i in range(len(equations)):
        ax[0,0].plot(eps,wdots[:,i],label=equations[i])
        ax[0,1].plot(eps,dydw[:,i],label=equations[i])
        ax[1,0].plot(eps,dydx[:,i],label=equations[i])
        ax[1,1].plot(eps, sols[:, i], label=equations[i])
    ax[0,0].set_title('Weights cosine similarity')
    ax[0,1].set_title(r'$\frac{dy}{dw}$ Cosine Similarity')
    ax[1,0].set_title(r'$\frac{dy}{dx}$ Cosine Similarity')
    ax[1, 1].set_title('Solutions cosine similarity')
    plt.legend()
    plt.show()
