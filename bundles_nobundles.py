"""
bundles and no bundles pretraining and model identification
Author: Shaan Desai
"""


import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff     # the differentiation operation
from neurodiffeq.conditions import BundleIVP,IVP  # the initial condition
from neurodiffeq.solvers import BundleSolver1D,Solver1D  # the solver
from neurodiffeq.networks import FCNN  # fully-connect neural network


# Damped oscilator paramters
u_0_min = 0
u_0_max = 1
u_0_prime = 0
t_0 = 0
t_f = 5
ts = np.linspace(t_0, t_f, 500)

init_vals = [BundleIVP(t_0=t_0, bundle_conditions=['u_0'])]
MAX_EPOCHS = 5000
MAX_EPOCHS_TEST = 250
PRETRAIN_EPOCHS = 250

def train_bundles(systems_dict, system_type, epochs):
    MODELS = {}
    for system_name, system in systems_dict.items():
        solver = BundleSolver1D(
            ode_system=system,
            conditions=init_vals,
            t_min=t_0,
            t_max=t_f,
            theta_min=(u_0_min),
            theta_max=(u_0_max)
        )
        solver.fit(max_epochs=epochs)
        MODELS[system_name] = solver.nets[0]
        print(f'model {system_name} bundles done')
    return MODELS


def train_no_bundles(systems_dict, system_type, epochs):
    MODELS = {}
    SOLUTIONS = {}
    for system_name, (system, INIT_VAL) in systems_dict.items():
        NETS = [FCNN(n_input_units=1, n_output_units=1, hidden_units=[30, 30])]
        solver = Solver1D(
            ode_system=system,
            conditions=INIT_VAL,
            t_min=t_0,
            t_max=t_f,
            nets=NETS
        )
        solver.fit(max_epochs=epochs)
        solution = solver.get_solution(best=False)(ts.reshape(-1, 1), to_numpy=True)
        SOLUTIONS[system_name] = solution.ravel()
        MODELS[system_name] = solver.nets[0]
        print(f'model {system_name} no bundles done')
    return SOLUTIONS, MODELS

def bundle_pretrain(systems_dict):
    TRAINED_ICS = {}
    for train_model_name, train_model in DIFFEQS_BUNDLES_TRAIN.items():
        for system_name, (system, INIT_VALS) in systems_dict.items():
            solver = BundleSolver1D(
                ode_system=train_model,
                conditions=init_vals,
                t_min=t_0,
                t_max=t_f,
                theta_min=(u_0_min),
                # The order must always be the conditions first and in the same order as in bundle_conditions
                theta_max=(u_0_max),
                nets=[MODELS_TRAIN[train_model_name]]
                # The order must always be the conditions first and in the same order as in bundle_conditions
            )
            solver.fit(max_epochs=1)
            solution = solver.get_solution()
            x1_net = solution(ts, INIT_VALS[0].u_0 * np.ones(len(ts)), to_numpy=True)
            TRAINED_ICS[f'{train_model_name}_{system_name}'] = x1_net
    return TRAINED_ICS

def no_bundle_pretrain(train_dict, test_dict, epochs,nets=True):
    PRETRAINED_SOLUTIONS = {}
    for system_name_train, (system_train, INIT_VALS_train) in train_dict.items():
        for system_name_test, (system_test, INIT_VALS_test) in test_dict.items():
            solver = Solver1D(
                ode_system=system_test,
                conditions=INIT_VALS_test,
                t_min=t_0,
                t_max=t_f,
                nets=DIFFEQS_NO_BUNDLES_TRAIN[system_name_train]
            )
            solver.fit(max_epochs=epochs)
            solution = solver.get_solution(best=False)(ts.reshape(-1, 1), to_numpy=True)
            PRETRAINED_SOLUTIONS[f'{system_name_train}_{system_name_test}'] = solution
    return PRETRAINED_SOLUTIONS


DIFFEQS_BUNDLES_TRAIN = {
    'exp': lambda u, t, u_0: [diff(u, t) + u],
    'exp1': lambda u, t, u_0: [diff(u, t) - u],
    'tanh': lambda u, t, u_0: [diff(u, t) + u ** 2 - 1],
    'sigmoid': lambda u, t, u_0: [diff(u, t) + u ** 2 - u],
    'newt_cool': lambda u, t, u_0: [diff(u, t) - 3 * (5 - u)],
    'cubic': lambda u, t, u_0: [diff(u, t) - t ** 2 - 1],
    'baseline': lambda u, t, u_0: [diff(u, t)],
}

u_0_inits = np.random.rand(7, 5)

DIFFEQS_NO_BUNDLES_TRAIN = {
    'exp_a': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 0])]],
    'exp_e': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 4])]],
    'exp1_a': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 0])]],
    'exp1_e': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 4])]],
    'tanh_a': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 0])]],
    'tanh_e': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 4])]],
    'sigmoid_a': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 0])]],
    'sigmoid_e': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 4])]],
    'newt_cool_a': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 0])]],
    'newt_cool_e': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 4])]],
    'cubic_a': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 0])]],
    'cubic_e': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 4])]],
    'baseline_a': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 0])]],
    'baseline_e': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 4])]],
}


DIFFEQS_TEST = {
    'exp_a': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 0])]],
    'exp_b': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 1])]],
    'exp_c': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 2])]],
    'exp_d': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 3])]],
    'exp_e': [lambda u, t: [diff(u, t) + u], [IVP(t_0=0.0, u_0=u_0_inits[0, 4])]],

    'exp1_a': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 0])]],
    'exp1_b': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 1])]],
    'exp1_c': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 2])]],
    'exp1_d': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 3])]],
    'exp1_e': [lambda u, t: [diff(u, t) - u], [IVP(t_0=0.0, u_0=u_0_inits[1, 4])]],

    'tanh_a': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 0])]],
    'tanh_b': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 1])]],
    'tanh_c': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 2])]],
    'tanh_d': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 3])]],
    'tanh_e': [lambda u, t: [diff(u, t) + u ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[2, 4])]],

    'sigmoid_a': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 0])]],
    'sigmoid_b': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 1])]],
    'sigmoid_c': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 2])]],
    'sigmoid_d': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 3])]],
    'sigmoid_e': [lambda u, t: [diff(u, t) + u ** 2 - u], [IVP(t_0=0.0, u_0=u_0_inits[3, 4])]],

    'newt_cool_a': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 0])]],
    'newt_cool_b': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 1])]],
    'newt_cool_c': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 2])]],
    'newt_cool_d': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 3])]],
    'newt_cool_e': [lambda u, t: [diff(u, t) - 3 * (5 - u)], [IVP(t_0=0.0, u_0=u_0_inits[4, 4])]],

    'cubic_a': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 0])]],
    'cubic_b': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 1])]],
    'cubic_c': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 2])]],
    'cubic_d': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 3])]],
    'cubic_e': [lambda u, t: [diff(u, t) - t ** 2 - 1], [IVP(t_0=0.0, u_0=u_0_inits[5, 4])]],

    'baseline_a': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 0])]],
    'baseline_b': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 1])]],
    'baseline_c': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 2])]],
    'baseline_d': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 3])]],
    'baseline_e': [lambda u, t: [diff(u, t)], [IVP(t_0=0.0, u_0=u_0_inits[6, 4])]],

}


ts_np = ts
c1 = .5 * np.log((u_0_inits[2, 0] - 1) / (-u_0_inits[2, 0] - 1))
c2 = .5 * np.log((u_0_inits[2, 1] - 1) / (-u_0_inits[2, 1] - 1))
c3 = .5 * np.log((u_0_inits[2, 2] - 1) / (-u_0_inits[2, 2] - 1))
c4 = .5 * np.log((u_0_inits[2, 3] - 1) / (-u_0_inits[2, 3] - 1))
c5 = .5 * np.log((u_0_inits[2, 4] - 1) / (-u_0_inits[2, 4] - 1))

d1 = (1 - u_0_inits[3, 0]) / u_0_inits[3, 0]
d2 = (1 - u_0_inits[3, 1]) / u_0_inits[3, 1]
d3 = (1 - u_0_inits[3, 2]) / u_0_inits[3, 2]
d4 = (1 - u_0_inits[3, 3]) / u_0_inits[3, 3]
d5 = (1 - u_0_inits[3, 4]) / u_0_inits[3, 4]

SOLUTIONS_GT_TEST = {
    'exp_a': u_0_inits[0, 0] * np.exp(-ts_np),
    'exp_b': u_0_inits[0, 1] * np.exp(-ts_np),
    'exp_c': u_0_inits[0, 2] * np.exp(-ts_np),
    'exp_d': u_0_inits[0, 3] * np.exp(-ts_np),
    'exp_e': u_0_inits[0, 4] * np.exp(-ts_np),

    'exp1_a': u_0_inits[1, 0] * np.exp(ts_np),
    'exp1_b': u_0_inits[1, 1] * np.exp(ts_np),
    'exp1_c': u_0_inits[1, 2] * np.exp(ts_np),
    'exp1_d': u_0_inits[1, 3] * np.exp(ts_np),
    'exp1_e': u_0_inits[1, 4] * np.exp(ts_np),

    'tanh_a': (np.exp(2 * ts_np) - np.exp(2 * c1)) / (np.exp(2 * ts_np) + np.exp(2 * c1)),
    'tanh_b': (np.exp(2 * ts_np) - np.exp(2 * c2)) / (np.exp(2 * ts_np) + np.exp(2 * c2)),
    'tanh_c': (np.exp(2 * ts_np) - np.exp(2 * c3)) / (np.exp(2 * ts_np) + np.exp(2 * c3)),
    'tanh_d': (np.exp(2 * ts_np) - np.exp(2 * c4)) / (np.exp(2 * ts_np) + np.exp(2 * c4)),
    'tanh_e': (np.exp(2 * ts_np) - np.exp(2 * c5)) / (np.exp(2 * ts_np) + np.exp(2 * c5)),

    'sigmoid_a': (np.exp(ts_np)) / (d1 + np.exp(ts_np)),
    'sigmoid_b': (np.exp(ts_np)) / (d2 + np.exp(ts_np)),
    'sigmoid_c': (np.exp(ts_np)) / (d3 + np.exp(ts_np)),
    'sigmoid_d': (np.exp(ts_np)) / (d4 + np.exp(ts_np)),
    'sigmoid_e': (np.exp(ts_np)) / (d5 + np.exp(ts_np)),

    'newt_cool_a': u_0_inits[4, 0] * np.exp(-3 * ts_np) + 5,
    'newt_cool_b': u_0_inits[4, 1] * np.exp(-3 * ts_np) + 5,
    'newt_cool_c': u_0_inits[4, 2] * np.exp(-3 * ts_np) + 5,
    'newt_cool_d': u_0_inits[4, 3] * np.exp(-3 * ts_np) + 5,
    'newt_cool_e': u_0_inits[4, 4] * np.exp(-3 * ts_np) + 5,

    'cubic_a': u_0_inits[5, 0] + ts_np ** 3 / 3 + ts_np,
    'cubic_b': u_0_inits[5, 1] + ts_np ** 3 / 3 + ts_np,
    'cubic_c': u_0_inits[5, 2] + ts_np ** 3 / 3 + ts_np,
    'cubic_d': u_0_inits[5, 3] + ts_np ** 3 / 3 + ts_np,
    'cubic_e': u_0_inits[5, 4] + ts_np ** 3 / 3 + ts_np,

    'baseline_a': u_0_inits[6, 0] * np.ones(len(ts_np)),
    'baseline_b': u_0_inits[6, 1] * np.ones(len(ts_np)),
    'baseline_c': u_0_inits[6, 2] * np.ones(len(ts_np)),
    'baseline_d': u_0_inits[6, 3] * np.ones(len(ts_np)),
    'baseline_e': u_0_inits[6, 4] * np.ones(len(ts_np)),

}


MODELS_TRAIN_BUNDLES = train_bundles(DIFFEQS_BUNDLES_TRAIN, 'train', MAX_EPOCHS)
SOLUTIONS_TRAIN_NO_BUNDLES, MODELS_TRAIN_NO_BUNDLES = train_no_bundles(DIFFEQS_NO_BUNDLES_TRAIN, 'train', MAX_EPOCHS)
SOLUTIONS_TEST,MODELS_TEST_NO_BUNDLES = train_no_bundles(DIFFEQS_TEST, 'test', MAX_EPOCHS_TEST)
PRETRAINED_BUNDLES_SOLUTIONS = bundle_pretrain(DIFFEQS_TEST)
PRETRAINED_NO_BUNDLES_SOLUTIONS = no_bundle_pretrain(DIFFEQS_NO_BUNDLES_TRAIN,DIFFEQS_TEST,PRETRAIN_EPOCHS)


def metric_evaluation(metric):
    errors_pre_gt = np.zeros((len(DIFFEQS_TEST), len(DIFFEQS_BUNDLES_TRAIN)+len(DIFFEQS_NO_BUNDLES_TRAIN)))
    errors_train_test = np.zeros((4, len(DIFFEQS_TEST), len(DIFFEQS_BUNDLES_TRAIN)))
    for j, (system_name, system) in enumerate(DIFFEQS_BUNDLES_TRAIN.items()):
        for i, (system_name_test, system_test) in enumerate(DIFFEQS_TEST.items()):
            errors_pre_gt[i, j] = metrics[metric](PRETRAINED_BUNDLES_SOLUTIONS[f'{system_name}_{system_name_test}'],
                                                  SOLUTIONS_GT_TEST[system_name_test])
            errors_train_test[0, i, j] = metrics[metric](SOLUTIONS_TEST[system_name_test],
                                                         PRETRAINED_BUNDLES_SOLUTIONS[f'{system_name}_{system_name_test}'])

    for j, (system_name, system) in enumerate(DIFFEQS_BUNDLES_TRAIN.items()):
        for i, (system_name_test, system_test) in enumerate(DIFFEQS_TEST.items()):
            errors_pre_gt[i, j+len(DIFFEQS_BUNDLES_TRAIN)] = metrics[metric](PRETRAINED_NO_BUNDLES_SOLUTIONS[f'{system_name}_{system_name_test}'],
                                                  SOLUTIONS_GT_TEST[system_name_test])
            errors_train_test[0, i, j+len(DIFFEQS_BUNDLES_TRAIN)] = metrics[metric](SOLUTIONS_TEST[system_name_test],
                                                         PRETRAINED_NO_BUNDLES_SOLUTIONS[f'{system_name}_{system_name_test}'])

    return errors_pre_gt, errors_train_test


mean_squared_error = lambda a, b: ((a - b) ** 2).mean()

metrics = {'L2': mean_squared_error,
           'cosine_similarity': cosine_similarity
           }

errors_pre_gt, errors_train_test = metric_evaluation('L2')

newres = np.zeros(errors_pre_gt.shape)
for i in range(newres.shape[0]):
    indx = errors_train_test[j,i,:].argsort()[:2]#np.argmin(errors_pre_gt[i])
    newres[i, indx[0]] = 1
    newres[i, indx[1]] = 2

newres1 = np.zeros(errors_train_test.shape)
for j in range(newres1.shape[0]):
    for i in range(newres1.shape[1]):
        indx = errors_train_test[j,i,:].argsort()[:2]#np.argmin(errors_train_test[j,i,:])
        newres1[j,i, indx[0]] = 1
        newres1[j, i, indx[1]] = 2

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].imshow(newres)
ax[1].imshow(newres1[0, :])
# ax[2].spy(newres1[1,:])
# ax[3].spy(newres1[2,:])
# ax[4].spy(newres1[3, :])

ax[0].set_title('Best pretraining model')
ax[0].set_xticks(np.arange(len(DIFFEQS_TRAIN)))
ax[0].set_xticklabels(list(DIFFEQS_TRAIN.keys()), rotation=90)
ax[0].set_yticks(np.arange(len(DIFFEQS_TEST)))
ax[0].set_yticklabels(list(DIFFEQS_TEST.keys()))

ax[1].set_title('train-test solution')
ax[1].set_xticks(np.arange(len(DIFFEQS_TRAIN)))
ax[1].set_xticklabels(list(DIFFEQS_TRAIN.keys()), rotation=90)
ax[1].set_yticks(np.arange(len(DIFFEQS_TEST)))
ax[1].set_yticklabels(list(DIFFEQS_TEST.keys()))

# ax[2].set_title('train-test solution')
# ax[2].set_xticks(np.arange(len(DIFFEQS_TRAIN)))
# ax[2].set_xticklabels(list(DIFFEQS_TRAIN.keys()),rotation=90)
# ax[2].set_yticks(np.arange(len(DIFFEQS_TEST)))
# ax[2].set_yticklabels(list(DIFFEQS_TEST.keys()))

# ax[3].set_title('train-test dydx')
# ax[3].set_xticks(np.arange(len(DIFFEQS_TRAIN)))
# ax[3].set_xticklabels(list(DIFFEQS_TRAIN.keys()),rotation=90)
# ax[3].set_yticks(np.arange(len(DIFFEQS_TEST)))
# ax[3].set_yticklabels(list(DIFFEQS_TEST.keys()))

# ax[4].set_title('train-test (solution,dydx) together')
# ax[4].set_xticks(np.arange(len(DIFFEQS_TRAIN)))
# ax[4].set_xticklabels(list(DIFFEQS_TRAIN.keys()), rotation=90)
# ax[4].set_yticks(np.arange(len(DIFFEQS_TEST)))
# ax[4].set_yticklabels(list(DIFFEQS_TEST.keys()))

plt.savefig('bundle_nobundle.pdf')
# plt.xlabel('pretraining model')
# plt.ylabel('test diffeq')
# plt.show()

