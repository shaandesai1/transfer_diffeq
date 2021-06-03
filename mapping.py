"""
first_order to train a group of keys
"""

# from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import numpy as np
import torch
from neurodiffeq import diff  # the differentiation operation
from neurodiffeq.conditions import IVP  # the initial condition
from neurodiffeq.networks import FCNN  # fully-connect neural network
from neurodiffeq.solvers import Solver1D
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from neurodiffeq.callbacks import PeriodLocal
from neurodiffeq.callbacks import WeightCallback,WeightCallback1, SolutionCallback, SaddleCallback

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
    MODELS = {}
    for system_name, system in systems_dict.items():
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

        NETS = [FCNN(n_input_units=1, n_output_units=1, hidden_units=[10])]
        solver = Solver1D(
            ode_system=system,
            conditions=INIT_VAL,
            t_min=T_MIN,
            t_max=T_MAX,
            nets=NETS
        )
        solver.fit(max_epochs=epochs,callbacks=[cb,cb0,cb1,cb2])
        solution = solver.get_solution(best=False)(ts.reshape(-1, 1), to_numpy=True)
        SOLUTIONS[system_name] = solution
        MODELS[system_name] = solver.nets[0]
        print(f"***{system_type}_{system_name}_trained***")

    return SOLUTIONS, MODELS


# corpus of training diffeqs
DIFFEQS_TRAIN = {
    'exp': lambda u, t: [diff(u, t) + u],
    'exp1': lambda u, t: [diff(u, t) - u],
    'tanh': lambda u, t: [diff(u, t) + u ** 2 - 1],
    'sigmoid': lambda u, t: [diff(u, t) + u ** 2 - 1],
    'newt_cool': lambda u, t: [diff(u, t) - 3 * (5 - u)],
    'cubic': lambda u, t: [diff(u, t) - t ** 2 - 1],
    # 'r1': lambda u,t:[diff(u,t)-u-t*torch.exp(t)],
    # 'r2': lambda u,t:[diff(u,t)-2*u-t],
    'baseline': lambda u, t: [diff(u, t)]
}

DIFFEQS_TEST = {
    'exp': lambda u, t: [diff(u, t) + u],
    'exp1': lambda u, t: [diff(u, t) - u],
    'tanh': lambda u, t: [diff(u, t) + u ** 2 - 1],
    'sigmoid': lambda u, t: [diff(u, t) + u ** 2 - 1],
    'newt_cool': lambda u, t: [diff(u, t) - 3 * (5 - u)],
    'cubic': lambda u, t: [diff(u, t) - t ** 2 - 1],
    # 'r1': lambda u,t:[diff(u,t)-u-t*torch.exp(t)],
    # 'r2': lambda u,t:[diff(u,t)-2*u-t],
    'baseline': lambda u, t: [diff(u, t)]
}

# train all the keys (MAXEPS)
MAX_EPOCHS = 3000
MAX_EPOCHS_TEST = 300

SOLUTIONS_TRAIN, _ = train_methods(DIFFEQS_TRAIN, 'train', MAX_EPOCHS)
SOLUTIONS_TEST, _ = train_methods(DIFFEQS_TEST, 'test', MAX_EPOCHS)

# option 0
# L2 loss
BURNIN = 1
eps = [i for i in range(BURNIN,MAX_EPOCHS) if i % SUB_RATE == 0]
equations = list(DIFFEQS_TRAIN.keys())

solution_df = np.zeros((4, 2, len(eps), len(equations)))

# for eqs_test in list(DIFFEQS_TEST.keys()):
#     for meth_dex in range(len(equations)):
#         for ep_index, ep in enumerate(eps):
#             param_a = np.load(f'data/{equations[meth_dex]}_train_w/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_test_w/{ep}.npy')
#             solution_df[0, 0, ep_index, meth_dex] = np.dot(np.sort(param_a), np.sort(param_b)) / (
#                     np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))
#             solution_df[0, 1, ep_index, meth_dex] = ((param_a - param_b) ** 2).mean()
#
#             param_a = np.load(f'data/{equations[meth_dex]}_train_solution/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_test_solution/{ep}.npy')
#             solution_df[1, 0, ep_index, meth_dex] = np.dot(param_a.ravel(), param_b.ravel()) / (
#                     np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))
#             solution_df[1, 1, ep_index, meth_dex] = ((param_a - param_b) ** 2).mean()
#
#             param_a = np.load(f'data/{equations[meth_dex]}_train_dydw/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_test_dydw/{ep}.npy')
#             solution_df[2, 0, ep_index, meth_dex] = np.dot(np.sort(param_a.ravel()), np.sort(param_b.ravel())) / (
#                     np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))
#             solution_df[2, 1, ep_index, meth_dex] = ((param_a - param_b) ** 2).mean()
#
#             param_a = np.load(f'data/{equations[meth_dex]}_train_dydx/{ep}.npy')
#             param_b = np.load(f'data/{eqs_test}_test_dydx/{ep}.npy')
#             solution_df[3, 0, ep_index, meth_dex] = np.dot(param_a.ravel(), param_b.ravel()) / (
#                     np.linalg.norm(param_a, ord=2) * np.linalg.norm(param_b, ord=2))
#             solution_df[3, 1, ep_index, meth_dex] = ((param_a - param_b) ** 2).mean()
#
#     fig, ax = plt.subplots(2, 4, figsize=(20, 12))
#     fig.suptitle(eqs_test)
#     for i in range(len(equations)):
#         for j in range(4):
#             ax[0, j].plot(eps, solution_df[j, 0, :, i], label=equations[i])
#             ax[1, j].plot(eps, solution_df[j, 1, :, i], label=equations[i])
#
#     ax[0, 0].set_title('Weights cosine similarity')
#     ax[0, 1].set_title(r'$\frac{dy}{dw}$ Cosine Similarity')
#     ax[0, 2].set_title(r'$\frac{dy}{dx}$ Cosine Similarity')
#     ax[0, 3].set_title('Solutions cosine similarity')
#
#     ax[1, 0].set_title('Weights L2')
#     ax[1, 1].set_title(r'$\frac{dy}{dw}$ L2')
#     ax[1, 2].set_title(r'$\frac{dy}{dx}$ L2')
#     ax[1, 3].set_title('Solutions L2')
#
#     plt.legend()
#     # plt.show()
#     plt.savefig(f'{eqs_test}_test_v_keys.pdf')



#compare test at 300 to train at 3000
meth_list=['w','solution','dydw','dydx']
for meth in meth_list:
    MAX_EPOCHS_TEST = 300

    X = []
    for sys_name in list(SOLUTIONS_TRAIN.keys()):
        X.append( np.load(f'data/{sys_name}_train_{meth}/{MAX_EPOCHS}.npy'))

    Xtest = []
    for sys_name in list(SOLUTIONS_TEST.keys()):
        Xtest.append( np.load(f'data/{sys_name}_test_{meth}/{MAX_EPOCHS_TEST}.npy'))
    #
    # # option 1: PCA
    X = np.concatenate(X)
    X = X.reshape(len(SOLUTIONS_TRAIN.keys()), -1)
    Xtest = np.concatenate(Xtest)
    Xtest = Xtest.reshape(len(SOLUTIONS_TEST.keys()), -1)
    pca = PCA(n_components=2)
    pca_model = pca.fit(X)
    pca_fit = pca_model.transform(X)
    pca_fit_test = pca_model.transform(Xtest)
    text_X = list(SOLUTIONS_TRAIN.keys())
    text_Xtest = list(SOLUTIONS_TEST.keys())
    # #
    fig, ax = plt.subplots()
    fig.suptitle(f'{meth}')
    ax.scatter(pca_fit[:, 0], pca_fit[:, 1], color='red', label=f'train_{MAX_EPOCHS}epochs')
    ax.scatter(pca_fit_test[:, 0], pca_fit_test[:, 1], color='blue', label=f'test_{MAX_EPOCHS_TEST}epochs')
    for i, txt in enumerate(text_X):
        ax.annotate(txt, (pca_fit[i, 0], pca_fit[i, 1]))
    for i, txt in enumerate(text_Xtest):
        ax.annotate(txt, (pca_fit_test[i, 0], pca_fit_test[i, 1]))

    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    plt.savefig(f'PCA_{meth}.jpg')
    # plt.show()
    #
    # #
    # # #option 2: SVD
    svd = TruncatedSVD(n_components=2)
    svd_model = svd.fit(X)
    svd_fit = svd_model.transform(X)
    svd_fit_test = svd_model.transform(Xtest)
    fig, ax = plt.subplots()
    fig.suptitle(f'{meth}')
    ax.scatter(svd_fit[:, 0], svd_fit[:, 1], color='red', label=f'train_{MAX_EPOCHS}epochs')
    ax.scatter(svd_fit_test[:, 0], svd_fit_test[:, 1], color='blue', label=f'test_{MAX_EPOCHS_TEST}epochs')
    for i, txt in enumerate(text_X):
        ax.annotate(txt, (svd_fit[i, 0], svd_fit[i, 1]))
    for i, txt in enumerate(text_Xtest):
        ax.annotate(txt, (svd_fit_test[i, 0], svd_fit_test[i, 1]))
    ax.set_xlabel('SVD1')
    ax.set_ylabel('SVD2')
    plt.savefig(f'SVD_{meth}.jpg')

    # plt.show()





fig, ax = plt.subplots(1,2,figsize=(15,10))
for i in range(10):
    SOLUTIONS_TRAIN, _ = train_methods(DIFFEQS_TRAIN, 'train', MAX_EPOCHS)
    X = []
    for sys_name in list(SOLUTIONS_TRAIN.keys()):
        X.append(SOLUTIONS_TRAIN[sys_name])
    # option 1: PCA
    X = np.concatenate(X)
    X = X.reshape(len(SOLUTIONS_TRAIN.keys()), -1)

    pca = PCA(n_components=2)
    pca_model = pca.fit(X)
    pca_fit = pca_model.transform(X)
    text_X = list(SOLUTIONS_TRAIN.keys())
    ax[0].scatter(pca_fit[:, 0], pca_fit[:, 1], color='red', label=f'train_{MAX_EPOCHS}epochs')
    for i, txt in enumerate(text_X):
        ax[0].annotate(txt, (pca_fit[i, 0], pca_fit[i, 1]))

    svd = TruncatedSVD(n_components=2)
    svd_model = svd.fit(X)
    svd_fit = svd_model.transform(X)
    ax[1].scatter(svd_fit[:, 0], svd_fit[:, 1], color='red', label=f'train_{MAX_EPOCHS}epochs')
    for i, txt in enumerate(text_X):
        ax[1].annotate(txt, (svd_fit[i, 0], svd_fit[i, 1]))

ax[0].set_xlabel('PCA1')
ax[0].set_ylabel('PCA2')
ax[1].set_xlabel('SVD1')
ax[1].set_ylabel('SVD2')
# plt.show()
plt.savefig('multipleruns')
