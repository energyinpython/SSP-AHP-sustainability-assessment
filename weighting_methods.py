import numpy as np
from correlations import *
from normalizations import *


# equal weighting
def equal_weighting(X):
    N = np.shape(X)[1]
    return np.ones(N) / N


# entropy weighting
def entropy_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    m, n = np.shape(pij)

    H = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            if pij[i, j] != 0:
                H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))

    return w


# standard deviation weighting
def std_weighting(X):
    stdv = np.std(X, axis = 0)
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    suma = np.sum(difference, axis = 0)
    C = std * suma
    w = C / (np.sum(C, axis = 0))
    return w


# Equal distribution of main weights on the hierarchical structure of the model criteria
def structured_equal_weights(modules, main_weights):
    flag_begin = True
    crit_list = []
    num_of_modules = len(modules)
    for g, module in enumerate(modules):
        num_of_submodules = len(module)
        for submodule in module:
            num_of_elements = len(submodule)
            subweights = np.ones(num_of_elements) * ((main_weights[g] / num_of_submodules) / num_of_elements)
            
            if flag_begin:
                old_subweights = copy.deepcopy(subweights)
                flag_begin = False
            else:
                old_subweights = np.hstack((old_subweights, subweights))
            for sub in submodule:
                crit_list.append(sub)
            
    return old_subweights, crit_list
