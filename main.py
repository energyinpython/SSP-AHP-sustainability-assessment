import numpy as np
import pandas as pd
import copy
import os
import sys

import matplotlib
from itertools import product

from PyPDF2 import PdfFileMerger

from normalizations import *
from visualizations import *
from weighting_methods import *

from ssp_ahp import AHP
from rank_preferences import *


def main():
    # Input data
    folder_name = './input'
    # Name of file with input data
    file_name = 'data_2019.csv'
    path_data = os.path.join(folder_name, file_name)
    data = pd.read_csv(path_data, index_col = 'Country')

    # Hierarchical model criteria tree
    modules = [
        [
            ['C1', 'C2'],
            ['C3', 'C4'],
            ['C5', 'C6', 'C7']
        ],
        [
            ['C8', 'C9', 'C10', 'C11'],
            ['C12'],
            ['C13', 'C14']
        ],
        [
            ['C15'],
            ['C16', 'C17']
        ],
        [
            ['C18'],
            ['C19', 'C20']
        ],
        [
            ['C21', 'C22'],
            ['C23', 'C24'],
            ['C25']
        ]
    ]

    modules_indexes = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
        [20, 21, 22, 23, 24]
    ]

    # matrix with pairwise comparison criteria based on Saaty scale - weights provided by the domain expert
    PCcriteria = np.array(([[1, 1, 5, 3, 9],
    [1, 1, 3, 5, 7],
    [1/5, 1/3, 1, 1, 9],
    [1/3, 1/5, 1, 1, 7],
    [1/9, 1/7, 1/9, 1/7, 1]
    ]))

    # End of input data

    
    df_data = data.iloc[:len(data) - 1, :]
    df_types = data.iloc[len(data) - 1, :]
    types = df_types.to_numpy()

    list_alt_names = [r'$A_{' + str(i + 1) + '}$' for i in range(0, len(df_data))]

    matrix = df_data.to_numpy()

    # calculation of main weights with AHP weighting method
    ahp = AHP()
    ahp._check_consistency(PCcriteria)

    # calculation of the priority vector of criteria (criteria weights)
    # the normalized columns sum method
    # W = ahp._normalized_column_sum_method_classic(PCcriteria)

    # eigenvector method
    W = ahp._calculate_eigenvector(PCcriteria)

    G = [r'$G_{' + str(i + 1) + '}$' for i in range(PCcriteria.shape[1])]

    df_saaty_weights = pd.DataFrame(W.reshape(1, -1), index = ['Weights'], columns = G)
    df_saaty_weights.to_csv('output/ahp_weights.csv')
    # ----------------------------------------------------------------------------------wybor wag
    # Equal weights of main modules
    # main_weights = np.ones(len(modules)) / len(modules)
    # weights based on Saaty scale (determined subjectively by expert)
    main_weights = copy.deepcopy(W)

    
    # choose weighting method: ahp critic entropy
    weights_type = 'ahp'

    if weights_type == 'ahp':
        main_weights = copy.deepcopy(W)
        weights, crit_list = structured_equal_weights(modules, main_weights)
    # objective weights
    elif weights_type == 'critic':
        weights = critic_weighting(matrix)
        crit_list = [r'$C_{' + str(i + 1) + '}$' for i in range(0, df_data.shape[1])]
    elif weights_type == 'entropy':
        weights = entropy_weighting(matrix)
        crit_list = [r'$C_{' + str(i + 1) + '}$' for i in range(0, df_data.shape[1])]


    # save weights to csv
    df_weights = pd.DataFrame(weights.reshape(1, -1), index = ['Weights'], columns = crit_list)
    df_weights.to_csv('output/weights_' + weights_type + '.csv')


    # SSP-AHP Simulation 1
    # changes in the coefficient s in all criteria dimensions simultaneously
    # The value 0 of s coefficient corresponds to the classical AHP method
    df_sust = pd.DataFrame(index = list_alt_names)
    df_sust_pref = copy.deepcopy(df_sust)
    sust_coeffs = np.arange(0, 1.05, 0.05)
    for s in sust_coeffs:
        s_set = np.ones(matrix.shape[1]) * s
        pref = ahp(matrix, weights, types, mad = True, s = s_set)
        df_sust_pref[str(s)] = pref
        rank = rank_preferences(pref, reverse = True)
        df_sust[str(s)] = rank
    df_sust_pref.to_csv('output/sust_utility_vals_' + weights_type + '.csv')
    df_sust.to_csv('output/sust_rank_' + weights_type + '.csv')
    pdf = plot_sustainability(sust_coeffs, df_sust, weights_type)

    max_vector = df_sust_pref.iloc[:, -1].to_numpy()

    # Simulation 2
    # changes in the coefficient s in all combinations of criteria dimensions
    df_combs_pref = pd.DataFrame(index = list_alt_names)
    df_combs_rank = copy.deepcopy(df_combs_pref)

    indexes = []
    
    for el, c in enumerate(product([0, 1], repeat = 5)):
        c_array = np.array(c)
        ind = []
        for it, ce in enumerate(c):
            if ce:
                ind += modules_indexes[it]
        indexes.append(ind)

        ind_headers = np.where(c_array == 1)[0]
        new_G = ''
        for i in ind_headers:
            new_G += G[i] + ' '

        for s in sust_coeffs:
            s_set = np.zeros(matrix.shape[1])
            s_set[ind] += s
            
            pref = ahp(matrix, weights, types, mad = True, s = s_set)
            if np.any(pref == max_vector):
                break
        df_combs_pref[new_G] = pref
        rank = rank_preferences(pref, reverse = True)
        df_combs_rank[new_G] = rank
    
    df_combs_pref = df_combs_pref.T
    df_combs_pref = df_combs_pref.rename_axis('G')
    df_combs_pref.to_csv('output/crit_comb_utility_vals_' + weights_type + '.csv')

    df_combs_rank = df_combs_rank.T
    df_combs_rank = df_combs_rank.rename_axis('G')
    df_combs_rank.to_csv('output/crit_comb_ranks_' + weights_type + '.csv')
    
        
    
    # Simulation 3
    # preparation of 32 charts for each of the main criteria dimension G combination in separate pdfs
    # and one joined pdf with all 32 charts
    
    pdfs = []
    for el, c in enumerate(product([0, 1], repeat = 5)):
        df_sust = pd.DataFrame(index = list_alt_names)
        s_combs = np.array(c)

        ind_headers = np.where(s_combs == 1)[0]
        new_G = ''
        for i in ind_headers:
            new_G += G[i] + ' '

        # loop with s coefficient
        for s in sust_coeffs:
            
            s_set = np.zeros(matrix.shape[1])
            s_set[indexes[el]] += s
            # print('Set...')
            # print(indexes[el])
            # print(s_set)
            pref = ahp(matrix, weights, types, mad = True, s = s_set)
            rank = rank_preferences(pref, reverse = True)
            df_sust[str(s)] = rank
        pdf = plot_sustainability(sust_coeffs, df_sust, weights_type, str(el), title = str(new_G))
        pdfs.append(pdf)

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write("output/pdfjoined_" + weights_type + ".pdf")
    merger.close()
    
    
if __name__ == '__main__':
    main()