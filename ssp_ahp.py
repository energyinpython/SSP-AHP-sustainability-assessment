import numpy as np
from normalizations import *
from mcdm_method import MCDM_method


class AHP(MCDM_method):
    def __init__(self, normalization_method = minmax_normalization):
        self.normalization_method = normalization_method


    def __call__(self, matrix, weights, types, mad = False, s = 0):
        AHP._verify_input_data(matrix, weights, types)
        return AHP._ahp(self, matrix, weights, types, self.normalization_method, mad, s)


    def _check_consistency(self, X):
        if X.shape[0] != X.shape[1]:
            raise ValueError('Number of rows and columns of pairwise comparison matrix must be equal')
        n = X.shape[1]
        RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        lambdamax = np.amax(np.linalg.eigvals(X).real)
        CI = (lambdamax - n) / (n - 1)
        CR = CI / RI[n - 1]
        print("Inconsistency index: ", CR)
        if CR > 0.1:
            print("The pairwise comparison matrix is inconsistent")


    def _calculate_eigenvector(self, X):
        if X.shape[0] != X.shape[1]:
            raise ValueError('Number of rows and columns of pairwise comparison matrix must be equal')
        val, vec = np.linalg.eig(X)
        eig_vec = np.real(vec)
        eig_vec = eig_vec[:, 0]
        S = eig_vec / np.sum(eig_vec)
        S = S.ravel()
        return S


    # normalization of decision matrix was performed in the previous stage
    # thus divide by sum is not needed
    def _normalized_column_sum_method(self, X):
        return np.sum(X, axis = 1)


    def _normalized_column_sum_method_classic(self, X):
        return np.sum(X, axis = 1) / np.sum(X)

    
    def _classic_ahp(self, alt_matrices, weights):
        for alt in alt_matrices:
            self._check_consistency(alt)

        m = alt_matrices[0].shape[0]
        n = len(weights)

        S = np.zeros((m, n))
        for el, alt in enumerate(alt_matrices):
            S[:, el] = self._calculate_eigenvector(alt)

        Sw = S * weights
        S_final = self._normalized_column_sum_method_classic(Sw)
        return S_final


    @staticmethod
    def _ahp(self, matrix, weights, types, normalization_method, mad, s):
        nmatrix = normalization_method(matrix, types)
        
        if mad == True:
            std_val = np.abs(np.mean(nmatrix, axis = 0) - nmatrix) * s
            nmatrix = nmatrix - std_val
        weighted_matrix = nmatrix * weights
        pref = self._normalized_column_sum_method(weighted_matrix)
        return pref