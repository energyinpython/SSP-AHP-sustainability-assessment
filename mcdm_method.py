from abc import ABC
import numpy as np

class MCDM_method(ABC):

    def __call__(self, matrix, weights, types):
        pass

    @staticmethod
    def _verify_input_data(matrix, weights, types):
        m, n = matrix.shape
        if len(weights) != n:
            raise ValueError('The size of the weight vector must be the same as the number of criteria')
        if len(types) != n:
            raise ValueError('The size of the types vector must be the same as the number of criteria')
        check_types = np.all((types == 1) | (types == -1))
        if check_types == False:
            raise ValueError('Criteria types can only have a value of 1 for profits and -1 for costs')