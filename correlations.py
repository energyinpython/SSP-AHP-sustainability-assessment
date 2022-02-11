import numpy as np

# weighted spearman coefficient rw
def weighted_spearman(R, Q):
    N = len(R)
    denominator = N**4 + N**3 - N**2 - N
    numerator = 6 * sum((R - Q)**2 * ((N - R + 1) + (N - Q + 1)))
    rW = 1 - (numerator / denominator)
    return rW

# rank similarity coefficient WS
def coeff_WS(R, Q):
    sWS = 0
    N = len(R)
    for i in range(N):
        sWS += 2**(-int(R[i]))*(abs(R[i]-Q[i])/max(abs(R[i] - 1), abs(R[i] - N)))
    WS = 1 - sWS
    return WS

# pearson coefficient
def pearson_coeff(R, Q):
    numerator = np.sum((R - np.mean(R)) * (Q - np.mean(Q)))
    denominator = np.sqrt(np.sum((R - np.mean(R))**2) * np.sum((Q - np.mean(Q))**2))
    corr = numerator / denominator
    return corr
