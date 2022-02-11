import numpy as np

# reverse = True: descending order (TOPSIS, CODAS), False: ascending order (VIKOR, SPOTIS)
def rank_preferences(pref, reverse = True):
    rank = np.zeros(len(pref))
    sorted_pref = sorted(pref, reverse = reverse)
    pos = 1
    for i in range(len(sorted_pref) - 1):
        ind = np.where(sorted_pref[i] == pref)[0]
        rank[ind] = pos
        if sorted_pref[i] != sorted_pref[i + 1]:
            pos += 1
    ind = np.where(sorted_pref[i + 1] == pref)[0]
    rank[ind] = pos
    return rank.astype(int)
