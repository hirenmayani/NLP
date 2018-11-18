import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    pathExtractPtrs = np.empty((N-1,L))
    t = np.empty((N,L))
    y = [0]*N
#     for i in xrange(N):
#         # stupid sequence
#         y.append(i % L)
    # score set to 0
    
    def getMax(trans_scores, t,i,j):
        temps = []
        for  k in range(L):
            temps.append(trans_scores[k,j] + t[i-1,k])
        e = max(temps)
        return temps.index(e),e
        
    def getMax2(end_scores, t):
        temps = []
        for  k in range(L):
            temps.append(end_scores[k] + t[-1,k])
        e = max(temps)
        
        return temps.index(e),e
        
        
    for wi in range(N):
        for lj in range(L):
            if wi ==0:
                t[wi,lj] = start_scores[wi]+emission_scores[wi,lj]
            else:
                i,e = getMax(trans_scores, t,wi,lj)
                pathExtractPtrs[wi-1,lj] = i
                t[wi,lj] = emission_scores[wi,lj] +  e
        
    j,score = getMax2(end_scores, t)
    y[N-1] = j
    
    for i in range(N-2,-1,-1):
        j  = int(pathExtractPtrs[i,j])
        y[i] = j
        
    return (score, y)
