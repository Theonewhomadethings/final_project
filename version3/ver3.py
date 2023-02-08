import numpy as np
from math import comb
'''
K = number of single body energies
N = fixed number of fermions in the system
'''
def main(K, N):
    numbOfStatesN = comb(K, N) #252   
    indicesN = set()
    for i in range(1, 2**K):
        e_k = np.random.normal(loc = 0, scale = 1, size = K) #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
        n_k = np.array([0]*N + [1]*(K-N)) # N = zeroes, N-l ones
        np.random.shuffle(n_k)
        #print(n_k)
        for j in range(len(n_k)):
            n_k[j] = (2**j) *n_k[j]
        
        sum1 = np.sum(n_k)
        #if (len(indicesN) < numbOfStatesN):
        indicesN.add(sum1)
        #indicesN.append(sum1)
     #   fillOrNot = np.dot(e_k, indicesN)
    print(indicesN)
    print(len(indicesN))
  


main(K = 10, N = 5) 
