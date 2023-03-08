import numpy as np
from math import comb
import math

indicesN = []
E_n = []
def main(K, N, stand_dev, delta_t, iteration2):
    e_k = np.random.normal(loc = 0, scale = 1, size = K) #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
    #print("n  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] N") # debug print statement
    #print("\n")
    numbOfStatesN = comb(K, N) #252   
    for n in range(0, 2**K):
        n_k = list(f"{n:010b}")
        test_list = list(map(int, n_k))
        dotP = np.dot(test_list, e_k)
        # print(test_list)
        #tempSum = np.sum(n_k)
        tempSum = np.sum(test_list)
        if (tempSum == N):
            indicesN.append(n)
            E_n.append(dotP)
           # if (n < 2**K):
           # print(n, test_list, tempSum)
            if n == 677:
                initial_stateInt = n
                initial_stateBin = test_list
    #print("How many fermions =  ", len(indicesN))
    #print("size of energy array", len(E_n))
    E_n.sort() #sort energy values in ascending order
    H_0 = np.zeros(shape = (numbOfStatesN, numbOfStatesN))
    np.fill_diagonal(H_0, val = E_n)
   # print(H_0)
    wMatrix = np.random.normal(loc = 0, scale = stand_dev, size = (numbOfStatesN, numbOfStatesN))
    wMatrixT = np.transpose(wMatrix)
   # print("W matrix =", wMatrix)
    #print("W transpose m = ", wMatrixT)
    wPrimeT = np.add(wMatrix, wMatrixT)
    wPrime = np.divide(wPrimeT, 2)  #(wMatrix + wMatrix) / 2
   # print("wPrime = ", wPrime)
 #   print(np.shape(wPrime))
   # print(len(wMatrix))
    H = np.add(H_0, wPrime)
    #print(H)
    w, v = np.linalg.eig(H)
    #print("These are the eigenvalues: ", w)
    #print("This is the shape of the eigenvalues", np.shape(w))
    eD = np.zeros(shape = (numbOfStatesN, numbOfStatesN), dtype= "complex")#empty matrix
   # np.fill_diagonal(D, val = w)
    for i in range(252):
        eD[i][i] = np.exp(-1j*w[i]*delta_t)
  #    print(eD)
    '''
    w = (252, 252)
    v = (252)
    Note for the eigenvector the column v[:,i] is the eigenvector
1x    corresponding to the eigenvalue w[i]    
    '''
    eigenvector1= v[:,1]
    eigvalue1 = w[1]
    #print(eigenvector1)
    #print(eigvalue1)
    p_t = np.transpose(v)
    p_dagger = np.conjugate(p_t)
   # evolution_op1 = v*eD*p_t
    evolOPt = np.multiply(v, eD) 
    evolOP = np.multiply(evolOPt, p_dagger)
    check1 = np.multiply(v, p_dagger) #p inverse
    checkI =  np.linalg.inv(v)
    print(v)
    checkIfull = np.multiply(v, checkI)
    print(checkI)
 #   check2 = np.multiply(check1, p) # pinverse * p = I
  #  print(check2)
  #  print(np.shape(evolution_op1)) #DEBUG 2
   # print(evolOP) # DEBUG 1

    #print(initial_stateBin, initial_stateInt)
    init_state = np.zeros(shape = (numbOfStatesN))
    #init_state[initial_stateInt] = 1
    #print(np.shape(init_state))
   
    #for k in range(len(init_state)):
           # weights = []
         #creating empty list for the resulting vector
            #variable for each row of action
            #items = 0
            #for l in range(len(init_state)):
                #adding the result of matrix vector itemwise multiplication
               # items += evolution_op2[k][l] * init_state[l]
              #  weights.append(items)
#    print(np.shape(weights))
main(K=10, N = 5, stand_dev = 0.05, delta_t=0.01, iteration2 = 1000)


