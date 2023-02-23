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
                #print(n, test_list, tempSum)
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
    wPrime = (wMatrix + wMatrixT) / 2
   # print("wPrime = ", wPrime)
 #   print(np.shape(wPrime))
   # print(len(wMatrix))
    H = np.add(H_0, wPrime)
    #print(H)
    w, v = np.linalg.eig(H)
    #print("These are the eigenvalues: ", w)
    #print("This is the shape of the eigenvalues", np.shape(w))
    D = np.zeros(shape = (numbOfStatesN, numbOfStatesN))#empty matrix
    np.fill_diagonal(D, val = w)
    E = np.zeros(shape=(numbOfStatesN, numbOfStatesN), dtype= "complex")
    for i in range(252):
        w_element = D[i][i]
        E[i][i] = np.exp(-1j*w_element*delta_t)
    #print(np.shape(E))
    p = v 
    '''
    w = (252, 252)
    v = (252)
    Note for the eigenvector the column v[:,i] is the eigenvector
    corresponding to the eigenvalue w[i]    
    '''
    eigenvector1= v[:,1]
    eigvalue1 = w[1]
    #print(eigenvector1)
    #print(eigvalue1)
    p_t = np.transpose(p)
    evolution_op = p*E*p_t
    #print(evolution_op)
    initial_state = np.zeros(shape = (numbOfStatesN))

    print(initial_state)

    for i in range(iteration2):
        intial_state = np.multiply(evolution_op, initial_state) 

main(K=10, N = 5, stand_dev = 0.05, delta_t=0.01, iteration2 = 1000)









'''
question 1
In the initial state how many ones and zeroes should there be

question 2 
Is it on purpose or random whether there is a 1 or 0 placed in this vector

question 3
what is the maximum number of 1s and 0s in the vector

question 4
quick side question, would you count this project as a theoretical or computaional physics project?
''
