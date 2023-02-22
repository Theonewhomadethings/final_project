import numpy as np
from math import comb
import math

indicesN = []
E_n = []

def main(K, N, stand_dev, delta_t):
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
    print(w)
    z = 0 + 1j # 
    #tempEvol = math.exp(-z*delta_t*D)
    #print(tempEvol)
    tempEvoleig = np.zeros(shape=(numbOfStatesN, numbOfStatesN))
    wNew = []
    for n in w:
        wNew.append(math.exp(z*n*delta_t))
    print(wNew)
main(K=10, N = 5, stand_dev = 0.05, delta_t=0.01)

def interaction(stand_dev):
    wMatrix = np.random.normal(loc = 0, scale = stand_dev, size = ())
    wMatrixT = np.transpose(wMatrix)
    wOnwT = np.multiply(wMatrix, wMatrixT)
    print(wOnwT)
    print(wOnwT.shape())
    #wTrace = np.trace(wOnwT)
   # print(wTrace)
    #dimensions of matrix?
    #independent iddenticallly distributed  entries meaning
    #gaussian of mean 0, standard deviation sigma_w
    #areal asumetric matrix
    
    #sanity check trace of w multiplied by its transpose tr(W*W^T) APPROX = STANDARD DEV^2 TIMES THE NUMBER OF MANY BODY STATES
    #Trace - defined to be the sum of elements on the main diagonal

#interaction(stand_dev=0.05)
