import numpy as np
from math import comb

indicesN = []
E_n = []

def main(K, N, stand_dev):
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
 #   print(wMatrix)
   # print("\n")
    wMatrixT = np.transpose(wMatrix)
   # print(wMatrixT)
   # print("\n")
    wOnwT = np.multiply(wMatrix, wMatrixT)
    #print(wOnwT)
   # print(wOnwT.shape())
    wTrace = np.trace(wOnwT)
   #print("trace = ", wTrace)
    check = ((stand_dev)**2)* numbOfStatesN
    #print("STD squared * numb of many body states = ", check)

    #now build the sum Hhat = H0 + what
    totalHamiltonian = np.add(H_0, wMatrix)
    print(type(totalHamiltonian[0][0])) #shape = (252, 252)
    #question 2
    # diagonalise total hamiltonian to get eigenvalues
    diag1 = np.diag(totalHamiltonian) # (252, )
    #print(np.shape(diag1))
    w, v = np.linalg.eig(totalHamiltonian)
    print("Eigenvalues = ", w)
    print("Eigenvectors", v)
main(K=10, N = 5, stand_dev = 0.05)

def interaction(stand_dev,K, N):
    numbOfStatesN = comb(K, N)
    wMatrix = np.random.normal(loc = 0, scale = stand_dev, size = (numbOfStatesN, numbOfStatesN))
    print(wMatrix)
    print("\n")
    wMatrixT = np.transpose(wMatrix)
    print(wMatrixT)
    print("\n")
    wOnwT = np.multiply(wMatrix, wMatrixT)
    print(wOnwT)
   # print(wOnwT.shape())
    wTrace = np.trace(wOnwT)
    print("trace = ", wTrace)
    check = ((stand_dev)**2)* numbOfStatesN
    print("STD squared * numb of many body states = ", check)
    #dimensions of matrix?
    #independent iddenticallly distributed  entries meaning
    #gaussian of mean 0, standard deviation sigma_w
    #areal asumetric matrix
    #sanity check trace of w multiplied by its transpose tr(W*W^T) APPROX = STANDARD DEV^2 TIMES THE NUMBER OF MANY BODY STATES
    #Trace - defined to be the sum of elements on the main diagonal
#nteraction(stand_dev=0.05, K=10, N = 5)