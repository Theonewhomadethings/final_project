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
   # print(evolution_op)
    initial_state = indicesN[0]
    print(len(indicesN))
   # for i in range(252):

main(K=10, N = 5, stand_dev = 0.05, delta_t=0.01)
'''
question 1
How do I intrepret the eigenvector columns that correspond to a single eigenvalue
w = (252, 252)
v = (252, )

Question 2
When choosing this initial state what kind of form does it take,
does it matter what it is
show him my current initial state to see if it works

Question 3
I am struggling to construct my for loop for question 3
for i in range()?
for n in array?
how many iterations
what does it mean by on each time step

Question 4)
what is the evolution operator being multiplied by in each iteration in the loop?
Because I am struggling to understand this . I need a vector of the same dimensions length wise to make the multiplaction work right?
the only vector i have of the same size is the indices N array

Question 5) 
ask if hes available for a quick meeting on friday possibly?
cos I have been stuck on question 3 and will probably need to double check this question result to check if it is correct
and ask about question 4
as next week I want to try present him the final project code draft as I need to submit a draft paper too.

Question 6) 
At the beggining or end, ask him to maybe check your current code or ask him if its easier to send it via github or email, to just ask him to take a look and see if there are any errors I am missing.

Also ask about the form of your evolution operator matrix as it looks a bit weird, show him step by step how you construct it as the P and Ptranspose array might be wrong as your strugling to work with it.

question 7) 
is the evolution operator meant to be complex?
as i was having issues debugging and multiplying real matrices with imaginary matrices?
'''
