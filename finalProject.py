'''
Prerequisites
'''
import numpy as np

'''
Main **Note to user: Current bug where the while loop only works half the time. if 3 arrays arent outputted simply re run program**
'''

def Main(N):
    # part a code The single body energies E_k, the number of energy values are up to the given integer K=10. 
    e_k = np.random.normal(loc = 0, scale = 1, size = 10) #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
    print("E_k single body energy values: ", e_k, "\n") # debug statement
    indicesN = []
    n_k = []
    s1 = 0
    while (s1 < N): #part b code
        n_k = np.random.choice([0, 1], size = 10)
        s1 = np.sum(n_k)
        if s1 == N:
            print("n_k array of binary numbers = 5 fermions present", n_k, "\n") #debug statement
            
            for i in range(len(e_k)):
                n = (n_k[i]) * (2**i)
                indicesN.append(n)
            print("Vector of indices n:", indicesN)
        continue
Main(5)

'''
Debug 
'''

#print(E_k.shape)