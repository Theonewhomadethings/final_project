import numpy as np

#question does it matter whether you read the arrays left to right or right to left?
#negative e_k values? ok?
'''
Right now I am creating a single many body energy value
But i NEED to create a vector of many body energies for the states 
containing N fermions
'''

def Main(l, N):
    #for n in [1, 2**l    
    E_n = []
    for n in range(1, 2**l):
        e_k = np.random.normal(loc = 0, scale = 1, size = l) #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
        n_k = np.array([0]*N + [1]*(l-N)) # N = zeroes, N-l ones
        np.random.shuffle(n_k) 
        newArr = [0]*len(n_k)

        for i in range(len(n_k)):
            variable1 = 2**i
            newArr[i] = variable1
            #print(variable1) #debug
        
        indicesN = n_k*newArr
        sum1 = np.sum(indicesN)
        #E_n[n-1] = np.dot(e_k, n_k)
        E_n.append(np.dot(e_k, n_k))
    #print(E_n)
    print(len(E_n))

    #print(n_k) #debug
   # print(newArr) #debug
    #print(indicesN)
   # print(sum1)
    #print(e_k)
   # print(E_n)
   # print(fillOrNot)




            

Main(l = 10, N= 5)