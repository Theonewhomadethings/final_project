import numpy as np


def Main(l, N):
    e_k = np.random.normal(loc = 0, scale = 1, size = l) #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
   # for i in [1, 2**l]:
    arr = np.array([0]*N + [1]*(l-N)) # N = zeroes, N-l ones
    np.random.shuffle(arr) 
    print(arr)
    print(arr.size)
   
Main(l = 10, N= 5)