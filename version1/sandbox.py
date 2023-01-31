import numpy as np 

N = 10 #N-k ones
K = 5 # zeroes

arr = np.array([0]*K + [1]*(N-K))
print(arr)
np.random.shuffle(arr)
print(arr)
print(len(arr))
print(sum(arr))