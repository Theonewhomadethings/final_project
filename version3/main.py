import numpy as np

def main(K, N):
    for i in range(0, 2**K):
        n_k = list(f"{i:010b}")
        test_list = list(map(int, n_k))
       # print(test_list)
        tempSum = np.sum(test_list)
        #if (tempSum == N):
         #   print(i, test_list)
        print(i, n_k)
main(K=10, N = 5)
