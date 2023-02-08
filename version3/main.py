import numpy as np

indicesN = []
E_n = []


def main(K, N):
    e_k = np.random.normal(loc = 0, scale = 1, size = K) #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
    print("n  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] N") # debug print statement
    print("\n")
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
            if (n < 2**K):
                print(n, test_list, tempSum)
    print("How many fermions =  ", len(indicesN))
    print("size of energy array", len(E_n))
main(K=10, N = 5)
