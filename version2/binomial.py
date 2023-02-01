from math import comb


def factorial(N):
    ans = 1
    for i in range(1, N):
        ans = ans*i
    print(ans)
    
def NCR(N, K):
    return factorial(N) / factorial(K)*factorial(N-K)


#NCR(5, 10)
        