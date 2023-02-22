import numpy as np
import math

def test(t):
    w = [] #array of size 252, with random numbers between -3 and 3 the values here arent important just the method really
    #so
    wTemp = np.random.uniform(low=-3, high=3, size=252)
    D = np.zeros(shape=(252, 252))
    np.fill_diagonal(D, val = wTemp)
    #print(D) #252, 252
    #now dont ask why but I want to do this with the D matrix to get:
    #e^(-i*D*t)
    #where D = [[w1, 0, 0...], [0, w2, 0....],..... [0, 0, w252]]
    #I want Dprime = [[e^(-i*w1*t), 0, 0, ...], [0, e^(-i*w2*t) , 0,...],..... [...0, 0, e^(-i*w252*t)]]
    E = np.zeros(shape = (252, 252), dtype= 'complex')
    for i in range(252):
        w_element = D[i][i]
        E[i][i] = np.exp(-1j* w_element*t)
    print(E)
test(t = 0.01)
