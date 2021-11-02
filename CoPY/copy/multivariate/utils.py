import itertools
import numpy as np
import math

def maximum_n(n, x):
    """
        Step 2 of Algorithm 2.2.
    """
    for i in range(1,n):
        if(x[0] < x[i]) : x[0] = x[i]
    return x[0]

def subsets(d):
    """All subsets of \{1,\dots,d} (empty set is not taken into account).
    
    Output
    ------
        pset (list[list]) : a list containing (2^d)-1 list([int]).
    """
    x = range(0,d)
    pset = [list(subset) for i in range(0, len(x)+1) for subset in itertools.combinations(x,i)]
    del pset[0]
    return pset

def rpstable(cexp):
    """Sample from a Positive Stable distribution.
    """
    if cexp==1: return 0
    tcexp = 1-cexp
    u = np.random.uniform(size = 1) * math.pi
    w = math.log(np.random.exponential(size = 1))
    a = math.log(math.sin(tcexp*u)) + (cexp / tcexp) * math.log(math.sin(cexp*u)) - (1/tcexp) * math.log(math.sin(u))
    return (tcexp / cexp) * (a-w)

"""
    Define a multivariate Unit-Simplex
"""

def simplex(d, n = 50, a = 0, b = 1):
    """http://www.eirene.de/Devroye.pdf
    Algorithm page 207.
    """
    output = np.zeros([n,d])
    for k in range(0,n):
        x_ = np.zeros(d+1)
        y_ = np.zeros(d)
        for i in range(1, d):
            x_[i] = np.random.uniform(a,b)
        x_[d] = 1.0
        x_ = np.sort(x_)
        for i in range(1,d+1):
            y_[i-1] = x_[i] - x_[i-1]
        output[k,:] = y_
    return output