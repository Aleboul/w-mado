import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('seaborn-whitegrid')
from mpl_toolkits import mplot3d
import itertools

def frechet(x):
    return np.exp(-1/x)


def rpstable(cexp):

    if cexp==1: return 0
    tcexp = 1-cexp
    u = np.random.uniform(size = 1) * math.pi
    w = math.log(np.random.exponential(size = 1))
    a = math.log(math.sin(tcexp*u)) + (cexp / tcexp) * math.log(math.sin(cexp*u)) - (1/tcexp) * math.log(math.sin(u))
    return (tcexp / cexp) * (a-w)

def maximum_n(n, x):

    for i in range(1,n):
        if(x[0] < x[i]): x[0] = x[i]
    return x[0]

def rmvalog_tawn(n, d, nb, alpha, asy):
    sim = np.zeros(n*d)
    gevsim = np.zeros(nb*d)
    maxsim = np.zeros(nb)
    for i in range(0,n):
        for j in range(0, nb):
            if alpha[j] != 1:
                s = rpstable(alpha[j])
            else: s = 0
            for k in range(0, d):
                if asy[j*d+k] != 0:
                    gevsim[j*d+k] = asy[j*d+k] * math.exp(alpha[j] * (s -math.log(np.random.exponential(size = 1))))
        
        for j in range(0,d):
            for k in range(0,nb):
                maxsim[k] = gevsim[k*d+j]

            sim[i*d+j] = maximum_n(nb, maxsim)
    
    return sim

def subsets(d):
    x = range(0,d)
    pset = [list(subset) for i in range(0, len(x)+1) for subset in itertools.combinations(x,i)]
    del pset[0]
    return pset

def mvalog_check(asy, dep, d):
    if(dep.any() <= 0 or dep.any() >1):
        raise TypeError("invalid argument for dep")
    nb = 2**d - 1
    if(not isinstance(asy, list) or len(asy) != nb):
        raise TypeError("asy sould be a list of length", nb)
    
    def tasy(theta, b):
        trans = np.zeros([nb,d])
        for i in range(0,nb):
            j = b[i]   
            trans[i, j] = theta[i]
        return trans

    b = subsets(d)
    asy = tasy(asy, b)
    return asy


def rmvalog(n, dep, asy, d=2):
    nb = int(2**d - 1)
    dep = np.repeat(dep, nb-d)
    asy = mvalog_check(asy, dep, d=d).reshape(-1)
    print(asy)
    dep = np.concatenate([np.repeat(1,d), dep], axis = None)
    sim = frechet(rmvalog_tawn(n,d,nb,dep,asy))
    return sim.reshape(n,d)


#asy = [0.9,0.0,[0.1,1.0]]
#dep = [1/2.5]
#test = rmvalog(10000, dep, asy)
#print(test)
#fig, ax = plt.subplots()
#ax.plot(test[:,0], test[:,1],'.', color = 'salmon', markersize = 1)
#plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/2d.pdf")

#asy = [0.7,0.1,0.6,[0.1,0.2], [0.1,0.1], [0.4,0.1], [0.1,0.3,0.2]]
#dep = [0.05, 0.005, 0.9,0.9]

asy = [.4, .1, .6, [.3,.2], [.1,.1], [.4,.1], [.2,.3,.2]]
dep = [.1,.1,.1,.1]
test = rmvalog(1000, dep, asy, d = 3)
print(test)
fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(test[:,0],test[:,1],test[:,2], color = 'lightblue', s = 1)
plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/copula_3d.pdf")