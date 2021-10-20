import numpy as np
import math
import itertools

from base import Multivariate, CopulaTypes, Extreme

"""
    Commentaire : 
        Gérer les indices de dérivations
"""

class Logistic(Extreme):
    """
        Class for multivariate Logistic copula model
    """

    copula_type = CopulaTypes.LOGISTIC
    theta_interval = [0, 1]
    invalid_thetas = [0]

    def _A(self, t):
        """
            Return the value of the Pickands dependence function taken on t

            Inputs
            ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
        """
        s = np.sum(t)
        value_ = math.pow(math.pow(1-s, 1/self.theta) + np.sum(np.power(t, 1/self.theta)),self.theta)

        return value_

    def _Adot(self, t, j):
        """
            Return the value of jth partial derivative of the Pickands dependence function taken on t

            Inputs
            ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
                         j : index of the partial derivative > 2
        """
        j = j - 2
        s = np.sum(t)
        value_1 = (1/self.theta * math.pow(t[j],(1-self.theta)/self.theta) - 1/self.theta * math.pow(1-s,(1-self.theta)/self.theta))
        value_2 = math.pow(self._A(t), (self.theta - 1)/self.theta)
        value_  = self.theta * value_1 * value_2
        return value_

    def rmvlog_tawn(self):
        sim = np.zeros(self.n_sample * self.d)
        for i in range(0 , self.n_sample):
            s = self._rpstable()
            for j in range(0,self.d):
                sim[i*self.d + j] = math.exp(self.theta * (s - math.log(np.random.exponential(size = 1))))
        return sim

    def sample_unimargin(self):
        sim = self._frechet(self.rmvlog_tawn())
        return sim.reshape(self.n_sample, self.d)

class Asymmetric_Logistic(Extreme):
    """
        Class for multivariate asymmetric logistic copula model
    """

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC

    """
        Add some check functions for the dependence and asymmetric parameters
    """

    def _A(self, t):
        """
            Return the value of the Pickands dependence function taken on t

            Inputs
            ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
        """
        s = np.sum(t)
        w = np.insert(t,0,1-s) # COMMENT : changer les lignes de code pour prendre un élément du simplexe de dimension d
        nb = int(2**self.d - 1)
        dep = np.repeat(self.theta, nb - self.d)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        asy = self.mvalog_check(dep)
        A_ = []
        for b in range(0, nb):
            x = np.power(asy[b,:], 1/dep[b])
            y = np.power(w, 1/dep[b])
            value = np.dot(x, y)
            A_.append(np.power(value, dep[b]))

        return np.sum(A_)

    def _Adot(self, t,j):
        """
            Return the value of jth partial derivative of the Pickands dependence function taken on t

            Inputs
            ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
                         j : index of the partial derivative >= 2
        """
        j= j - 2 # start at 0
        s = np.sum(t)
        w = np.insert(t,0,1-s) # COMMENT : voir au dessus
        nb = int(2**self.d - 1)
        dep = np.repeat(self.theta, nb - self.d)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        asy = self.mvalog_check(dep)
        Adot_ = []
        for b in range(0, nb):
            z = np.zeros(self.d) ; z[0] = -np.power(1-s, (1-dep[b]) / dep[b]) ; z[j+1] = np.power(t[j], (1-dep[b]) / dep[b])
            x = np.power(asy[b,:], 1/dep[b])
            y = np.power(w, 1/dep[b])
            value_1 = np.dot(x, z)
            value_2 = np.power(np.dot(x,y), (dep[b] - 1))
            Adot_.append(value_1 * value_2)

        return np.sum(Adot_)


    def rmvalog_tawn(self,nb, alpha, asy):
        sim = np.zeros(self.n_sample*self.d)
        gevsim = np.zeros(nb*self.d)
        maxsim = np.zeros(nb)
        for i in range(0,self.n_sample):
            for j in range(0, nb):
                if alpha[j] != 1:
                    s = rpstable(alpha[j]) # généraliser ça
                else: s = 0
                for k in range(0, self.d):
                    if asy[j*self.d+k] != 0:
                        gevsim[j*self.d+k] = asy[j*self.d+k] * math.exp(alpha[j] * (s -math.log(np.random.exponential(size = 1))))

            for j in range(0,self.d):
                for k in range(0,nb):
                    maxsim[k] = gevsim[k*self.d+j]

                sim[i*self.d+j] = maximum_n(nb, maxsim)

        return sim
    
    def mvalog_check(self, dep):
        if(dep.any() <= 0 or dep.any() > 1.0):
            raise TypeError('invalid argument for theta')
        nb = 2 ** self.d - 1
        if(not isinstance(self.asy, list) or len(self.asy) != nb) :
            raise TypeError('asy should be a list of length', nb)

        def tasy(theta, b):
            trans = np.zeros([nb,self.d])
            for i in range(0, nb):
                j = b[i]
                trans[i,j] = theta[i]
            return trans
        
        b = subsets(self.d)
        asy = tasy(self.asy, b)
        return asy

    def sample_unimargin(self):
        nb = int(2**self.d -1)
        dep = np.repeat(self.theta, nb - self.d)
        asy = self.mvalog_check(dep).reshape(-1)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        sim = self._frechet(self.rmvalog_tawn(nb,dep,asy))
        return sim.reshape(self.n_sample,self.d)

def maximum_n(n, x):
    """
        Step 2 of Algorithm 2.2
    """
    for i in range(1,n):
        if(x[0] < x[i]) : x[0] = x[i]
    return x[0]

def subsets(d):
    x = range(0,d)
    pset = [list(subset) for i in range(0, len(x)+1) for subset in itertools.combinations(x,i)]
    del pset[0]
    return pset

def rpstable(cexp):

    if cexp==1: return 0
    tcexp = 1-cexp
    u = np.random.uniform(size = 1) * math.pi
    w = math.log(np.random.exponential(size = 1))
    a = math.log(math.sin(tcexp*u)) + (cexp / tcexp) * math.log(math.sin(cexp*u)) - (1/tcexp) * math.log(math.sin(u))
    return (tcexp / cexp) * (a-w)