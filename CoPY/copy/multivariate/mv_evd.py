"""Multivariate extreme value copula module contains methods for sampling from a multivariate
extreme value copula and to compute the asymptotic variance of the w-madogram under missing or
complete data.

Multivariate extreme value copulas are characterized by their stable tail dependence function 
which the restriction to the unit simplex gives the Pickands dependence function. The copula 
function

..math:: C(u) = exp\{-\ell(-log(u_1), \dots, -log(u_d))\}, \quad 0 < u_j \leq 1,

is a multivariate extreme value copula. To sample from a multivariate extreme value copula, we 
implement the Algoritm 2.1 and 2.2 from Stephenson (2002).

Structure :

- Extreme value copula (:py:class:`Extreme`) from copy.multivariate.base.py
    - Logistic model (:py:class:`Logistic`)
    - Asymmetric logistic model (:py:class:`Asymmetric_logistic`)
"""
import numpy as np
import math
import utils

from base import Multivariate, CopulaTypes, Extreme

"""
    Commentaire : 
        Gérer les indices de dérivations
"""

class Logistic(Extreme):
    """
        Class for multivariate Logistic copula model.
    """

    copula_type = CopulaTypes.LOGISTIC
    theta_interval = [0, 1]
    invalid_thetas = [0]

    def _A(self, t):
        """Return the value of the Pickands dependence function taken on t.
        ..math:: A(t) = (\sum_{j=1}^d t_i^{1/\theta})^\theta, \quad t \in \Delta^{d-1}.

        Inputs
        ------
            t (list[float]) : list of elements of the simplex in R^{d}
        """
        s = np.sum(t)
        value_ = math.pow(np.sum(np.power(t, 1/self.theta)),self.theta)

        return value_

    def _Adot(self, t, j):
        """Return the value of jth partial derivative of the Pickands dependence function taken on t

        Inputs
        ------
            t(list[float]) : list of elements of the simplex in R^{d}
                         j : index of the partial derivative \geq 1
        """
        s = np.sum(t[1:]) # \sum_{j=1}^{d-1} t_j
        value_1 = (1/self.theta * math.pow(t[j],(1-self.theta)/self.theta) - 1/self.theta * math.pow(1-s,(1-self.theta)/self.theta))
        value_2 = math.pow(self._A(t), (self.theta - 1)/self.theta)
        value_  = self.theta * value_1 * value_2
        return value_

    def rmvlog_tawn(self):
        """ Algorithm 2.1 of Stephenson (2002).
        """
        sim = np.zeros(self.n_sample * self.d)
        for i in range(0 , self.n_sample):
            s = utils.rpstable(self.theta)
            for j in range(0,self.d):
                sim[i*self.d + j] = math.exp(self.theta * (s - math.log(np.random.exponential(size = 1))))
        return sim

    def sample_unimargin(self):
        """Draws a sample from a multivariate Logistic model.

        Output
        ------
        sim (np.array([float])) : dataset of shape n_sample x d
        """
        sim = self._frechet(self.rmvlog_tawn())
        return sim.reshape(self.n_sample, self.d)

class Asymmetric_logistic(Extreme):
    """
        Class for multivariate asymmetric logistic copula model
    """

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC

    """
        Add some check functions for the dependence and asymmetric parameters
    """

    def _A(self, t):
        """Return the value of the Pickands dependence function taken on t
        ..math:: A(t) = \sum_{b \in B} (\sum_{j \in b} (\psi_{j,b} t_j)^{1/\theta_b}))^{\theta_b}, \quad t \in \Delta^{d-1}

        Inputs
        ------
            t (list[float]) : list of elements of the simplex in R^{d}
        """
        nb = int(2**self.d - 1)
        dep = np.repeat(self.theta, nb - self.d)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        asy = self.mvalog_check(dep)
        A_ = []
        for b in range(0, nb):
            x = np.power(asy[b,:], 1/dep[b])
            y = np.power(t, 1/dep[b])
            value = np.dot(x, y)
            A_.append(np.power(value, dep[b]))

        return np.sum(A_)

    def _Adot(self, t,j):
        """Return the value of jth partial derivative of the Pickands dependence function taken on t

        Inputs
        ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
                         j : index of the partial derivative >= 1
        """
        nb = int(2**self.d - 1)
        dep = np.repeat(self.theta, nb - self.d)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        asy = self.mvalog_check(dep)
        Adot_ = []
        for b in range(0, nb):
            z = np.zeros(self.d) ; z[0] = -np.power(t[0], (1-dep[b]) / dep[b]) ; z[j] = np.power(t[j], (1-dep[b]) / dep[b])
            x = np.power(asy[b,:], 1/dep[b])
            y = np.power(t, 1/dep[b])
            value_1 = np.dot(x, z)
            value_2 = np.power(np.dot(x,y), (dep[b] - 1))
            Adot_.append(value_1 * value_2)

        return np.sum(Adot_)


    def rmvalog_tawn(self,nb, alpha, asy):
        """ Algorithm 2.2 of Stephenson (2008). """
        sim = np.zeros(self.n_sample*self.d)
        gevsim = np.zeros(nb*self.d)
        maxsim = np.zeros(nb)
        for i in range(0,self.n_sample):
            for j in range(0, nb):
                if alpha[j] != 1:
                    s = utils.rpstable(alpha[j])
                else: s = 0
                for k in range(0, self.d):
                    if asy[j*self.d+k] != 0:
                        gevsim[j*self.d+k] = asy[j*self.d+k] * math.exp(alpha[j] * (s -math.log(np.random.exponential(size = 1))))

            for j in range(0,self.d):
                for k in range(0,nb):
                    maxsim[k] = gevsim[k*self.d+j]

                sim[i*self.d+j] = utils.maximum_n(nb, maxsim)

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
        
        b = utils.subsets(self.d)
        asy = tasy(self.asy, b)
        y = np.sum(asy)
        indices = [index for index in range(len(dep)) if dep[index] == 1.0]
        if y != self.d:
            raise TypeError("asy does not satisfy the appropriate constraints, sum")
        for index in indices:
            if np.sum(dep[index]) > 0 and (index >= self.d):
                raise TypeError("asy does not satisfy the appropriate constrains")
        return asy

    def sample_unimargin(self):
        nb = int(2**self.d -1)
        dep = np.repeat(self.theta, nb - self.d)
        asy = self.mvalog_check(dep).reshape(-1)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        sim = self._frechet(self.rmvalog_tawn(nb,dep,asy))
        return sim.reshape(self.n_sample,self.d)