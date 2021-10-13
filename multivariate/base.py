import numpy as math
import math

from enum import Enum

def min(a, b):
  
    if a <= b:
        return a
    else:
        return b

class CopulaTypes(Enum):
    """
        Available multivariate copula
    """

    LOGISTIC = 1
    ASYMETRIC_LOGISTIC = 2

class Multivariate(object):
    """
        Base class for multivariate copulas.
        This class allows to instantiate all its subclasses and serves
        as a unique entry point for the multivariate copulas classes.
        It permit also to compute the variance of the w-madogram for a given point.
        
        Inputs
        ------
            copula_type : substype of the copula
            random_seed : seed for the random generator
                      d : dimension
        Attributes
        ----------
            copula_type(CopulaTypes) : family of the copula that belongs to
            theta_interval(list[float]) : interval of valid theta for the given copula family
            invalid_thetas(list[float]) : values, that even though they belong to
                                          :attr: `theta_interval`, shouldn't be considered as valid.
            theta(list[float]) : parameter for the parametric copula
            var_mado(list[float]) : value of the theoretical variance for a given point in the simplex
    """
    copula_type = None
    theta_interval = []
    invalid_thetas = []
    n_sample = []
    asy = []
    theta = []
    d = []

    def __init__(self, random_seed = None, theta = None, n_sample = None, asy = None, d = None):
        """
            Initialize Multivariate object.
            
            Inputs
            ------
                copula_type (copula_type or st) : subtype of the copula
                random_seed (int or None) : seed for the random generator
                theta (list[float] or None) : list of parameters for the parametric copula
                asy (list[float] or None) : list of asymetric values for the copula
                n_sample (int or None) : number of sampled observation
        """
        self.random_seed = random_seed
        self.theta = theta
        self.n_sample = n_sample
        self.asy = asy
        self.d = d

    def _frechet(x):
        """
            Probability distribution function for Frechet's law.
        """
        return np.exp(-1/x)
    
    def _rpstable(self):
        """
            Simulation from a positive stable distribution.
            See Stephenson (2003) Section 3 for details.
        """

        if cexp==1: return 0
        tcexp = 1-self.theta
        u = np.random.uniform(size = 1) * math.pi
        w = math.log(np.random.exponential(size = 1))
        a = math.log(math.sin(tcexp*u)) + (self.theta / tcexp) * math.log(math.sin(self.theta*u)) - (1/tcexp) * math.log(math.sin(u))
        return (tcexp / self.theta) * (a-w)

class Extreme(Multivariate):
    """
        Base class for multivariate extreme value copulas.
        This class allows to use methods which use the Pickands dependence function.
    """

    def _l(self, u):
        """
            Return the value of the stable tail dependence function on u.
            Pickands is parameterize as A(w_2, \dots, w_d)
        """
        s = np.sum(u)
        u_ = u[1:]
        w_ = u_/s
        value_ = np.sum(u)*self._A(w_)
        return value_

    def _C(self, u):
        """
            Return the value of the copula taken on u
            .. math:: C(u) = exp(-\ell(-log(u_1), \dots, -log(u_d))), \quad u \in [0,1]^d

            Inputs
            ------
            u(list[float]) : list of float between 0 and 1
        """
        log_u_ = np.log(u)
        value_ = math.exp(-self._l(-log_u_))

    def _mu(self, u, j):
        """
            Return the value of the jth partial derivative of l

            Inputs
            ------
            u(list[float]) : list of float between 0 and 1
            j(int) : jth derivative of the stable tail dependence function
        """
        ## Besoin de définir les dérivées partielles de la Pickands

    def _dotC(self, u, j):
        """
            Return the value of \dot{C}_j taken on u
            .. math:: \dot{C}_j = C(u)/u_j * _mu_j(u)
        """


"""
    Define a multivariate Unit-Simplex
"""

def simplex(d, n = 50, a = 0, b = 1):
    """
        http://www.eirene.de/Devroye.pdf
        Algorithm page 207
    """
    if d==2:
        output = np.linspace(a,b,n)
        return np.c_[w,1-w]
    else:
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