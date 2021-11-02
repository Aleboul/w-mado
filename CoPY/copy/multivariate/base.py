"""Base module contains method for sampling from a multivariate extreme value copula and
to compute the asymptotic variance of the w-madogram with missing or complete data.

A multivariate copula $C : [0,1]^d \rightarrow [0,1]$ of a d-dimensional random vector $X$ allows
us to separate the effect of dependence from the effect of the marginal distributions. The
copula function completely chracterizes the stochastic dependence between the margins of $X$.
Extreme value copulas are characterized by the stable tail dependence function which the restriction
to the unit simplex is called Pickands dependence function.

Structure :

- Multivariate copula (:py:class:`Multivariate`)
    - Extreme value copula (:py:class:`Extreme`)
"""

import numpy as np
import math

from scipy.integrate import quad
from enum import Enum

class CopulaTypes(Enum):
    """
        Available multivariate copula
    """

    LOGISTIC = 1
    ASYMMETRIC_LOGISTIC = 2

class Multivariate(object):
    """Base class for multivariate copulas.
    This class allows to instantiate all its subclasses and serves
    as a unique entry point for the multivariate copulas classes.
    It permit also to compute the variance of the w-madogram for a given point.

    Attributes
    ----------
        copula_type                 : substype of the copula
        random_seed                 : seed for the random generator
        d                           : dimension
        copula_type(CopulaTypes)    : family of the copula that belongs to
        theta_interval(list[float]) : interval of valid theta for the given copula family
        invalid_thetas(list[float]) : values, that even though they belong to
                                      :attr: `theta_interval`, shouldn't be considered as valid.
        theta(list[float])          : parameter for the parametric copula
        var_mado(list[float])       : value of the theoretical variance for a given point in the simplex

    Methods
    -------
        sample (np.array([float])) : array of shape n_sample x d of the desired multivariate copula model
                                     where the margins are inverted by the specified generalized inverse 
                                     of a cdf.
    """
    copula_type = None
    theta_interval = []
    invalid_thetas = []
    n_sample = []
    asy = []
    theta = []
    d = []

    def __init__(self, random_seed = None, theta = None, n_sample = None, asy = None, d = None):
        """Initialize Multivariate object.
            
        Inputs
        ------
            copula_type (copula_type or st) : subtype of the copula
            random_seed (int or None)       : seed for the random generator
            theta (list[float] or None)     : list of parameters for the parametric copula
            asy (list[float] or None)       : list of asymetric values for the copula
            n_sample (int or None)          : number of sampled observation
            d (int or None)                 : dimension
        """
        self.random_seed = random_seed
        self.theta = theta
        self.n_sample = n_sample
        self.asy = asy
        self.d = d

    def _frechet(self,x):
        """
            Probability distribution function for Frechet's law.
        """
        return np.exp(-1/x)
    
    def sample(self, inv_cdf):
        """Draws a bivariate sample the desired copula and invert it by
        a given generalized inverse of cumulative distribution function

        Inputs
        ------
            inv_cdf : generalized inverse of cumulative distribution function

        Output
        ------
            output (np.array([float]) with sape n_sample x d) : sample where the margins
                                                                are specified inv_cdf.
        """
        sample_ = self.sample_unimargin()
        output = np.array([inv_cdf(sample_[:,j]) for j in range(0, self.d)])
        output = np.ravel(output).reshape(self.n_sample, self.d, order = 'F')
        return output

class Extreme(Multivariate):
    """Base class for multivariate extreme value copulas.
    This class allows to use methods which use the Pickands dependence function.

    Methods
    -------
        sample_uni (np.array([float])) : sample from the desired multivariate copula model. 
                                         Margins are uniform on [0,1].
        var_FMado (float)              : gives the asymptotic variance of w-madogram for a
                                         multivariate extreme value copula.
    
    Examples
    --------
        >>> import base
        >>> import mv_evd
        >>> import matplotlib.pyplot as plt

        >>> copula = mv_evd.Logistic(theta = 0.5, d = 3, n_sample = 1024)
        >>> sample = copula.sample_uni()

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection = '3d')
        >>> ax.scatter3D(sample[:,0],sample[:,1],sample[:,2], c = 'lightblue', s = 1.0, alpha = 0.5)
        >>> plt.show()
    """

    def _l(self, u):
        """Return the value of the stable tail dependence function on u.
        Pickands is parametrize as A(w_0, \dots, w_{d-1}) with w_0 = 1-\sum_{j=1}^{d-1} w_j

        Inputs
        ------
            u (list[float]) : d-list with each components between 0 and 1.
        """
        s  = np.sum(u)
        w_ = u / s
        value_ = s*self._A(w_)
        return value_

    def _C(self, u):
        """Return the value of the copula taken on u
        .. math:: C(u) = exp(-\ell(-log(u_1), \dots, -log(u_d))), \quad u \in [0,1]^d.

        Inputs
        ------
            u (list[float]) : d-list of float between 0 and 1.
        """
        log_u_ = np.log(u)
        value_ = math.exp(-self._l(-log_u_))
        return value_

    def _mu(self, u, j):
        """Return the value of the jth partial derivative of l.
        ..math:: \dot{\ell}_j(u), \quad u \in ]0,1[^d, \quad, j \in \{0,\dots,d-1\}.

        Inputs
        ------
            u (list[float]) : list of float between 0 and 1.
            j (int)         : jth derivative of the stable tail dependence function.
        """
        s = np.sum(u)
        w_ = u / s
        if j == 0 :
            deriv_ = []
            for j in range(1,self.d):
                value_deriv = self._Adot(w_, j) * w_[j]
                deriv_.append(value_deriv)
            value_ = self._A(w_) - np.sum(deriv_)
            return value_
        else :
            deriv_ = []
            for i in range(1, self.d):
                if i == j:
                    value_deriv = -(1-w_[i]) * self._Adot(w_, i)
                    deriv_.append(value_deriv)
                else:
                    value_deriv = self._Adot(w_, i) * w_[i]
                    deriv_.append(value_deriv)
            value_ = self._A(w_) - np.sum(deriv_)
            return value_

    def _dotC(self, u, j):
        """Return the value of \dot{C}_j taken on u.
        .. math:: \dot{C}_j = C(u)/u_j * _mu_j(u), \quad u \in [0,1]^d, \quad j \in \{0 , \dots, d-1\}.
        """
        value_ = (self._C(u) / u[j]) * self._mu(u,j)
        return value_

    def true_wmado(self, w):
        """Return the value of the w_madogram taken on w.

        Inputs
        ------
            w (list of [float]) : element of the simplex.
        """
        value = self._A(w) / (1+self._A(w)) - (1/self.d)*np.sum(w / (1+w))
        return value


    def _integrand_ev1(self, s, w, j):
        """First integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq 0.
        """
        z = s*w / (1-w[j])
        z[j] = (1-s) # start at 0 if j = 1
        A_j = self._A(w) / w[j]
        value_ = self._A(z) + (1-s)*(A_j + (1-w[j])/w[j] - 1) + s*w[j] / (1-w[j])+1
        return math.pow(value_,-2)

    def _integrand_ev2(self, s, w, j, k):
        """Second integrand.

        Inputs
        ------
            s (float)       : float between 0 and 1.
            w (list[float]) : d-array of the simplex.
            j (int)         : int \geq 0.
            k (int)         : int \neq j.
        """
        z = 0 * w
        z[j] = (1-s)
        z[k] = s
        A_j = self._A(w) / w[j]
        A_k = self._A(w) / w[k]
        value_ = self._A(z) + (1-s) * (A_j + (1-w[j])/w[j] - 1) + s * (A_k + (1-w[k])/w[k] -1) + 1
        return math.pow(value_, -2)
    
    def _integrand_ev3(self,s, w, j, k):
        """Third integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq 0.
            k(int)         : int \neq j.
        """
        z = 0 * w
        z[j] = (1-s)
        z[k] = s
        value_ = self._A(z) + (1-s) * (1-w[j])/w[j] + s * (1-w[k])/w[k]+1
        return math.pow(value_,-2)

    def _integrand_ev4(self, s, w, j):
        """Fourth integrand.

        Inputs
        ------
            s (float)       : float between 0 and 1.
            w (list[float]) : d-array of the simplex.
            j (int)         : int \geq 0.
        """
        z = s*w / (1-w[j])
        z[j] = (1-s)
        value_ = self._A(z) + (1-s) * (1-w[j])/w[j] + s*w[j]/(1-w[j])+1
        return math.pow(value_,-2)

    def _integrand_ev5(self, s, w, j,k):
        """Fifth integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq 1.
            k(int)         : int \geq j.
        """
        z = 0 * w
        z[j] = (1-s)
        z[k] = s
        A_k = self._A(w) / w[k]
        value_ = self._A(z) + (1-s) * (1-w[j])/w[j] + s * (A_k + (1-w[k])/w[k]-1)+1
        return math.pow(value_,-2)

    def _integrand_ev6(self, s, w, j,k):
        """Sixth integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq k.
            k(int)         : int \geq 0.
        """
        z = 0 * w
        z[k] = (1-s)
        z[j] = s
        A_k = self._A(w) / w[k]
        value_ = self._A(z) + (1-s) * (A_k + (1-w[k])/w[k]-1) + s * (1-w[j])/w[j]+1
        return math.pow(value_,-2)

    def var_mado(self, w, P, p, corr = {False, True}):
        """Return the variance of the Madogram for a given point on the simplex

        Inputs
        ------
            w (list[float])  : array in the simplex .. math:: (w_0, \dots, w_{d-1}).
            P (array[float]) : d \times d array of probabilities, margins are in the diagonal
                               while probabilities of two entries may be missing are in the antidiagonal.
            p ([float])      : joint probability of missing.
        """

        if corr :
            lambda_ = w
        else :
            lambda_ = np.zeros(self.d)

        ## Calcul de .. math:: \sigma_{d+1}^2
        squared_gamma_1 = math.pow(p,-1)*(math.pow(1+self._A(w),-2) * self._A(w) / (2+self._A(w)))
        squared_gamma_ = []
        for j in range(0,self.d):
            v_ = math.pow(P[j][j],-1)*(math.pow(self._mu(w,j) / (1+self._A(w)),2) * w[j] / (2*self._A(w) + 1 + 1 - w[j]))
            squared_gamma_.append(v_)
        gamma_1_ = []
        for j in range(0, self.d):
            v_1 = self._mu(w,j) / (2 * math.pow(1+self._A(w),2)) * (w[j] / (2*self._A(w) + 1 + 1 - w[j]))
            v_2 = self._mu(w,j) / (2 * math.pow(1+self._A(w),2))
            v_3 = self._mu(w,j) / (w[j]*(1-w[j])) * quad(lambda s : self._integrand_ev1(s, w, j), 0.0, 1-w[j])[0]
            v_  = math.pow(P[j][j],-1)*(v_1 - v_2 + v_3)
            gamma_1_.append(v_)
        tau_ = []
        for k in range(0, self.d):
            for j in range(0, k):
                v_1 = self._mu(w,j) * self._mu(w,k) * math.pow(1+self._A(w),-2)
                v_2 = self._mu(w,j) * self._mu(w,k) / (w[j] * w[k]) * quad(lambda s : self._integrand_ev2(s, w, j, k), 0.0, 1.0)[0]
                v_  = (P[j][k] / (P[j][j] * P[k][k]))*(v_2 - v_1)
                tau_.append(v_)

        squared_sigma_d_1 = squared_gamma_1 + np.sum(squared_gamma_) - 2 * np.sum(gamma_1_) + 2 * np.sum(tau_)
        if p < 1:
            ## Calcul de .. math:: \sigma_{j}^2
            squared_sigma_ = []
            for j in range(0, self.d):
                v_ = (math.pow(p,-1) - math.pow(P[j][j],-1))*math.pow(1+w[j],-2) * w[j]/(2+w[j])
                v_ = math.pow(1+lambda_[j]*(self.d-1),2) * v_
                squared_sigma_.append(v_)

            ## Calcul de .. math:: \sigma_{jk} with j < k
            sigma_ = []
            for k in range(0, self.d):
                for j in range(0,k):
                    v_1 = 1/(w[j] * w[k]) * quad(lambda s : self._integrand_ev3(s, w, j, k), 0.0,1.0)[0]
                    v_2 = 1/(1+w[j]) * 1/(1+w[k])
                    v_  = (math.pow(p,-1) - math.pow(P[j][j],-1) - math.pow(P[k][k],-1) + P[j][k]/(P[j][j]*P[k][k]))*(v_1 - v_2)
                    v_  = (1+lambda_[j]*(self.d-1)) * (1+lambda_[k]*(self.d-1)) * v_
                    sigma_.append(v_)

            ## Calcul de .. math:: \sigma_{j}^{(1)}, j \in \{1,dots,d\}
            sigma_1_ = []
            for j in range(0, self.d):
                v_1 = 1/(w[j] *(1-w[j])) * quad(lambda s : self._integrand_ev4(s, w, j), 0.0,1 - w[j])[0]
                v_2 = 1/(1+self._A(w)) * (1/(2+self._A(w)) - 1 / (1+w[j]))
                v_  = (math.pow(p,-1) - math.pow(P[j][j],-1))*(v_1 + v_2)
                v_  = (1+lambda_[j]*(self.d-1))*v_
                sigma_1_.append(v_)

            sigma_2_ = []
            for k in range(0, self.d):
                for j in range(0, self.d):
                    if j == k:
                        v_ = 0
                        sigma_2_.append(v_)
                    elif j < k:
                        v_1 = self._mu(w,k) / (w[j] * w[k]) * quad(lambda s : self._integrand_ev5(s, w, j, k), 0.0,1.0)[0]
                        v_2 = self._mu(w,k) / (1+self._A(w)) * 1 /(1+w[j])
                        v_  = (math.pow(P[k][k],-1) - P[j][k]/(P[j][j]*P[k][k]))*(v_1 - v_2)
                        v_  = (1 + lambda_[j]*(self.d-1))*v_
                        sigma_2_.append(v_)
                    else :
                        v_1 = self._mu(w,k) / (w[j] * w[k]) * quad(lambda s : self._integrand_ev6(s, w, j, k), 0.0,1.0)[0]
                        v_2 = self._mu(w,k) / (1+self._A(w)) * 1 /(1+w[j])
                        v_  = (math.pow(P[k][k],-1) - P[k][j]/(P[j][j]*P[k][k]))*(v_1 - v_2)
                        v_  = (1 + lambda_[j]*(self.d-1))*v_
                        sigma_2_.append(v_)

            if corr :
                return (1/self.d**2) * np.sum(squared_sigma_) + squared_sigma_d_1 + (2/self.d**2) * np.sum(sigma_) - (2/self.d) * np.sum(sigma_1_) + (2/self.d) * np.sum(sigma_2_)
            else :
                return (1/self.d**2) * np.sum(squared_sigma_) + squared_sigma_d_1 + (2/self.d**2) * np.sum(sigma_) - (2/self.d) * np.sum(sigma_1_) + (2/self.d) * np.sum(sigma_2_)
        
        else:
            return squared_sigma_d_1