"""Base module contains method for sampling from a bivariate copula and 
to compute the asymptotic variance of the \lambda-madogram with missing or complete data.

A bivariate copula $C : [0,1]^2 \rightarrow [0,1]$ of a random vector $(X,Y)$ allows us
to separate the effect of dependence from the effect of the marginal distributions (see [Sklar, 1959]).
The copula completely characterizes the stochastic dependence between $X$ and $Y$. 
An important class of copulas are Archimedean copulas and extreme value copulas (in the framework of 
extremes). Copulas belonging to the first class are characterized by a generator function. For the 
second class, copulas are characterized by the Pickands dependence function.

Structure :

- Bivariate copula (:py:class:`Bivariate`)
    - Archimedean copula (:py:class:`Archimedean`)
    - Extreme value copula (:py:class:`Extreme`)
"""

import numpy as np
import math

from enum import Enum
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

class CopulaTypes(Enum):
    """ Available copula families. """

    CLAYTON = 1
    AMH = 3
    GUMBEL = 4
    FRANK = 5
    JOE = 6
    NELSEN_9 = 9
    NELSEN_10 = 10
    NELSEN_12 = 12
    NELSEN_13 = 13
    NELSEN_14 = 14
    NELSEN_15 = 15
    NELSEN_22 = 22
    HUSSLER_REISS = 23
    ASYMMETRIC_LOGISTIC = 24
    ASYMMETRIC_NEGATIVE_LOGISTIC = 25
    ASSYMETRIC_MIXED_MODEL = 26
    STUDENT = 27

class Bivariate(object):
    """Base class for bivariate copulas.
    This class allows to instantiate all its subclasses and serves
    as a unique entry point for the bivariate copulas class.

    Attributes
    ----------
        copula_type(CopulaTypes)               : Family of the copula that belongs to
        theta_interval(list[float])            : Interval of valid thetas for the given copula family
        invalid_thetas(list[float])            : Values that, even though they belong to
                                                 :attr:`theta_interval`, shouldn't be considered valid.
        theta, psi1, psi2(float, float, float) : Parameters for the copula.
        n_sample (int)                         : length of the simulated dataset.
        random_seed (int)                      : seed for the random generator.
        
    Methods
    -------
        sample (np.array[float]) : sample where the uniform margins are inverted by
                                   a generalized inverse of a cdf.
    """

    copula_type = None
    theta_interval = []
    invalid_thetas = []
    theta = []
    n_sample = []
    psi1 = []
    psi2 = []
    random_seed = None

    def __init__(self, random_seed = None, theta = None, n_sample = None, psi1 = None, psi2 = None):
        """Initialize Bivariate object.

        Inputs
        ------
            random_seed (int or None)         : seed for the random generator.
            theta, psi1, psi2 (float or None) : parameters for the copula.
            n_sample (int)                    : length of the simulated dataset.
        """
        self.random_seed = random_seed
        self.n_sample = n_sample
        self.theta, self.psi1, self.psi2 = theta, psi1, psi2

    def check_theta(self):
        """Validate the theta inserted.

        Raises
        ------
            ValueError : If there is not in :attr:`theta_interval` or is in :attr:`invalid_thetas`.
        """
        lower, upper = self.theta_interval
        if (not lower <= self.theta <= upper) or (self.theta in self.invalid_thetas):
            message = 'The inserted theta value {} is out of limits for the given {} copula.'
            raise ValueError(message.format(self.theta, self.copula_type.name))

    def _generate_randomness(self):
        """Generate a bivariate sample draw identically and
        independently from a uniform over the segment [0,1].

        Output
        ------
            output (np.array([float]) with shape n_sample x 2) : a n_sample x 2 array with each component
                                                                 sampled from the desired copula under 
                                                                 the unit interval.
        """
        np.random.seed(self.random_seed)
        v_1 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample)
        v_2 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample)
        output = np.vstack([v_1, v_2]).T
        return output

    def sample(self, inv_cdf):
        """Draws a bivariate sample the desired copula and invert it by
        a given generalized inverse of cumulative distribution function

        Inputs
        ------
            inv_cdf : generalized inverse of cumulative distribution function

        Output
        ------
            output (np.array([float]) with sape n_sample x 2) : sample where the margins
                                                                are specified inv_cdf.

        Examples
        --------
            >>> copula = extreme_value_copula.Asy_log(theta = 2, psi1 = 1.0, psi2 = 1.0, n_samples = 1024)
            >>> sample = copula.sample()

            >>> fig, ax = plt.subplots()
            >>> ax.scatter(sample[:,0], sample[:,1], s = 1, col = 'salmon')
            >>> plt.show()
        """
        intput = self.sample_unimargin()
        output = np.zeros((self.n_sample,2))
        ncol = intput.shape[1]
        for i in range(0, ncol):
            output[:,i] = inv_cdf(intput[:,i])

        return output

class Archimedean(Bivariate):
    """Base class for bivariate archimedean copulas.
    This class allows to use methods which use the generator 
    function and its inverse.

    Methods
    -------
        sample_uni (np.array[float]) : sample the desired copula
                                       where the margins are uniform on [0,1].
        var_FMado ([float])          : give the asymptotic variance of the lambda-FMadogram 
                                       for an archimedean copula.
    
    Examples
    --------
        >>> import base
        >>> import Archimedean
        >>> archimedean = Archimedean.Clayton(theta = 1.0, n_sample = 1024)
    """

    def _C(self,u,v):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = \phi^\leftarrow (\phi(u) + \phi(v)), 0<u,v<1
        """
        value_ = self._generator_inv(self._generator(u) + self._generator(v))
        return value_
    def _dotC1(self,u,v):
        """Return the value of the first partial derivative taken on (u,v)
        .. math:: C(u,v) = \phi'(u) / \phi'(C(u,v)), 0<u,v<1
        """ 
        value_1 = self._generator_dot(u) 
        value_2 = self._generator_dot(self._C(u,v))
        return value_1 / value_2

    def _dotC2(self,u,v):
        """Return the value of the first partial derivative taken on (u,v)
        .. math:: C(u,v) = \phi'(v) / \phi'(C(u,v)), 0<u,v<1
        """
        value_1 = self._generator_dot(v) 
        value_2 = self._generator_dot(self._C(u,v))
        return value_1 / value_2

    def sample_unimargin(self):
        """Draws a bivariate sample from archimedean copula with uniform margins

        Output
        ------
            output (np.array([float]) with shape n_sample x 2) : sample from the desired
                                                                 copula with uniform margins.
        
        Examples
        --------
            >>> copula = Archimedean.Joe(theta = 2.0, n_sample = 1024)
            >>> sample = copula.sample_unimargin()
            
            >>> fig, ax = plt.subplots()
            >>> ax.scatter(sample[:,0], sample[:,1], s = 1, col = 'salmon')
            >>> plt.show()
        """
        self.check_theta()
        output = np.zeros((self.n_sample,2))
        X = self._generate_randomness()
        Epsilon = 1e-12
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = ( x - self._generator(x) / self._generator_dot(x)) - v[1]
                return(value_)
            sol = brentq(func, Epsilon,1-Epsilon)
            u = [self._generator_inv(v[0] * self._generator(sol)) , self._generator_inv((1-v[0])*self._generator(sol))]
            output[i,:] = u
        return output

    def _integrand_v1(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._C(min(u1,v1), min(u2,v2))
        value_2 = self._C(u1,u2)
        value_3 = self._C(v1,v2)
        value_  = value_1 - value_2 * value_3
        return(value_)

    def _integrand_v2(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC1(u1,u2)
        value_2 = self._dotC1(v1,v2)
        value_3 = (min(u1, v1) - u1*v1)
        value_  = value_1 * value_2 * value_3
        return(value_)
        
    def _integrand_v3(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC2(u1,u2)
        value_2 = self._dotC2(v1,v2)
        value_3 = (min(u2, v2) - u2*v2)
        value_  = value_1 * value_2 * value_3
        return(value_)

    def _integrand_cv12(self, x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC1(v1,v2)
        value_2 = self._C(min(u1,v1), u2) - self._C(u1,u2) * v1
        value_  = value_1 * value_2
        return(value_)

    def _integrand_cv13(self, x,y, lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC2(v1,v2)
        value_2 = self._C(u1, min(u2,v2)) - v2 * self._C(u1,u2)
        value_  = value_1 * value_2
        return(value_)

    def _integrand_cv23(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC1(u1,u2)
        value_2 = self._dotC2(v1,v2)
        value_3 = self._C(u1, v2) - u1 * v2
        value_  = value_1 * value_2 * value_3
        return(value_)

    def var_FMado(self,lmbd):
        """Compute the asymptotic variance of the lambda-FMadogram (see [Naveau et al., 2009]).

        .. math:: \nu(\lambda) = \frac{1}{2} \mathbb{E}\left[| F(X)^{1/\lambda} - G(Y)^{1/(1-\lambda)}|\right]

        Input
        -----
            lmbd (float) : float between 0 and 1 which correspond to the weight
                           of the lambda-FMadogram.
        """
        v1 = dblquad(lambda x,y : self._integrand_v1(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        v2 = dblquad(lambda x,y : self._integrand_v2(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        v3 = dblquad(lambda x,y : self._integrand_v3(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv12 = dblquad(lambda x,y : self._integrand_cv12(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv13 = dblquad(lambda x,y : self._integrand_cv13(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv23 = dblquad(lambda x,y : self._integrand_cv23(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        return(v1 + v2 + v3 - 2*cv12 - 2*cv13 + 2 * cv23)

class Extreme(Bivariate):
    """Base class for extreme value copulas.
    This class allows to use methods which use the Pickands dependence function.

    Methods
    -------
        sample_uni (np.array([float])) : sample where the margins are uniform on [0,1].
        true_FMado (float)             : true value of the lambda-FMadogram.
        var_FMado (float)              : give the asymptotic variance of the lambda-FMadogram
                                         for an extreme value copula.

    Examples
    --------
        >>> import base
        >>> import extreme_value_copula
        >>> evc = extreme_value_copula.Husler_Reiss(theta = 1.0, n_sample = 1024)
    """

    def _C(self,u,v):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = (uv)^{A(\frac{log(v)}{log(uv)})}, \quad 0<u,v<1
        """
        value_ = math.pow(u*v, self._A(math.log(v) / math.log(u*v)))
        return value_

    def _Kappa(self,t):
        """Return the value of Kappa taken on t
        .. math:: Kappa(t) = A(t) - t*A'(t), \quad 0<t<1
        """
        return (self._A(t) - t * self._Adot(t))

    def _Zeta(self, t):
        """Return the value of Zeta taken on t
        .. math:: Kappa(t) = A(t) + (1-t)*A'(t), \quad 0<t<1
        """
        return (self._A(t) + (1-t)*self._Adot(t))

    def _dotC1(self,u,v):
        """Return the value of \dot{C}_1 taken on (u,v)
        .. math:: \dot{C}_1  = (C(u,v) / u) * (Kappa(log(v)/log(uv))), \quad 0<u,v<1
        """
        t = math.log(v) / math.log(u*v)
        value_ = (self._C(u,v) / u) * self._Kappa(t)
        return(value_)

    def _dotC2(self, u,v):
        """Return the value of \dot{C}_2 taken on (u,v)
        .. math:: \dot{C}_2  = (C(u,v) / v) * (Zeta(log(v)/log(uv))), \quad 0<u,v<1
        """
        t = math.log(v) / math.log(u*v)
        value_ = (self._C(u,v) / v) * self._Zeta(t)
        return(value_)

    def sample_unimargin(self):
        """Draw a bivariate sample from an extreme value copula.
        Margins are uniform.

        Output
        ------
            output (np.array([float]) with shape n_sample x 2) : A sample from the desired extreme value copula
                                                                 with uniform margins.
        """
        self.check_theta()
        Epsilon = 1e-12
        output = np.zeros((self.n_sample,2))
        X = self._generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = self._dotC1(v[0],x) - v[1]
                return(value_)
            sol = brentq(func, Epsilon,1-Epsilon)
            u = [v[0], sol]
            output[i,:] = u
        return output

    def _integrand_ev1(self,s, lmbd):
        value_ = self._A(s) + (1-s)*(self._A(lmbd) / (1-lmbd) + lmbd/(1-lmbd) - 1) + s*(1-lmbd)/lmbd + 1
        return math.pow(value_,-2)

    def _integrand_ev2(self, s, lmbd):
        value_ = self._A(s)+ s * (self._A(lmbd) / (lmbd) + (1-lmbd) / lmbd -1 ) + (1-s) * lmbd/(1-lmbd) + 1
        return math.pow(value_,-2)

    def _integrand_ev3(self, s, lmbd):
        value_ = self._A(s) + (1-s) * (self._A(lmbd) / (1-lmbd) + lmbd/(1-lmbd)-1) + s*(self._A(lmbd) / (lmbd) + (1-lmbd)/lmbd - 1)+ 1
        return math.pow(value_, -2)

    def _integrand_sigma12(self, s, lmbd):
        value_ = self._A(s) + (1-s) * lmbd/(1-lmbd) + s * (1-lmbd) / lmbd + 1
        return math.pow(value_,-2)

    def true_FMado(self, lmbd):
        """Return the true value of the lambda-FMadogram
        
        Inputs
        ------
            lmbd (float) : weight of the lambda-FMadogram.
        """
        value_ = self._A(lmbd) / (1 + self._A(lmbd)) - 0.5 * ((1-lmbd) / (1+1-lmbd) + lmbd / (1+lmbd))
        return value_

    def var_FMado(self, lmbd, p_xy, p_x, p_y, corr = True):
        """Compute asymptotic variance of lambda-FMadogram using the specific form for
        bivariate extreme value copula.

        Inputs
        ------
            lmbd (float)                         : weight of the lambda-FMadogram.
            p_xy, p_x, p_y (float, float, float) : joint and margins probabilities of missing.
            corr (logical)                       : if True, return the asymptotic variance of the corrected version.
                                                   if False, return the asymptotic variance of the hybrid version.

        Output
        ------
            value_ (float) : value of the asymptotic variance of the lambda-FMadogram.

        Examples
        --------
            >>> lmbd = 0.5
            >>> copula = extreme_value_copula.Asy_log(theta = 2.0, psi1 = 1.0, psi2 = 1.0)
            >>> p_xy, p_x, p_y = 1.0, 1.0, 1.0 # no missing data

            >>> x = copula.var_FMado(lmbd, p_xy, p_x, p_y, corr = False)
            >>> print(x)
        """
        # value for sigma_1^2

        sigma_1 = (math.pow(p_xy,-1) - math.pow(p_x,-1)) * ((1-lmbd) / (2+1-lmbd)) * math.pow(1+1-lmbd,-2)

        # value for sigma_2^2

        sigma_2 = (math.pow(p_xy,-1) - math.pow(p_y,-1)) * (lmbd / (2+lmbd)) * math.pow(1+lmbd,-2)

        # Value for sigma_3^2

        gamma_1 = math.pow(1+self._A(lmbd),-2) * self._A(lmbd) / (2+ self._A(lmbd))
        gamma_2 = math.pow(self._Kappa(lmbd) / (1+self._A(lmbd)),2) * ((1-lmbd) / (2*self._A(lmbd) + 1 + lmbd))
        gamma_3 = math.pow(self._Zeta(lmbd) / (1+self._A(lmbd)),2) * (lmbd / (2*self._A(lmbd) + 1 + 1 - lmbd))
        value_1  = (math.pow(p_xy,-1) * gamma_1 + math.pow(p_x,-1) * gamma_2 + math.pow(p_y,-1)*gamma_3)

        value_21 = self._Kappa(lmbd) * math.pow(lmbd*(1-lmbd), -1) * quad(lambda s : self._integrand_ev1(s, lmbd), 0.0, lmbd)[0]
        value_22 = self._Kappa(lmbd) / (2 * math.pow(1+self._A(lmbd),2)) * ( (1-lmbd) / (2*self._A(lmbd) + 1 + lmbd) - 1)
        gamma_12 = value_21 + value_22
        value_2  = math.pow(p_x,-1) * gamma_12 ; 

        value_31 = self._Zeta(lmbd) * math.pow(lmbd*(1-lmbd),-1) * quad(lambda s : self._integrand_ev2(s,lmbd),lmbd,1.0)[0]
        value_32 = self._Zeta(lmbd) / (2 * math.pow(1+self._A(lmbd),2)) * ( lmbd / (2*self._A(lmbd) + 1 + 1 - lmbd) - 1)
        gamma_13 = value_31 + value_32
        value_3  = math.pow(p_y,-1)*gamma_13 ; 

        value_41 = quad(lambda s : self._integrand_ev3(s, lmbd), 0.0, 1.0)[0]
        value_42 = self._Kappa(lmbd) * self._Zeta(lmbd) * math.pow(1+self._A(lmbd),-2)
        gamma_23 = self._Kappa(lmbd) * self._Zeta(lmbd) * math.pow(lmbd*(1-lmbd),-1) * value_41 - value_42
        value_4  = (p_xy / (p_x*p_y)) * gamma_23 

        sigma_3  = value_1 - 2 * value_2 - 2 * value_3 + 2 * value_4

        # value for sigma_12

        value_11 = math.pow(lmbd*(1-lmbd),-1) * quad(lambda s : self._integrand_sigma12(s, lmbd), 0.0, 1.0)[0]
        value_12 = 1 / ((1+lmbd) * (1+1-lmbd))
        sigma_12 = (math.pow(p_xy,-1) - math.pow(p_x,-1) - math.pow(p_y,-1) + p_xy / (p_x*p_y)) * (value_11 - value_12)

        # value for sigma_13

        value_11 = math.pow(lmbd * (1-lmbd), -1) * quad(lambda s : self._integrand_sigma12(s,lmbd), 0.0, lmbd)[0]
        value_12 = math.pow(1+self._A(lmbd),-1) * (math.pow(2 + self._A(lmbd),-1) - math.pow(1+1-lmbd,-1))
        value_1  = (value_11 + value_12)

        value_21 = self._Zeta(lmbd) * math.pow(lmbd*(1-lmbd),-1) * quad(lambda s : self._integrand_ev2(s,lmbd), 0.0, 1.0)[0]
        value_22 = self._Zeta(lmbd) * math.pow(1 + self._A(lmbd),-1) * math.pow(1+1-lmbd,-1) 
        value_2  = (value_21 - value_22)
        
        sigma_13 = (math.pow(p_xy,-1) - math.pow(p_x,-1)) * value_1 - (math.pow(p_y,-1) - p_xy / (p_x*p_y)) * value_2 
        test_1 = (math.pow(p_y,-1) - p_xy / (p_x*p_y)) * value_2 
        # value for sigma_23

        value_11 = math.pow(lmbd * (1-lmbd), -1) * quad(lambda s : self._integrand_sigma12(s,lmbd), lmbd, 1.0)[0]
        value_12 = math.pow(1 + self._A(lmbd), -1) * (math.pow(2+self._A(lmbd),-1) - math.pow(1+lmbd,-1))
        value_1  = (value_11 + value_12)

        value_21 = self._Kappa(lmbd) * math.pow(lmbd*(1-lmbd),-1) * quad(lambda s : self._integrand_ev1(s,lmbd), 0.0, 1.0)[0]
        value_22 = self._Kappa(lmbd) * math.pow(1 + self._A(lmbd),-1) * math.pow(1+lmbd,-1)
        value_2  = (value_21 - value_22)

        sigma_23 = (math.pow(p_xy,-1) - math.pow(p_y,-1)) * value_1 - (math.pow(p_x,-1) - p_xy / (p_x*p_y)) * value_2
        test_2 = (math.pow(p_x,-1) - p_xy / (p_x*p_y)) * value_2
        if corr :
            value_   = math.pow(1+1-lmbd,2)*0.25 * sigma_1 + math.pow(1+lmbd,2)*0.25 * sigma_2 + sigma_3 + (1+lmbd)*(1+1-lmbd)*0.5 * sigma_12 - (1+1-lmbd)*sigma_13 - (1+lmbd)*sigma_23
        else :
            value_   = 0.25 * sigma_1 + 0.25 * sigma_2 + sigma_3 + 0.5 * sigma_12 - sigma_13 - sigma_23
        
        return value_
