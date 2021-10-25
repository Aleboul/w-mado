import numpy as np
import math
from scipy.integrate import quad, dblquad
from enum import Enum

"""
    Commentaires :
        Fixer une décision, w part de 0 ou 1.
        Ajouter des vérifications de contraintes.
"""

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
    ASYMMETRIC_LOGISTIC = 2

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

    def _frechet(self,x):
        """
            Probability distribution function for Frechet's law.
        """
        return np.exp(-1/x)
    
    def _rpstable(self):
        """
            Simulation from a positive stable distribution.
            See Stephenson (2003) Section 3 for details.
        """

        if self.theta==1: return 0
        tcexp = 1-self.theta
        u = np.random.uniform(size = 1) * math.pi
        w = math.log(np.random.exponential(size = 1))
        a = math.log(math.sin(tcexp*u)) + (self.theta / tcexp) * math.log(math.sin(self.theta*u)) - (1/tcexp) * math.log(math.sin(u))
        return (tcexp / self.theta) * (a-w)
    
    def sample(self, inv_cdf):
        sample_ = self.sample_unimargin()
        output = np.array([inv_cdf(sample_[:,j]) for j in range(0, self.d)])
        output = np.ravel(output).reshape(self.n_sample, self.d, order = 'F')
        return output.reshape(self.n_sample, self.d)

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
        return value_

    def _mu(self, u, j):
        """
            Return the value of the jth partial derivative of l

            Inputs
            ------
            u(list[float]) : list of float between 0 and 1
            j(int) : jth derivative of the stable tail dependence function
        """
        s = np.sum(u)
        u_ = u[1:]
        w_ = u_/s
        if j == 1 :
            deriv_ = []
            for i in range(2,self.d+1):
                value_deriv = self._Adot(w_, i) * w_[i-2] # i - 2 to start at zero which is w_2, we compute the array of derivative
                deriv_.append(value_deriv)
            value_ = self._A(w_) - np.sum(deriv_)
            return value_
        else :
            deriv_ = []
            for i in range(2, self.d+1):
                if i == j:
                    value_deriv = -(1-w_[i-2]) * self._Adot(w_, i)
                    deriv_.append(value_deriv)
                else:
                    value_deriv = self._Adot(w_, i) * w_[i-2]
                    deriv_.append(value_deriv)
            value_ = self._A(w_) - np.sum(deriv_)
            return value_

    def _dotC(self, u, j):
        """
            Return the value of \dot{C}_j taken on u
            .. math:: \dot{C}_j = C(u)/u_j * _mu_j(u)
        """
        value_ = (self._C(u) / u[j-1]) * self._mu(u,j)
        return value_

    def true_wmado(self, w):
        """
            Return the value of the w_madogram taken on w

            Inputs
            ------
            w (list of [float]) : element of the simplex
        """
        w_ = w[1:]
        value = self._A(w_) / (1+self._A(w_)) - (1/self.d)*np.sum(w / (1+w))
        return value


    def _integrand_ev1(self, s, w, j):
        """
            First integrand

            Inputs
            ------
            s(float) : float between 0 and 1
            w(list[float]) : d-array of the simplex
            j(int) : int \geq 1 
        """
        z = s*w / (1-w[j-1])
        w2d = w[1:]
        z[j-1] = (1-s) # start at 0 if j = 1
        z2d = z[1:]
        A_j = self._A(w2d) / w[j-1]
        value_ = self._A(z2d) + (1-s)*(A_j + (1-w[j-1])/w[j-1] - 1) + s*w[j-1] / (1-w[j-1])+1
        return math.pow(value_,-2)

    def _integrand_ev2(self, s, w, j, k):
        """
            Second integrand

            Inputs
            ------
            s(float) : float between 0 and 1
            w(list[float]) : d-array of the simplex
            j(int) : int \geq 1 
            k(int) : int \neq j
        """
        w2d = w[1:]
        z = 0 * w
        z[j-1] = (1-s)
        z[k-1] = s
        z2d = z[1:] 
        A_j = self._A(w2d) / w[j-1]
        A_k = self._A(w2d) / w[k-1]
        value_ = self._A(z2d) + (1-s) * (A_j + (1-w[j-1])/w[j-1] - 1) + s * (A_k + (1-w[k-1])/w[k-1] -1) + 1
        return math.pow(value_, -2)
    
    def _integrand_ev3(self,s, w, j, k):
        """
            Third integrand

            Inputs
            ------
            s(float) : float between 0 and 1
            w(list[float]) : d-array of the simplex
            j(int) : int \geq 1
            k(int) : int \neq j
        """
        w2d = w[1:] # w_{2:d} = (w_2, \dots, d_2)
        z = 0 * w
        z[j-1] = (1-s)
        z[k-1] = s
        z2d = z[1:]
        value_ = self._A(z2d) + (1-s) * (1-w[j-1])/w[j-1] + s * (1-w[k-1])/w[k-1]+1
        return math.pow(value_,-2)

    def _integrand_ev4(self, s, w, j):
        """
            thourth integrand

            Inputs
            ------
            s(float) : float between 0 and 1
            w(list[float]) : d-array of the simplex
            j(int) : int \geq 1
        """
        z = s*w / (1-w[j-1])
        w2d = w[1:]
        z[j-1] = (1-s) # start at 0 if j = 1
        z2d = z[1:]
        value_ = self._A(z2d) + (1-s) * (1-w[j-1])/w[j-1] + s*w[j-1]/(1-w[j-1])+1
        return math.pow(value_,-2)

    def _integrand_ev5(self, s, w, j,k):
        """
            thourth integrand

            Inputs
            ------
            s(float) : float between 0 and 1
            w(list[float]) : d-array of the simplex
            j(int) : int \geq 1
            k(int) : int \geq j
        """
        w2d = w[1:] # w_{2:d} = (w_2, \dots, d_2)
        z = 0 * w
        z[j-1] = (1-s)
        z[k-1] = s
        z2d = z[1:]
        A_k = self._A(w2d) / w[k-1]
        value_ = self._A(z2d) + (1-s) * (1-w[j-1])/w[j-1] + s * (A_k + (1-w[k-1])/w[k-1]-1)+1
        return math.pow(value_,-2)

    def _integrand_ev6(self, s, w, j,k):
        """
            thourth integrand

            Inputs
            ------
            s(float) : float between 0 and 1
            w(list[float]) : d-array of the simplex
            j(int) : int \geq k
            k(int) : int \geq 1
        """
        w2d = w[1:] # w_{2:d} = (w_2, \dots, d_2)
        z = 0 * w
        z[k-1] = (1-s)
        z[j-1] = s
        z2d = z[1:]
        A_k = self._A(w2d) / w[k-1]
        value_ = self._A(z2d) + (1-s) * (A_k + (1-w[k-1])/w[k-1]-1) + s * (1-w[j-1])/w[j-1]+1
        return math.pow(value_,-2)

    def var_mado(self, w, P,p):
        """
            Return the variance of the Madogram for a given point on the simplex

            Inputs
            ------
            w (list[float]) : array in the simplex .. math:: (w_1, \dots, w_d)
            P (array[float]) : d \times d array of probabilities, margins are in the diagonal
                               while probabilities of two entries may be missing are in the antidiagonal
            p ([float]) : joint probability of missing
        """
        w2d = w[1:]

        ## Calcul de .. math:: \sigma_{d+1}^2
        squared_gamma_1 = math.pow(p,-1)*(math.pow(1+self._A(w2d),-2) * self._A(w2d) / (2+self._A(w2d)))
        squared_gamma_ = []
        for i in range(0,self.d):
            j = i+1
            v_ = math.pow(P[i][i],-1)*(math.pow(self._mu(w,j) / (1+self._A(w2d)),2) * w[i] / (2*self._A(w2d) + 1 + 1 - w[i]))
            squared_gamma_.append(v_)
        gamma_1_ = []
        for i in range(0, self.d):
            j = i+1
            v_1 = self._mu(w,j) / (2 * math.pow(1+self._A(w2d),2)) * (w[i] / (2*self._A(w2d) + 1 + 1 - w[i]))
            v_2 = self._mu(w,j) / (2 * math.pow(1+self._A(w2d),2))
            v_3 = self._mu(w,j) / (w[i]*(1-w[i])) * quad(lambda s : self._integrand_ev1(s, w, j), 0.0, 1-w[i])[0]
            v_  = math.pow(P[i][i],-1)*(v_1 - v_2 + v_3)
            gamma_1_.append(v_)
        tau_ = []
        for k in range(1, self.d+1):
            for j in range(1, k):
                v_1 = self._mu(w,j) * self._mu(w,k) * math.pow(1+self._A(w2d),-2)
                v_2 = self._mu(w,j) * self._mu(w,k) / (w[j-1] * w[k-1]) * quad(lambda s : self._integrand_ev2(s, w, j, k), 0.0, 1.0)[0] ## COMMENT : Très moche les indices
                v_  = (P[j-1][k-1] / (P[j-1][j-1] * P[k-1][k-1]))*(v_2 - v_1) ## COMMENT : INDICES !!!
                tau_.append(v_)

        squared_sigma_d_1 = squared_gamma_1 + np.sum(squared_gamma_) - 2 * np.sum(gamma_1_) + 2 * np.sum(tau_)
        if p < 1:
            ## Calcul de .. math:: \sigma_{j}^2
            squared_sigma_ = []
            for i in range(0, self.d):
                v_ = (math.pow(p,-1) - math.pow(P[i][i],-1))*math.pow(1+w[i],-2) * w[i]/(2+w[i])
                squared_sigma_.append(v_)

            ## Calcul de .. math:: \sigma_{jk} with j < k
            sigma_ = []
            for k in range(1, self.d+1):
                for j in range(1,k):
                    v_1 = 1/(w[j-1] * w[k-1]) * quad(lambda s : self._integrand_ev3(s, w, j, k), 0.0,1.0)[0]
                    v_2 = 1/(1+w[j-1]) * 1/(1+w[k-1])
                    v_  = (math.pow(p,-1) - math.pow(P[j-1][j-1],-1) - math.pow(P[k-1][k-1],-1) + P[j-1][k-1]/(P[j-1][j-1]*P[k-1][k-1]))*(v_1 - v_2) # COMMENT : INDICES
                    sigma_.append(v_)

            ## Calcul de .. math:: \sigma_{j}^{(1)}, j \in \{1,dots,d\}
            sigma_1_ = []
            for i in range(0, self.d):
                j = i + 1
                v_1 = 1/(w[i] *(1-w[i])) * quad(lambda s : self._integrand_ev4(s, w, j), 0.0,1 - w[i])[0]
                v_2 = 1/(1+self._A(w2d)) * (1/(2+self._A(w2d)) - 1 / (1+w[i]))
                v_  = (math.pow(p,-1) - math.pow(P[i][i],-1))*(v_1 + v_2)
                sigma_1_.append(v_)

            sigma_2_ = []
            for k in range(1, self.d+1):
                for j in range(1, self.d+1):
                    if j == k:
                        v_ = 0
                        sigma_2_.append(v_)
                    elif j < k:
                        v_1 = self._mu(w,k) / (w[j-1] * w[k-1]) * quad(lambda s : self._integrand_ev5(s, w, j, k), 0.0,1.0)[0]
                        v_2 = self._mu(w,k) / (1+self._A(w2d)) * 1 /(1+w[j-1])
                        v_  = (math.pow(P[k-1][k-1],-1) - P[j-1][k-1]/(P[j-1][j-1]*P[k-1][k-1]))*(v_1 - v_2)
                        sigma_2_.append(v_)
                    else :
                        v_1 = self._mu(w,k) / (w[j-1] * w[k-1]) * quad(lambda s : self._integrand_ev6(s, w, j, k), 0.0,1.0)[0]
                        v_2 = self._mu(w,k) / (1+self._A(w2d)) * 1 /(1+w[j-1])
                        v_  = (math.pow(P[k-1][k-1],-1) - P[k-1][j-1]/(P[j-1][j-1]*P[k-1][k-1]))*(v_1 - v_2)
                        sigma_2_.append(v_)

            return (1/self.d**2) * np.sum(squared_sigma_) + squared_sigma_d_1 + (2/self.d**2) * np.sum(sigma_) - (2/self.d) * np.sum(sigma_1_) + (2/self.d) * np.sum(sigma_2_)
        
        else:
            return squared_sigma_d_1


"""
    Define a multivariate Unit-Simplex
"""

def simplex(d, n = 50, a = 0, b = 1):
    """
        http://www.eirene.de/Devroye.pdf
        Algorithm page 207
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