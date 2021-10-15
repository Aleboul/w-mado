import numpy as np
import math

from base import Multivariate, CopulaTypes, Extreme

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
            s = self._rpstable(self.theta)
            for j in range(0,d):
                sim[i*self.d + j] = math.exp(self.theta * (s - math.log(np.random.exponential(size = 1))))
        return sim

    def rmvlog(self):
        sim = self._frechet(self.rmvlog_tawn())
        return sim.reshape(self.n_sample, self.d)