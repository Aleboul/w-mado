"""Archimedean module contains method for sampling from an archimedean copula.

Archimedean copulas are characterized by their generator function which is a continuous,
strictly decreasing function $\phi$ from $[0,1]$ to $[0,\infty]$. We note by $\phi^{\leftarrow}$ the
pseudo-inverse of $\phi$. Then the function from $[0,1]^2$ to $[0,1]$ given by

..math:: C(u,v) = \phi^{-1}(\phi(u) + \phi(v))

is an Archimedean copula. To sample from an archimedean copula, we need the generator function,
its pseudo inverse and also its derivative. For each class, these three quantities will be defined
as methods.

Structure :

- Archimedean copula (:py:class:`Archimedean`) from copy.bivariate.base.py
    - Clayton copula (:py:class:`Clayton`)
    - Ali-Mikhail-Haq copula (:py:class:`Amh`)
    - Gumbel copula (:py:class:`Gumbel`)
    - Frank copula (:py:class:`Frank`)
    - Joe copula (:py:class:`Joe`)
    - Nelsen_9 (:py:class:`Nelsen_9`) 
    - Nelsen_10 (:py:class:`Nelsen_10`)
    - Nelsen_11 (:py:class:`Nelsen_11`)
    - Nelsen_12 (:py:class:`Nelsen_12`)
    - Nelsen_13 (:py:class:`Nelsen_13`)
    - Nelsen_14 (:py:class:`Nelsen_14`)
    - Nelsen_15 (:py:class:`Nelsen_15`)
    - Nelsen_22 (:py:class:`Nelsen_22`)
"""

import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Clayton(Archimedean):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.CLAYTON
    theta_interval = [-1,float('inf')] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = \frac{1}{\theta}\left(t^{-\theta}-1\right), \quad 0 < t < 1
        """
        return (1.0/self.theta)*(np.power(t, -self.theta) - 1)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(u) = (1+\theta*t)^{-1/\theta}, \quad t \geq 0
        """
        return np.power((1.0+self.theta*t),-1/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = -(t)^{-\theta-1}, \quad 0 < t < 1
        """
        value_ = - np.power(t, -self.theta-1)
        return(value_)

class Amh(Archimedean):
    """Class for AMH copula model"""

    copula_type = CopulaTypes.AMH
    theta_interval = [-1.0,1.0] 
    invalid_thetas = [1.0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = log(\frac{1-\theta*(1-t)}{t}), \quad 0 < t < 1.
        """
        return np.log((1-self.theta*(1-t)) / t)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0.
        """
        value_1 = 1-self.theta
        value_2 = np.exp(t) - self.theta
        value_  = value_1 / value_2
        return(value_)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{\theta-1}{t(1-\theta*(1-t))}, \quad 0 < t < 1.
        """
        value_1 = self.theta-1
        value_2 = t*(1-self.theta*(1-t))
        return(value_1 / value_2)

class Gumbel(Archimedean):

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [1,float('inf')]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1.
        """
        return np.power(-np.log(t), self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(t) = e^{-t^{1/\theta}}, \quad t \geq 0.
        """
        return np.exp(-np.power(t,1/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t)).
        """
        value_1 = self.theta*np.power((-np.log(t)), self.theta)
        value_2 = t * np.log(t)
        value_  = value_1 / value_2
        return(value_)

class Frank(Archimedean):
    """Class for a frank copula model"""

    copula_type = CopulaTypes.FRANK
    theta_interval = [-float('inf'), float('inf')]
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1.
        """
        return -np.log((np.exp(-self.theta*t)-1) / (np.exp(-self.theta)-1))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(t) = (1-\theta) / (exp(t) - \theta), \quad t \geq 0.
        """
        return - (1 / self.theta)*np.log(1+np.exp(-t)*(np.exp(-self.theta)-1))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t)).
        """
        value_1 = self.theta * np.exp(-self.theta * t)
        value_2 = np.exp(-self.theta*t) - 1
        return(value_1 / value_2)

class Joe(Archimedean):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.JOE
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = -log(1-(1-t)^\theta), \quad 0 < t < 1.
        """
        return -np.log(1-np.power(1-t, self.theta))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(t) = 1-(1-exp(-t))^{1/\theta}, \quad t \geq 0.
        """
        return 1 - np.power(1-np.exp(-t), 1/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{-\theta(1-t)^{\theta-1}}{1-(1-t)^\theta}, \quad 0 < t < 1.
        """
        value_1 = -self.theta * np.power(1-t, self.theta-1)
        value_2 = 1 - np.power(1-t, self.theta) 
        return(value_1 / value_2)

class Nelsen_9(Archimedean):
    """Class for Nelsen_9 copula model"""

    copula_type = CopulaTypes.NELSEN_9
    theta_interval = [0,1] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = log(1-\theta*log(t)), \quad 0 < t < 1.
        """
        return np.log(1-self.theta*np.log(t))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(t) = exp(\frac{1-exp(t)}{\theta}), \quad t \geq 0.
        """
        return np.exp((1-np.exp(t))/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{\theta}{t * (\theta*log(t) - 1)}, \quad 0 < t < 1.
        """
        value_1 = self.theta
        value_2 = t * (self.theta * np.log(t) - 1)
        return(value_1 / value_2)

class Nelsen_10(Archimedean):
    """Class for Nelsen_10 copula model"""

    copula_type = CopulaTypes.NELSEN_10
    theta_interval = [0,1] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = log(2*(t)^{-\theta}-1), \quad 0 < t < 1.
        """
        return np.log(2*np.power(t,-self.theta)-1)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(t) = (\frac{exp(t)+1}{2})^{-1/\theta}, \quad t \geq 0.
        """
        value_1 = np.exp(t) + 1
        value_2 = 2
        return(np.power(value_1 / value_2, -1/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{-2*\theta*(t)^{-1-\theta}}{2t^{-\theta}-1}, \quad 0 < t < 1.
        """
        value_1 = -self.theta*2*np.power(t,-1-self.theta)
        value_2 = 2*np.power(t,-self.theta)-1
        return(value_1 / value_2)

class Nelsen_11(Archimedean):
    """Class for Nelsen_10 copula model"""

    copula_type = CopulaTypes.NELSEN_10
    theta_interval = [0,0.5] 
    invalid_thetas = [0.5]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = log(2-t^{\theta}), \quad 0 < t < 1.
        """
        return np.log(2-np.power(t,self.theta))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math::  \phi^\leftarrow(t) = (2 - exp(t))^{1/\theta}, \quad t \geq 0
        """
        return np.power(2-np.exp(t), 1/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{-\theta * t^{\theta - 1}}{2 - t^\theta}, \quad 0 < t < 1.
        """
        value_1 = -self.theta*np.power(t, self.theta-1)
        value_2 = 2 - np.power(t, self.theta)
        return(value_1 / value_2)

class Nelsen_12(Archimedean):
    """Class for Nelsen_12 copula model"""

    copula_type = CopulaTypes.NELSEN_12
    theta_interval = [0,float('inf')] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (\frac{1}{t} - 1)^\theta, \quad 0 < t < 1.
        """
        return np.power(1/t - 1 , self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math::  \phi^\leftarrow(t) = (1+t^{1/\theta})^{-1}, \quad t \geq 0.
        """
        value_1 = 1 + np.power(t, 1/self.theta)
        return(np.power(value_1,-1))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta * (-1/t^2) * (1/t - 1)^{\theta - 1}, \quad 0 < t < 1.
        """
        value_ = self.theta*(-1/np.power(t,2))*np.power(1/t - 1, self.theta - 1)
        return(value_)

class Nelsen_13(Archimedean):
    """Class for Nelsen_13 copula model"""

    copula_type = CopulaTypes.NELSEN_13
    theta_interval = [0,float('inf')] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (1-log(t))^{\theta}-1, \quad 0 < t < 1.
        """
        return np.power(1-np.log(t),self.theta)-1

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: 1 - (t+1)^{1/\theta}, \quad t \geq 0.
        """
        value_1 = 1-np.power(t+1,1/self.theta)
        return(np.exp(value_1))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = -\theta * (1-log(t))^{\theta-1}/t, \quad 0 < t < 1.
        """
        value_ = -self.theta*np.power(1-np.log(t),self.theta-1) / t
        return(value_)

class Nelsen_14(Archimedean):
    """Class for Nelsen_14 copula model"""

    copula_type = CopulaTypes.NELSEN_14
    theta_interval = [1.0,float('inf')] 
    invalid_thetas = [1.0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (t^{-1/\theta}-1)^\theta, \quad 0 < t < 1.
        """
        return np.power(np.power(t,-1/self.theta)-1, self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (t^{1/\theta}+1)^{-\theta}, \quad t \geq 0.
        """
        return np.power(np.power(t,1/self.theta)+1,-self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{-(t)^{-\theta-1}}{t^{-1/\theta}-1}, \quad 0 < t < 1.
        """
        value_1 = -np.power(t,-self.theta - 1)
        value_2 = np.power(t,-1/self.theta)-1
        return(value_1 * np.power(value_2,self.theta-1))

class Nelsen_15(Archimedean):
    """Class for Nelsen_15 copula model"""

    copula_type = CopulaTypes.NELSEN_15
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (1-t^{1/\theta})^\theta, \quad 0 < t < 1.
        """
        return np.power(1-np.power(t, 1/self.theta) , self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math::  \phi^\leftarrow(t) = (1-t^{1/\theta})^\theta, \quad t \geq 0.
        """
        return(np.power(1-np.power(t, 1/self.theta) , self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = -(t)^{1/\theta-1}(1-t^\theta)^{\theta-1}, \quad 0 < t < 1.
        """
        value_ = - np.power(t, 1/self.theta - 1)*np.power(1-np.power(t, self.theta), self.theta - 1)
        return(value_)

class Nelsen_22(Archimedean):
    """Class for Nelsen_22 copula model"""

    copula_type = CopulaTypes.NELSEN_22
    theta_interval = [0,1] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = arcsin(1-t^\theta), \quad 0 < t < 1.
        """
        return np.arcsin(1-np.power(t,self.theta))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-sin(t))^{1/\theta}, \quad t \geq 0.
        """
        return(np.power(1-np.sin(t),1/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \frac{-\theta*(t)^{\theta-1}}{(1-(t^\theta-1)^2)^0.5}
        """
        value_1 = - self.theta * np.power(t, self.theta-1)
        value_2 = np.power(1-np.power(np.power(t,self.theta)-1,2), 1/2)
        return(value_1 / value_2)


