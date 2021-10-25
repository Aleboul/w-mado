import numpy as np
import pandas as pd
from tqdm import tqdm

class Monte_Carlo(object):
    """
        Base class for Monte-Carlo simulations

        Inputs
        ------
        			           n_iter (int) : number of Monte Carlo simulation
			n_sample (list of int or [int]) : multiple length of sample
			                   lmbd (float) : value of lambda	
			 random_seed (Union[int, None]) : seed for the random generator
						  P (array [float]) : d \times d array of probabilities of presence
							copula (object) : law of the vector of the uniform margin
					   copula_miss (object) : dependence modeling of the couple (I,J)
                                w ([float]) : vector of weight belonging to the simplex
        Attributes
		----------
			          n_sample (list[int]) : several lengths used for estimation
							  lmbd (float) : parameter for the lambda-FMadogram
			lmbd_interval ([lower, upper]) : interval of valid thetas for the given lambda-FMadogram
						  simu (DataFrame) : results of the Monte-Carlo simulation
    """

    n_iter = None
    n_sample = []
    w = []
    random_seed = None
    copula = None
    copula_miss = None
    d = None

    def __init__(self, n_iter = None, n_sample = [], w = [], random_seed = [], copula = None, P = None, copula_miss = None, d = None):
        """
            Initialize Monte_Carlo object
        """

        self.n_iter = n_iter
        self.n_sample = n_sample
        self.w = w
        self.copula = copula
        self.P = P
        self.copula_miss = copula_miss
        self.d = self.copula.d

    def check_w(self):
        """
            Validate the weights inserted
            Raises :
                ValueError : If w is not in simplex
        """
        if (not len(self.w) == self.d and not np.sum(self.w) == 1):
            message = "The w value {} is not in the simplex."
            raise ValueError(message.format(self.w))

    def _ecdf(self, data, miss):
        """
            Compute ECDF

            Inputs
            ------
            data (list([float])) : array of observations

            Outputs
            -------
            Empirical uniform margin
        """

        index = np.argsort(data)
        ecdf  = np.zeros(len(index))
        for i in index:
            ecdf[i] = (1.0 / np.sum(miss)) * np.sum((data <= data[i]) * miss)
        return ecdf

    def _wmado(self, X, miss, corr) :
        """
            This function computes the w-madogram

            Inputs
            ------
            X (array([float]) of n_sample \times d) : a matrix
                                                  w : element of the simplex
                                miss (array([int])) : list of observed data
                               corr (True or False) : If true, return corrected version of w-madogram
            
            Outputs
            -------
            w-madogram
        """
        Nnb = X.shape[1]
        Tnb = X.shape[0]

        V = np.zeros([Tnb, Nnb])
        cross = np.ones(Tnb)
        for j in range(0, Nnb):
            cross *= miss[:,j]
            X_vec = np.array(X[:,j])
            Femp = self._ecdf(X_vec, miss[:,j])
            V[:,j] = np.power(Femp, 1/self.w[j])
            
        V *= cross.reshape(Tnb,1)
        if corr == True:
            return None
        else :
            value_1 = np.amax(V,1)
            value_2 = (1/self.copula.d) * np.sum(V, 1)
            mado = (1/(np.sum(cross))) * np.sum(value_1 - value_2)

        return mado

    def _gen_missing(self) :
        """
			This function returns an array max(n_sample) \times 2 of binary indicate missing of X or Y.
			Dependence between (I,J) is given by copula_miss. The idea is the following
			I \sim Ber(P[0][0]), J \sim Ber(P[1][1]) and (I,J) \sim Ber(copula_miss(P[0][0], P[1][1])).
			
			We simulate it by generating a sample (U,V) of length max(n_sample) from copula_miss.

			Then, X = 1 if U \leq P[0][0] and Y = 1 if V \leq P[1][1]. These random variables are indeed Bernoulli.
			
			Also \mathbb{P}(X = 1, Y = 1) = \mathbb{P}(U\leq P[0][0], V \leq P[1][1]) = C(P[0][0], P[1][1])
		"""
        if self.copula_miss is None:
        	return np.array([np.random.binomial(1, self.P[j][j], np.max(self.n_sample)) for j in range(0,self.d)]).reshape(np.max(self.n_sample), self.d)
        else :
        	sample_ = self.copula_miss.sample_unimargin()
        	miss_ = np.array([1 * (sample_[:,j] <= self.P[j][j]) for j in range(0,self.d)])
        	return miss_
    
    def finite_sample(self, inv_cdf, corr = {False, True}):
        """
            Perform Monte Carlo simulation to obtain empirical counterpart of the wmadogram
        """
        
        output = []

        for m in range(self.n_iter):
            wmado_store = np.zeros(len(self.n_sample))
            obs_all = self.copula.sample(inv_cdf)
            miss_all = self._gen_missing()
            for i in range(0, len(self.n_sample)):
                obs = obs_all[:self.n_sample[i]]
                miss = miss_all[:self.n_sample[i]]
                wmado = self._wmado(obs, miss, corr)
                wmado_store[i] = wmado
            
            output_cbind = np.c_[wmado_store, self.n_sample, np.arange(len(self.n_sample))]
            output.append(output_cbind)
        df_wmado = pd.DataFrame(np.concatenate(output))
        df_wmado.columns = ['wmado', 'n', 'gp']
        df_wmado['scaled'] = (df_wmado.wmado - self.copula.true_wmado(self.w)) * np.sqrt(df_wmado.n)
        return(df_wmado)