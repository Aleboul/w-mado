import numpy as np
import math
import matplotlib.pyplot as plt
import mv_evd
import base
import monte_carlo
import pandas as pd
from scipy.stats import norm
plt.style.use('seaborn-whitegrid')
from tqdm import tqdm

def max(a,b):
    if a >= b:
        return a
    else:
        return b

d = 3

n_iter = 100
#asy = [0.7,0.1,0.6,[0.1,0.2], [0.1,0.1], [0.4,0.1], [0.1,0.3,0.2]]
theta = 1.0
n_sample = [512]
copula = mv_evd.Logistic(theta = theta, d = d, n_sample = np.max(n_sample))
P = np.ones((d,d))
p = 1.0

W2 = np.linspace(0.01,0.99,100)
W3 = np.linspace(0.01,0.99,100)

#sigma_theo = []
#sigma_empi = []
#for w3 in tqdm(W3):
#    for w2 in W2: 
#        if(w2 + w3 >= 0.99):
#            sigma_empi_ = np.nan
#            sigma_theo_ = np.nan
#            sigma_empi.append(sigma_empi_)
#            sigma_theo.append(sigma_theo_)
#        else:
#            w1 = 1-w2-w3
#            x = np.array([w1,w2,w3])
#            Monte = monte_carlo.Monte_Carlo(n_iter = n_iter, n_sample = n_sample, w = x, copula = copula, P = P)
#            df_wmado = Monte.finite_sample(norm.ppf)
#            sigma_empi_ = df_wmado['scaled'].var()
#            sigma_theo_ = max(0,copula.var_mado(x,P, p))
#            sigma_empi.append(sigma_empi_)
#            sigma_theo.append(sigma_theo_)
#
#output = np.c_[sigma_empi, sigma_theo]
#df_wmado = pd.DataFrame(output)
#df_wmado.columns = ['empirical_sigma', 'theoretical_sigma']
#df_wmado.to_csv("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado_asy.csv")

df_wmado = pd.read_csv("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado_logistic_0.5_nsample512_niter100.csv")
print(df_wmado)

W2, W3 = np.meshgrid(W2, W3)
empirical_sigma = np.array(df_wmado['empirical_sigma'])
theoretical_sigma = np.array(df_wmado['theoretical_sigma'])
gap_sigma = np.abs(empirical_sigma - theoretical_sigma)
EMPIRICAL = empirical_sigma.reshape(W2.shape)
THEORETICAL = theoretical_sigma.reshape(W2.shape)
GAP = gap_sigma.reshape(W2.shape)
print(EMPIRICAL.shape)
print(THEORETICAL.shape)
print(W2.shape)
print(W3.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W2, W3, THEORETICAL, alpha = 0.5, color = 'black')
ax.scatter(W2, W3, EMPIRICAL, alpha = 0.5, color = 'lightgrey', s = 1)
ax.set_xlabel(r'$w_2$')
ax.set_ylabel(r'$w_3$')
ax.set_zlabel(r"$\sigma^2$")

ax.azim = -140
ax.dist = 10
ax.elev = 10

plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")

#levels = [0.0,0.007,0.014,0.021,0.028,0.035]
#fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
#contourf_ = ax[0].contourf(W2, W3, THEORETICAL, 4, cmap = 'Greys', linestyles = 'dashed', levels = levels)
#contourf_ = ax[1].contourf(W2, W3, EMPIRICAL, 4, cmap = 'Greys', linestyles = 'dashed', levels =levels)
#cbar = fig.colorbar(contourf_, ax=ax.ravel().tolist(), shrink=0.95)
#ax[0].set_xlim(0,1)
#ax[0].set_ylim(0,1)
#ax[0].set_xlabel(r'$w_2$')
#ax[0].set_ylabel(r'$w_3$')
#ax[1].set_xlim(0,1)
#ax[1].set_ylim(0,1)
#ax[1].set_xlabel(r'$w_2$')
#ax[1].set_ylabel(r'$w_3$')
#plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")

#fig, ax = plt.subplots()
#contourf_ = ax.contourf(W2, W3, GAP, 4, cmap = 'Greys')
#cbar = fig.colorbar(contourf_)
#ax.set_xlim(0,1)
#ax.set_ylim(0,1)
#ax.set_xlabel(r'$w_2$')
#ax.set_ylabel(r'$w_3$')
#plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")
