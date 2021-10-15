import numpy as np
import math
import matplotlib.pyplot as plt
import mv_evd
import pandas as pd
plt.style.use('seaborn-whitegrid')

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

test = simplex(3, n = 10000)
print(test)
theta = 1.0

pickands = mv_evd.Logistic(theta = theta, d = 3)

#C_ = []
#
#"""
#    plot 2d
#"""
#for x in test:
#    C_.append(pickands.var_mado(x))
#
#print(C_)
#fig, ax = plt.subplots()
#ax.scatter(test[:,1], C_, s = 1)
#plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")

"""
    plot 3d
"""

A = []
for x_ in test:
    x = x_
    A.append(pickands.var_mado(x))

data = pd.DataFrame(np.c_[test[:,1], test[:,2],A])
data.columns = ['w2', 'w3', 'z']
data = data.sort_values(['w3'])
print(data)
fig = plt.figure()
ax = plt.figure().add_subplot(projection='3d')
ax.scatter3D(data['w2'],data['w3'],data["z"], alpha = 0.5, s = 1.0)
#ax.plot_trisurf(data['w2'],data["w3"],data["z"], alpha = 0.5,linewidth = 0.2, antialiased = True)
ax.set_xlabel(r'$w_2$')
ax.set_ylabel(r'$w_3$')
ax.set_zlabel(r'$\sigma^2$')
plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")#