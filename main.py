# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:12:17 2025

@author: jtm44
"""

import torch
import numpy as np
import model
import matplotlib.pyplot as plt


##define a nonlinear function to learn
def f(x):
    return torch.sin(6*x)-2*x+0.5*x**2

#generate training data with N points with a bit of noise
N=100
x=torch.rand(N).reshape(-1,1)*3
y=f(x)+torch.normal(mean=torch.zeros_like(x),std=0.1)

#%%

####choose either the exact or sparse GP

##create the sparse GP
GP=model.sparse_GP(x,y,30)

#%%
##or create the exact gp
GP=model.exact_GP(x, y)
#%%
GP.train(100000,0.001)
#%%

###Generate test data
x1=torch.linspace(0,3,1000)
y1=f(x1)
##make predictions
y1_mu,y1_sig=GP(x1)

y1_mu,y1_sig=y1_mu.detach().numpy(),y1_sig.detach().numpy()
y1_std = np.sqrt(np.diag(y1_sig))


#plot results
plt.scatter(x,y,label='Observations')

try:
    GP.Z
    print('Approx GP')
    plt.scatter(GP.Z.detach(),GP.m.detach(),marker='x',label='Inducing Points')  ##if a sparse GP plot the inducing points
except Exception as e:
    print('Exact GP')
else:
    pass


plt.plot(x1,y1_mu,label='Mean Prediction')
plt.plot(x1,y1,label='Ground truth')
plt.fill_between(x1.squeeze(), 
                 y1_mu.squeeze() - 2 * y1_std.squeeze(), 
                 y1_mu.squeeze() + 2 * y1_std.squeeze(), 
                 color="blue", alpha=0.2, label="Confidence Interval (±2σ)")

# Labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()