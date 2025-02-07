# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:12:17 2025

@author: jtm44
"""

import torch
import numpy as np
import model
import matplotlib.pyplot as plt

def f(x):
    return torch.sin(5*x)-2*x+0.5*x**2
N=100
x=torch.rand(N).reshape(-1,1)*3
y=f(x)+torch.normal(mean=torch.zeros_like(x),std=0.1)

x1=torch.linspace(0,3,1000)
GP=model.sparse_GP(x,y,10)

#%%
GP.train(100000,0.01)
#%%
y1_mu,y1_sig=GP(x1)
y1_mu,y1_sig=y1_mu.detach().numpy(),y1_sig.detach().numpy()
y1_std = np.sqrt(np.diag(y1_sig))
plt.scatter(x,y)
plt.scatter(GP.Z.detach(),GP.m.detach(),marker='x')
plt.plot(x1,y1_mu)
plt.plot(x1,f(x1))
plt.fill_between(x1.squeeze(), 
                 y1_mu.squeeze() - 2 * y1_std.squeeze(), 
                 y1_mu.squeeze() + 2 * y1_std.squeeze(), 
                 color="blue", alpha=0.2, label="Confidence Interval (±2σ)")

# Labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()