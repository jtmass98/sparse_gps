# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:09:22 2025

@author: jtm44
"""

import numpy
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class GP(nn.Module):
    def __init__(self,X,y):
        super(GP, self).__init__()
        self.X=X.reshape(-1,1)
        
        self.y=y.reshape(-1,1)
        self.N=self.X.shape[0]
        self.l=torch.nn.Parameter(torch.Tensor([0.1]))
        self.sig=torch.nn.Parameter(torch.Tensor([1]))
        self.sig_n=torch.nn.Parameter(torch.Tensor([0.1]))
    
    def mle(self):
        K = self.rbf(self.X,self.X)+torch.eye(self.N)*self.sig_n**2
        
        # Compute Cholesky decomposition (for stability)
        L=torch.linalg.inv(K)  # Lower triangular matrix (N, N)
    
        # Solve for alpha = (K + sigma^2 I)^{-1} y using Cholesky
        alpha = torch.matmul(L,self.y)  # Equivalent to (K + sigma^2 I)^{-1} y
        detter=torch.linalg.slogdet(K)[1]
        # Compute log marginal likelihood
        lml = -0.5 * torch.dot(self.y.squeeze(), alpha.squeeze()) - 0.5*detter - 0.5 * self.N * torch.log(torch.tensor(2 * torch.pi))

        return -lml
    
    
    def train(self,num_epochs=1000):
        #initialise by making into parameters
    

        
        

        lr=0.01
        optimizer = optim.Adam(self.parameters(), lr=lr)
        iterator = (tqdm(range(num_epochs), desc="Epoch"))
        for e in iterator:
            optimizer.zero_grad()
            loss=self.mle()
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=[loss.item()])
        
        
        pass
    def forward(self,X_test):
        K1=self.rbf(self.X,X_test)
        Kxx=self.rbf(self.X,self.X)+torch.eye(self.N)*self.sig_n**2
        K3=self.rbf(X_test,X_test)
        L=torch.linalg.inv(Kxx)
        alpha = torch.matmul(L,self.y)  # Equivalent to (K + sigma^2 I)^{-1} y
    
        
        mu=torch.matmul(K1.T,alpha)
        Sig=K3-torch.matmul(K1.T,torch.matmul(L,K1))
        
        return mu,Sig
        
    def rbf(self,x1,x2):
        x1,x2=x1.reshape(-1,1),x2.reshape(-1,1)
        x1_exp = x1.repeat(1, x2.shape[0])  # Shape (N, M)
        x2_exp = x2.T.repeat(x1.shape[0], 1)  # Shape (N, M)
   
        # Compute squared differences
        dists_sq = (x1_exp - x2_exp) ** 2
        
        # Compute the RBF kernel
        K = self.sig**2 * torch.exp(-0.5 * dists_sq /self.l**2)
        
        return K
    
    
class sparse_GP(nn.Module):
    def __init__(self,X,y,M=10):
        super(sparse_GP, self).__init__()
        self.X=X.reshape(-1,1)
        self.y=y.reshape(-1,1)
        self.N=self.X.shape[0]
        self.M=M
        self.l=torch.nn.Parameter(torch.Tensor([0.1]))
        self.sig=torch.nn.Parameter(torch.Tensor([1]))
        self.sig_n=torch.nn.Parameter(torch.Tensor([0.1]))
        self.range=(torch.min(X),torch.max(X))
        Zs=torch.randint(0,self.N,(M,1))
        
        self.Zi=torch.nn.Parameter(self.X[Zs].clone())
        self.Z_set()
       
    
        self.m=torch.nn.Parameter(self.y[Zs].clone()).reshape((M,1))
        self.L = torch.nn.Parameter(torch.tril(torch.randn(M, M) * 0.1))


      
    def Z_set(self):
        self.Z=torch.clip(self.Zi,self.X.min(),self.X.max())
   
    def elbo(self):
        self.Z_set()
        #compute the relevant covariance matrices to map from u to f
        Kmm = self.rbf(self.Z, self.Z) + torch.eye(self.M) * 1e-5  # Stability
        Knm = self.rbf(self.X, self.Z)
        Knn = self.rbf(self.X, self.X) + torch.eye(self.N) * self.sig_n**2
        Kmn=Knm.T
        Kmm_inv = torch.inverse(Kmm)
        S=self.S()
        # Compute predictive mean and covariance of q(f) and then integrate p(y|f) to get a marginal distribution over y
        A = torch.matmul(Knm , Kmm_inv)
       
        mu_y = torch.matmul(A , self.m)
        cov_y = Knn - torch.matmul(A , Kmn) + torch.matmul(torch.matmul(A , S), A.T)
        # Compute ELBO loss
        residual = self.y - mu_y
        likelihood_expectation = -0.5 * residual.T @ torch.linalg.solve(cov_y, residual)-0.5 * torch.linalg.slogdet(cov_y)[1]-0.5 * self.N * torch.log(torch.tensor(2 * torch.pi))
        KL=0.5 * (torch.trace(torch.linalg.solve(Kmm, S)) + self.m.T @ torch.linalg.solve(Kmm, self.m) - self.M + torch.logdet(Kmm) - torch.logdet(S)) ##actually KL between p(u) and q(u) because it's easier to compute. p(u)=N(0,K_mm),q(u)=N(m,S)
       
        #likelihood_expectation pulls q(f) towards fitting the data, and KL pulls it towards the prior
        L = likelihood_expectation - KL

        return -L  # Minimize negative ELBO
    
    def S(self):
        return torch.matmul(self.L,self.L.T)
    def train(self,num_epochs=1000,lr=0.01):
        #initialise by making into parameters
   
  
        optimizer = optim.Adam(self.parameters(), lr=lr)
        iterator = (tqdm(range(num_epochs), desc="Epoch"))
        for e in iterator:
            optimizer.zero_grad()
            loss=self.elbo()
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=[np.round(loss.item(),2)])
        
        
        pass
    def forward(self,X_test):
        self.Z_set()
        Kmm = self.rbf(self.Z, self.Z) + torch.eye(self.M) * 1e-5
        Knm = self.rbf(X_test, self.Z)
        Kmn = Knm.T
        Kmm_inv = torch.inverse(Kmm)
        Knn=self.rbf(X_test, X_test)
        S=self.S()
        
        
        A = torch.matmul(Knm, Kmm_inv)
        mu_f = torch.matmul(A , self.m)
       
        cov_f = Knn - torch.matmul(A , Kmn) + torch.matmul(torch.matmul(A, S), A.T)
    
        
        return mu_f,cov_f
        
    def rbf(self,x1,x2):
        x1,x2=x1.reshape(-1,1),x2.reshape(-1,1)
        x1_exp = x1.repeat(1, x2.shape[0])  # Shape (N, M)
        x2_exp = x2.T.repeat(x1.shape[0], 1)  # Shape (N, M)
   
        # Compute squared differences
        dists_sq = (x1_exp - x2_exp) ** 2
        
        # Compute the RBF kernel
        K = self.sig**2 * torch.exp(-0.5 * dists_sq /self.l**2)
        
        return K
        