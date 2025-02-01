import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

def binomial_tree(mu,sigma,S0, N, T, step):

    u=np.exp(sigma*np.sqrt(step))
    d= 1.0/u
    p= 0.5+0.5*(mu/sigma)*np.sqrt(step)

    up_times=np.zeros((N,len(T)))
    down_times=np.zeros((N,len(T)))
    for idx in range(len(T)):
        up_times[:,idx]=np.random.binomial(T[idx]/step,p,N)
        down_times[:,idx]=T[idx]/step - up_times[:,idx]
