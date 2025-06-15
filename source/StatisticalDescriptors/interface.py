
from numpy.core.fromnumeric import shape
# from numpy.lib.function_base import diff
# from numpy import diff
import torch
import numpy as np
# from numpy.core.numeric import Inf
# import matplotlib.pyplot as plt

from .common import gradient



"""
******************************************************************************************************************
NOTE: So far, implemented only for the two-phase material !!
******************************************************************************************************************
"""



"""
==================================================================================================================
Get the interface of the phases
==================================================================================================================
"""

def interface(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).detach()
        fg_numpy = True
    else:
        fg_numpy = False

    G = gradient(X, diff=True) #, scheme='central')
    I = 1*G.norm(dim=0, p=np.inf)
    # I = 1*G.square().sum(dim=0) #**2 #p=np.inf)

    # dim = X.dim()
    # I = torch.zeros([dim**dim, *X.shape])

    # d = [-1, 0, 1]
    # for i in range(dim):
    #     for j in range(dim):
    #         for k in range(dim):
    #             I[i*dim**2 + j*dim + k] = X.roll(shifts=[d[i],d[j],d[k]], dims=[0,1,2]) - X #X.roll(shifts=[-d[i],-d[j],-d[k]], dims=[0,1,2])
    # I = I.norm(dim=0, p=np.inf)
    
    # V = torch.zeros([6, *X.shape])
    # for i in range(2):
    #     for j in range(3):
    #         d = [0]*3
    #         d[j] = (-1)**i
    #         V[3*i+j] = X.roll(shifts=d, dims=[0,1,2])
    # V = V.sum(dim=0)

    # I = I*X*(V>=3)

    return I.numpy() if fg_numpy else I



"""
==================================================================================================================
Compute the specific area: ration of the phases interface area to the volume (one of the phases)
==================================================================================================================
"""

def spec_area(X):
    # Y   = interface(X)
    # s   = Y.mean()
    # # vol = X.mean()
    # # s   = s / vol if vol else s

    # h = 1/torch.tensor(X.shape, dtype=float).norm().detach()

    # G = gradient(X, diff=True) #normalize=True)
    # Y = G.norm(dim=0)
    # s = Y.mean() #* h

    I = interface(X)
    s = I.mean()
    return s

    




