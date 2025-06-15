
import torch
import numpy as np
# from numpy.core.numeric import Inf
# import matplotlib.pyplot as plt
from math import pi

from . import common
from . interface import interface

"""
==================================================================================================================
Compute numerical curvatures (Gaussian and Mean)
==================================================================================================================
"""

def num_curvature(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).detach()
        fg_numpy = True
    else:
        fg_numpy = False


    I = interface(X)

    d = torch.zeros([3]).detach()
    dir_vec = torch.zeros([6, X.dim()]).detach()
    dir_ind = torch.zeros([6, *X.shape])
    for i in range(2):
        for j in range(3):
            d[:] = 0.
            d[j] = (-1)**i
            dir_vec[3*i+j] = d
            dir_ind[3*i+j] = I.roll((-1)**i, dims=j)  
    
    dir_vec = dir_vec.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    # dir_ind = dir_ind.clone() * I
    dir_ind = dir_ind.clone().unsqueeze(dim=1)

    nNeighbours = dir_ind.sum(dim=0).squeeze()

    I[nNeighbours==0.] = 0
    I[nNeighbours==1.] = 0
    I[nNeighbours==2.] = 0

    dir_ind = dir_ind.clone() * I
    nNeighbours = dir_ind.sum(dim=0).squeeze()

    Hdir = (dir_ind*dir_vec).sum(dim=0)
    A = 1/4 * nNeighbours #* np.prod(X.shape)**(-2/3)
    A = torch.where(nNeighbours !=0, 1/4 * nNeighbours, 1.)
    H = Hdir.norm(dim=0) / A
    K = (2*pi - pi/2 * nNeighbours) #/ A


    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    H = H*I
    K = K*I

    M3 = I[nNeighbours==3].sum()
    M4 = I[nNeighbours==4].sum()
    M5 = I[nNeighbours==5].sum()
    M6 = I[nNeighbours==6].sum()

    g = 1 + (M5 + 2*M6 - M3)/8

    return H, K, g



"""
==================================================================================================================
Compute Gaussian curvature of the image
==================================================================================================================
"""

def curvature(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).detach()
        fg_numpy = True
    else:
        fg_numpy = False

    # shape = X.shape
    # ndim  = X.dim()
    # G = torch.zeros((ndim,) + shape)
    # for j in range(ndim):
    #     G[j] = X - X.roll(-1, dims=j)
    # normG = G.norm(dim=0)
    # # G = torch.where(normG > 0, G/normG, G)

    # H = torch.zeros((ndim,ndim,) + shape)
    # for i in range(ndim):
    #     for j in range(ndim):
    #         H[i,j] = G[i] - G[i].roll(-1, dims=j)

    # T = torch.zeros_like(G)
    # T[0] = -G[1]
    # T[1] =  G[0]

    # HT = (H*T).sum(dim=1)
    # K  = (T*HT).sum(dim=0)




    # F = torch.fft.rfftn(X, s=[n for n in X.shape], norm='ortho')
    # k = common.get_frequencies(X)
    # scale = 1/np.prod(X.shape)**(1/2)
    # G1  = torch.fft.irfftn(1j*k[0] * F, s=[n for n in X.shape], norm='ortho') * scale
    # G2  = torch.fft.irfftn(1j*k[1] * F, s=[n for n in X.shape], norm='ortho') * scale
    # G   = (G1**2 + G2**2)**(1/2)
    # G1, G2 = G1/G, G2/G
    # T1  =  1*G2
    # T2  = -1*G1
    # FG1 = torch.fft.rfftn(G1, s=[n for n in X.shape], norm='ortho')
    # FG2 = torch.fft.rfftn(G2, s=[n for n in X.shape], norm='ortho')
    # G11 = torch.fft.irfftn(1j*k[0] * FG1, s=[n for n in X.shape], norm='ortho') * scale
    # G12 = torch.fft.irfftn(1j*k[1] * FG1, s=[n for n in X.shape], norm='ortho') * scale
    # G21 = torch.fft.irfftn(1j*k[0] * FG2, s=[n for n in X.shape], norm='ortho') * scale
    # G22 = torch.fft.irfftn(1j*k[1] * FG2, s=[n for n in X.shape], norm='ortho') * scale

    # K = G11 * T1**2 + (G12+G21) * T1*T2 + G22 * T2**2



    fg_use_fft = False
    fg_diff    = True
    G = -common.gradient(X, fft=fg_use_fft, diff=fg_diff, normalized=False, scheme='central') #'backward')
    # normG = G.norm(dim=0)
    # interface = torch.where(normG>0, True, False).detach()
    # G = G.clone()
    # G[:,interface] = G[:,interface].clone()/normG[interface]

    T1 = torch.zeros_like(G)
    T2 = torch.zeros_like(G)
    T1[0], T1[1], T1[2] =  1*G[1], -1*G[0], 0*G[2]
    T2[0], T2[1], T2[2] =  1*G[0]*G[2],  1*G[1]*G[2], -G[0]**2-G[1]**2
    T1 = common.normalize(T1)
    T2 = common.normalize(T2)
    # normT1 = T1.norm(dim=0)
    # normT2 = T2.norm(dim=0)
    # interface = torch.where(normT1>0, True, False).detach()
    # # T1 = torch.where(normT1.detach()>0, T1/normT1, 0.)
    # T1, T2 = T1.clone(), T2.clone()
    # T1[:,interface] = T1[:,interface].clone()/normT1[interface]
    # interface = torch.where(normT2>0, True, False).detach()
    # # T2 = torch.where(normT2.detach()>0, T2/normT2, 0.)
    # T2[:,interface] = T2[:,interface].clone()/normT2[interface]


    H = torch.stack([common.gradient(G[j], fft=fg_use_fft, diff=False, scheme='central') for j in range(X.dim())], dim=0)
    # interface = torch.where(normG>0, True, False).detach()
    # H[:,:,interface] = H[:,:,interface].clone()/normG[interface]

    HT1 = (H*T1).sum(dim=1)
    M11 = (T1*HT1).sum(dim=0)

    HT2 = (H*T2).sum(dim=1)
    M22 = (T2*HT2).sum(dim=0)

    M12  = (T1*HT2).sum(dim=0)
    M21  = (T2*HT1).sum(dim=0)

    # D = ((M11-M22)/2)**2 + M12*M21
    # # if torch.any(D<0):
    # #     print('D<0')
    # #     exit()
    # D = torch.relu(D)

    # K1 = (M11+M22)/2 + torch.sqrt(D)
    # K2 = (M11+M22)/2 - torch.sqrt(D)

    K1 = (M11+M22)/2
    K2 = M11*M22 - M12*M21

    # ### Gauss curvature
    # K = K1*K2
    # yield K.numpy() if fg_numpy else K

    # ### Mean curvature
    # K = (K1+K2)/2
    # yield K.numpy() if fg_numpy else K

    return K1, K2



"""
==================================================================================================================
Compute specific  curvature: ratio of the Gaussian curvature to the volume of a phase
==================================================================================================================
"""

def spec_curvature(X):
    K   = curvature(X)
    vol = X.sum()
    k   = K.sum() / vol if vol else torch.tensor(0.)
    return k

    




