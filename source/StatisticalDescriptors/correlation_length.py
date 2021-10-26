
import torch
import numpy as np
import matplotlib.pyplot as plt

from . import common

"""
==================================================================================================================
Autocorrelation of an image
==================================================================================================================
"""

def correlation_length(X, maxlen=30):        
    if not torch.is_tensor(X):
        X = torch.tensor(X).detach()
        fg_numpy = True
    else:
        fg_numpy = False

    dim = len(X.shape)
    # if dim == 2:
    #     X = X[:maxlen,:maxlen]
    # elif dim == 3:
    #     X = X[:maxlen,:maxlen,:maxlen]
    # else:
    #     raise Exception('Dimension is not supported.')

    k = common.get_frequencies(X)
    F = torch.fft.rfftn(X, s=[n for n in X.shape], norm='ortho')
    # Y = [None]*dim
    # for i in range(dim):
        # Y[i] = torch.fft.irfftn(1j*k[i]*F*F.conj(), s=[n for n in X.shape], norm='ortho')
        # Y[i] = Y[i] / np.prod(X.shape)**(1/2)
        # # y = Y[i].mean(dim=i)
        # l = torch.sum(torch.softmax(Y[i]) * k[i,0,...])
    J = torch.real(k[0]*F*F.conj())
    Z = torch.fft.irfftn(F*F.conj(), s=[n for n in X.shape], norm='ortho')/ np.prod(X.shape)**(1/2)
    # Y = torch.fft.irfftn(1j*k[0]*F*F.conj(), s=[n for n in X.shape], norm='ortho')
    Y = torch.fft.irfftn((k[0]**2 - k[1]**2) *F*F.conj(), s=[n for n in X.shape], norm='ortho')
    Y = Y / np.prod(X.shape)**(1/2)
    # Y = Y[:maxlen,:maxlen]
    y = Y.mean(dim=1)
    l = torch.sum(torch.softmax(-y) * k[0,:,0])
    return l

