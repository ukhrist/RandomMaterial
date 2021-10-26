
import torch
import numpy as np
import matplotlib.pyplot as plt

from . import common

"""
==================================================================================================================
Autocorrelation of an image
==================================================================================================================
"""

def correlation_curvature(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).detach()
        fg_numpy = True
    else:
        fg_numpy = False
    k = common.get_frequencies(X)
    k2= k.square().sum(dim=0)
    F = torch.fft.rfftn(X, s=[n for n in X.shape], norm='ortho')
    Y = torch.fft.irfftn(k2*F*F.conj(), s=[n for n in X.shape], norm='ortho')
    # Y = torch.flatten(Y)[0]
    Y = Y / np.prod(X.shape)**(1/2)
    return Y.numpy() if fg_numpy else Y

