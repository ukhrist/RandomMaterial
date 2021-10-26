
import torch
import numpy as np
import matplotlib.pyplot as plt

from .common import get_frequencies


"""
==================================================================================================================
Autocorrelation of an image
==================================================================================================================
"""

def autocorrelation(X, spectrum_only=False, deriv=0):
    if not torch.is_tensor(X):
        X = torch.tensor(X).detach()
        fg_numpy = True
    else:
        fg_numpy = False

    FX = torch.fft.rfftn(X, s=[n for n in X.shape], norm='ortho')
    FC = FX.abs()**2 # F*F.conj()

    if type(deriv) is int and deriv>0:
        k = get_frequencies(X).norm(dim=0)
        FC = (1j*k)**deriv * FC

    if spectrum_only:
        C = FC
    else:
        C = torch.fft.irfftn(FC, s=[n for n in X.shape], norm='ortho')
        C = C / np.prod(X.shape)**(1/2)

    return C.numpy() if fg_numpy else C

