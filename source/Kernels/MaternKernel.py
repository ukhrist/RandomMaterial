
import numpy as np
from math import lgamma, pi, sqrt

from scipy.special import kv as Kv
from scipy.special import gamma
# from scipy.special import hyp2f1

import torch
from torch import nn

from .AbstractKernel import Kernel, set_Metric, rotate

import matplotlib.pyplot as plt


"""
==================================================================================================================
Matern kernel class
==================================================================================================================
"""

class MaternKernel(Kernel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        nu      = kwargs.get('nu', 1.0)
        corrlen = kwargs.get('correlation_length', 0.1)
        angle   = kwargs.get('anisotropy_angle', 0)
        axle    = kwargs.get('anisotropy_axle',  None)

        if np.isscalar(corrlen):
            corrlen = [corrlen]
            self.isotropic = True
        else:
            assert(len(corrlen) == self.ndim)
            self.isotropic = False

        # corrlen, axle = set_Metric(ndim=self.ndim, length=corrlen, angle=angle, axle=axle, out='vectors')

        ### Learnable parameters
        self.log_nu      = nn.Parameter(torch.log(torch.tensor([nu], dtype=float)))
        self.log_corrlen = nn.Parameter(torch.log(torch.tensor(corrlen, dtype=float)))
        # self.log_corrlen = nn.Parameter(torch.tensor(corrlen, dtype=float).sqrt())
        # self.axle    = nn.Parameter(torch.tensor(axle.astype(np.float)))
        # self.Theta = nn.Linear(self.ndim, self.ndim, bias=False).double()
        # self.Theta.weight.data = torch.tensor(Theta)

    @property
    def nu(self):
        return torch.exp(self.log_nu)

    @nu.setter
    def nu(self, nu):
        self.log_nu.data = torch.log(torch.tensor(float(nu)))

    @property
    def corrlen(self):
        return torch.exp(self.log_corrlen) * torch.ones([self.ndim])
        # return self.log_corrlen.square() * torch.ones([self.ndim])


    @corrlen.setter
    def corrlen(self, corrlen):
        # corrlen = [corrlen]*self.ndim if np.isscalar(corrlen) else corrlen[:self.ndim]
        # corrlen = torch.tensor(corrlen)
        # self.log_corrlen.data = torch.log(corrlen)
        self.log_corrlen.data[:] = torch.tensor(corrlen).log()
        # self.log_corrlen.data[:] = torch.tensor(corrlen).double().sqrt()


    def eval_func(self, x, nu=None, rho=None): ### no torch
        if not nu: nu = self.nu.item()
        # if not rho: rho = np.sqrt(np.diag(self.Theta.detach().numpy())).mean()
        if not torch.is_tensor(x): x = torch.tensor(x)
        kappa = torch.sqrt(2*self.nu) / self.corrlen.mean()
        r = (kappa*x.abs()).detach().numpy()
        return Matern_function(nu, r)


    def eval_spec(self, freq):
        if not torch.is_tensor(freq): freq = torch.tensor(freq)

        nu, d = self.nu, self.ndim
        alpha = nu + d/2
        kappa = self.corrlen / torch.sqrt(2*self.nu)
        scale = self.set_scale() * kappa.prod().abs()


        Tw   = kappa*rotate(freq)
        wTTw = Tw.square().sum(dim=-1)

        # if self.nu.item() < 1000:
        self.Spectrum = scale * (1 + wTTw)**(-alpha)
        # else:# Squared-Exponential
        #     Spectrum = scale * np.exp(-1/4 * wTw)

        return self.Spectrum


    def set_scale(self): 
        nu, d = self.nu, self.ndim
        alpha = nu + d/2
        gamma_nu    = torch.exp(torch.lgamma(nu))
        gamma_alpha = torch.exp(torch.lgamma(alpha))
        # detTheta    = self.corrlen.prod()**2
        # scale       = gamma_nu/gamma_alpha * (2*nu / (4*pi))**(d/2) / torch.sqrt(detTheta)
        # scale       = torch.sqrt( 1 / scale )
        # scale       = (4*pi)**(d/2) * torch.sqrt(detTheta/(2*nu)**d) * gamma_alpha/gamma_nu
        # scale       = (4*pi)**(d/2) * gamma_alpha/gamma_nu
        scale       = (4*pi)**(d/2) * (alpha.lgamma()-nu.lgamma()).exp()
        return scale


"""
==================================================================================================================
Matern function
==================================================================================================================
"""

def Matern_function(nu, x):
    assert(nu > 0)
    sigma = 1 / (2**(nu-1) * gamma(nu))
    y = sigma * (x**nu) * Kv(nu, x) if x>0 else 1
    return y

Matern_function = np.vectorize(Matern_function, otypes=[float], excluded=['nu'])


#########################################################