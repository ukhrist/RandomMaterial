
import torch
from torch import nn
import numpy as np#
from math import sqrt
from time import time

from .GaussianRandomField import GaussianRandomField
from .StatisticalDescriptors import interface, gradient


"""
==================================================================================================================
Random two phase material (Abstract)
==================================================================================================================
"""

class TwoPhaseMaterial(GaussianRandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration_regime = kwargs.get('calibration_regime', False)

    def actfc_binary(self, x):
        return torch.heaviside(x, torch.tensor(0.))

    def actfc_smooth(self, x):
        h = 1/self.Window.domain_shape.max() #* 2
        eps = h #h**2
        return 0.5*(torch.tanh(x / eps) + 1.)

    @property
    def calibration_regime(self):
        return self.__calibration_regime

    @calibration_regime.setter
    def calibration_regime(self, regime_on):
        self.__calibration_regime = regime_on
        if regime_on:
            self.actfc = self.actfc_smooth
        else:
            self.actfc = self.actfc_binary


    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample_intensity(self, noise=None):
        ### virtual
        return None

    def sample(self, noise=None):
        IntensityField = self.sample_intensity(noise)
        PhaseField     = self.actfc(IntensityField)
        # PhaseField        = self.actfc_smooth(IntensityField)
        # PhaseField_smooth = self.actfc_smooth(IntensityField)
        # PhaseField_smooth = interface(PhaseField).detach() * IntensityField / gradient(IntensityField, fft=False).norm(dim=0).detach()
        # PhaseField        = PhaseField_smooth + (PhaseField - PhaseField_smooth).detach()
        return PhaseField




"""
==================================================================================================================
Gaussian level-set material (Simple random two phase material)
==================================================================================================================
"""

class GaussianMaterial(TwoPhaseMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = nn.Parameter(torch.tensor([0.]))
        self.vf  = kwargs['vf']


    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample_intensity(self, noise=None):
        IntensityField = self.sample_GRF(noise) - self.tau
        return IntensityField


    #--------------------------------------------------------------------------
    #   Statistics
    #--------------------------------------------------------------------------

    ### Volume fraction
    @property
    def vf(self):
        return tau2vf(self.tau.detach().numpy(), strategy=self.is_folded)

    @vf.setter
    def vf(self, vf):
        self.tau.data[:] = torch.tensor( vf2tau(vf, strategy=self.is_folded) )
    
    ### Two-point correlation function
    def S2(self, r):
        C = self.Covariance.eval_func(r)
        return Cov2S2(C, vf=self.vf, strategy=self.is_folded)

#######################################################################################################




      
"""
==================================================================================================================
GAUSSIAN LEVEL-CUT
==================================================================================================================
"""

from math import sqrt
from scipy.special import erf, erfinv, erfc, owens_t


#######################################################################################################

def levelcut(field, level=0):
    phase = np.where(field > level, 1, 0)
    return phase.astype(np.intc)

def get_vf(phase):
    return np.mean(phase)

#######################################################################################################
#	Volume fraction to Tau (and vice-versa)
#######################################################################################################

def vf2tau(vf, sigma=1, strategy=0):
    if strategy:
        # |u|<tau
        return sqrt(2)*sigma*erfinv(1-vf)
    else:
        # u<tau
        return sqrt(2)*sigma*erfinv(1-2*vf)

def tau2vf(tau, sigma=1, strategy=0):
    if strategy:
        # |u|<tau
        return 1-erf(np.abs(tau)/sqrt(2)/sigma)
    else:
        # u<tau
        return 0.5*(1-erf(tau/sqrt(2)/sigma))

def Cov2S2(g, vf=None, tau=None, strategy=0):
    assert( all(g>=0) and all(g<=1) )
    if vf is not None:
        tau = vf2tau(vf, strategy=strategy)
    elif tau is not None:
        vf = tau2vf(tau, strategy=strategy)
    else:
        print("Either vf or tau has to be given.")
        exit()
    x = np.sqrt((1-g)/(1+g))
    if strategy:
        # |u|<tau
        x1 = np.where(np.abs(x)<1.e-10, 1.e-10, x)
        S2 = 2*vf - 4*owens_t(tau, x) - 4*owens_t(tau, 1/x1)             
    else:
        # u<tau
        S2 = vf - 2 * owens_t(tau, x)
    return S2