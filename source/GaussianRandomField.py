
import numpy as np
import torch

from .RandomField import RandomField
from .Kernels import MaternKernel
from . import Correlators

"""
==================================================================================================================
Gaussian Random Field generator class
==================================================================================================================
"""

class GaussianRandomField(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_folded = kwargs.get('Folded_GRF', False)

        ### Empty covariance
        self.Covariance = kwargs.get('GRF_Covariance', MaternKernel)
        self.Covariance = self.Covariance(**kwargs)

        ### Sampling method
        CorrelatorName = kwargs.get('sampling_method', 'torch')
        self.set_Correlator(CorrelatorName)

    #--------------------------------------------------------------------------
    #   Initialize sampling method
    #--------------------------------------------------------------------------

    def set_Correlator(self, CorrelatorName, **kwargs):
        self.correlate = Correlators.set_Correlator(self, CorrelatorName)
        if self.verbose:
            print('Sampling method is set to {0:s}.'.format(CorrelatorName))


    #--------------------------------------------------------------------------
    #  Update covariance (update learnable parameters)
    #--------------------------------------------------------------------------

    def update(self):
        self.correlate.initialize()


    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    ### Sample noise
    def sample_noise(self):
        noise = self.PRNG.normal(loc=0, scale=1, size=self.Window.shape)
        noise = torch.tensor(noise).detach()
        return noise

    ### Sample GRF
    def sample_GRF(self, noise=None):
        if noise is None: noise = self.sample_noise()
        Sample = self.correlate(noise)
        Sample = self.Window.crop_to_domain(Sample)
        if self.is_folded: Sample = Sample.abs()
        return Sample

    ### Sample
    def sample(self, noise=None):
        return self.sample_GRF(noise)

    #--------------------------------------------------------------------------
    #   Two-point correlation function
    #--------------------------------------------------------------------------
    
    def S2(self, r):
        if not self.is_folded:
            return self.Covariance.eval_func(r)
        else:
            raise NotImplementedError()



###################################################################


