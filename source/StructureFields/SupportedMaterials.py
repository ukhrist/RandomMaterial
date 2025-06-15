
import torch
from torch import nn
from torch.backends.mkldnn import flags
from torch.nn.modules import activation
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from math import pi

from ..GaussianRandomField import GaussianRandomField
from ..TwoPhaseMaterial import TwoPhaseMaterial
from .OctetLattice import OctetLatticeStructure as LatticeStructure
from .Grains import GrainStructure
from .GrainClusters import GrainClustersStructure
from .Particles import ParticlesCollection
from .Fractures import FracturesCollection
from .Voronoi import VoronoiCollection
from .Cracks import CracksCollection
# from .Cracks_gauss import CracksCollection_gauss
from .Gyroid import Gyroid as GyroidStructure



#######################################################################################################
#	Random material with support
#######################################################################################################

class SupportedMaterial(TwoPhaseMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure  = None
        self.par_alpha  = nn.Parameter(torch.tensor([0.]))
        self.par_tau    = nn.Parameter(torch.tensor([0.]))
        # self.par_tau    = torch.tensor([0.])
        self.alpha      = kwargs.get('alpha', 0)
        self.tau        = kwargs.get('tau', 0)


    #--------------------------------------------------------------------------
    #   Parameters
    #--------------------------------------------------------------------------

    @property
    def alpha(self):
        # return torch.exp(self.par_alpha)
        # return self.par_alpha
        return 0.5 + 0.5*torch.tanh(self.par_alpha)
        # x2 = self.par_alpha.square()
        # return x2/(1+x2)

    @alpha.setter
    def alpha(self, alpha):
        # self.par_alpha.data[:] = torch.tensor(alpha, dtype=float).log()
        # self.par_alpha.data[:] = alpha
        self.par_alpha.data[:] = torch.tensor(2*alpha-1, dtype=float).atanh()
        # self.par_alpha.data[:] = torch.tensor(alpha/(1-alpha), dtype=float).sqrt()

    @property
    def tau(self):
        return torch.exp(self.par_tau)

    @tau.setter
    def tau(self, tau):
        self.par_tau.data[:] = torch.tensor(tau, dtype=float).log()
        
        

    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample_intensity(self, noise=None):
        IntensityPerturbation = self.sample_GRF(noise)
        IntensityStructure    = self.Structure.sample()
        # IntensityField        = torch.cos(pi/2*self.alpha) * IntensityStructure + torch.sin(pi/2*self.alpha) * IntensityPerturbation - 0*self.tau
        IntensityField        = (1-self.alpha) * IntensityStructure + self.alpha * IntensityPerturbation #- 0*self.tau
        return IntensityField




#######################################################################################################
#	Lattice material struture
#######################################################################################################

class LatticeMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = LatticeStructure(**kwargs)
        self.noise_quantile = nn.Parameter(torch.tensor([0.], dtype=float))        
        self.noise_sparsity = kwargs.get('noise_sparsity', 0)
        # self.noise_quantile.data[:] = kwargs.get('noise_quantile', 0)
        
    @property
    def noise_sparsity(self):
        # p = 0.5 * (1 + torch.erf(self.noise_quantile / 2**0.5))
        # return p
        return self.noise_quantile.item()

    @noise_sparsity.setter
    def noise_sparsity(self, noise_sparsity):
        # p = torch.tensor(noise_sparsity)
        # q = 2**0.5 * torch.erfinv(2*p-1)
        self.noise_quantile.data[:] = noise_sparsity

    # def sample_GRF(self, noise=None):
    #     if noise is None: noise = self.sample_noise()
    #     q = torch.exp(self.noise_quantile)
    #     # noise_sparse = noise * (noise.abs()-q).sigmoid()
    #     noise_sparse = noise * self.actfc_smooth(noise.abs()-q)
    #     # std = noise_sparse.square().mean().sqrt()
    #     # noise_sparse_normalized = noise_sparse / std
    #     return super().sample_GRF(noise_sparse)



#######################################################################################################
#	Particle material struture
#######################################################################################################

class ParticlesMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = ParticlesCollection(**kwargs)



#######################################################################################################
#	Fracture material struture
#######################################################################################################

class FracturesMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = FracturesCollection(**kwargs)


#######################################################################################################
#	Voronoi material struture
#######################################################################################################

class VoronoiMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = VoronoiCollection(**kwargs)



#######################################################################################################
#	Cracks material struture
#######################################################################################################

class CracksMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = CracksCollection(**kwargs)

# class CracksMaterial_gauss(SupportedMaterial):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.Structure = CracksCollection_gauss(**kwargs)



#######################################################################################################
#	Grains material struture
#######################################################################################################

class GrainMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = GrainStructure(**kwargs)


class GrainClustersMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = GrainClustersStructure(**kwargs)



#######################################################################################################
#	Gyroid struture
#######################################################################################################

class GyroidMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = GyroidStructure(**kwargs)

#######################################################################################################