
import torch
from torch import nn
from torch.backends.mkldnn import flags
from torch.nn.modules import activation
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np

from .MultiPhaseMaterial import MultiPhaseMaterial
from ..GaussianRandomField import GaussianRandomField
from ..StructureFields.SupportedMaterials import GrainMaterial



#######################################################################################################
#	Random material with support
#######################################################################################################

class MultiGrainMaterial(MultiPhaseMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure  = None
        self.par_alpha  = nn.Parameter(torch.tensor(0.))
        # self.par_tau    = nn.Parameter(torch.tensor(0.))
        self.par_tau    = torch.tensor(0.)
        self.alpha      = kwargs.get('alpha', 1)
        self.tau        = kwargs.get('tau', 0)


    #--------------------------------------------------------------------------
    #   Parameters
    #--------------------------------------------------------------------------

    @property
    def alpha(self):
        a = torch.exp(self.par_alpha)
        return a/(1+a)

    @alpha.setter
    def alpha(self, alpha):
        self.par_alpha.data = torch.log(torch.tensor(float( alpha/(1-alpha) )))

    @property
    def tau(self):
        return torch.exp(self.par_tau)

    @tau.setter
    def tau(self, tau):
        self.par_tau.data = torch.log(torch.tensor(float( tau )))
        
        

    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample_intensity(self, noise=None):
        IntensityPerturbation = self.sample_GRF(noise)
        IntensityStructure    = self.Structure.sample()
        # IntensityField        = (1-self.alpha) * IntensityStructure + self.alpha * IntensityPerturbation - self.tau
        IntensityField        = IntensityStructure + self.alpha * IntensityPerturbation - self.tau
        return IntensityField




#######################################################################################################
#	Lattice material struture
#######################################################################################################

class LatticeMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = LatticeStructure(**kwargs)


#######################################################################################################
#	Lattice material struture
#######################################################################################################

class ParticlesMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = ParticlesCollection(**kwargs)



#######################################################################################################
#	Grains material struture
#######################################################################################################

class GrainMaterial(SupportedMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Structure = GrainStructure(**kwargs)

#######################################################################################################