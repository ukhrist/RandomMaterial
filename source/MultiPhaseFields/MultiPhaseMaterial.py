
from math import pi
import torch
from torch import nn
from torch.backends.mkldnn import flags
from torch.nn.modules import activation
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np

from ..RandomField import RandomField
from ..GaussianRandomField import GaussianRandomField
from ..StructureFields.SupportedMaterials import GrainMaterial



#######################################################################################################
#	Multi-phase material class (Abstract)
#######################################################################################################

class MultiPhaseMaterial(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nPhases = kwargs.get('nPhases', 2)
        

    def actfc(self, x):
        return torch.argmax(x, dim=-1)

    def actfc_smooth(self, x):         
        return ( torch.softmax(x * 1.e2, dim=-1) * torch.arange(x.shape[-1]) ).sum(dim=-1)


    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------
    
    def sample_intensities(self, noise=None):
        return None

    def sample(self, noise=None, phases=None): ### phases=None - all phases
        IntensityVector = self.sample_intensities(noise)
        if phases is not None:
            PhaseField = IntensityVector[..., phases].detach().numpy()
        else:
            PhaseField = self.actfc(IntensityVector)
        return PhaseField


        
#######################################################################################################
#	Multi-grain material
#######################################################################################################

class MultiGrainMaterial(MultiPhaseMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        a = kwargs.pop('angle', pi/3)
        self.Phases = [GrainMaterial(**kwargs, angle=np.random.normal(a, 0.1*a), shift=np.random.uniform(0,100), shift_layers=i ) for i in range(self.nPhases)]
        config = kwargs.copy()
        config['nu'] = 5
        config['correlation_length'] = 0.1
        config['Folded'] = True
        self.aux_field = GaussianRandomField(**config) 

    def sample_intensities(self, noise=None):
        IntensityVector = torch.stack([ 0*0.1*self.aux_field.sample() + Phase.sample_intensity() for Phase in self.Phases ], dim=-1)
        return IntensityVector



#######################################################################################################