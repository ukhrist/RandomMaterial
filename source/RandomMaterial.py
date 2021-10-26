
import torch
from torch import nn
from torch.backends.mkldnn import flags
from torch.nn.modules import activation
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np

from .GaussianRandomField import GaussianRandomField
from .TwoPhaseMaterial import TwoPhaseMaterial
from .StatisticalDescriptors import interface



#######################################################################################################
#	Random material generator class
#######################################################################################################

class RandomMaterial(TwoPhaseMaterial):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.nPhases = kwargs.get('nPhases', 2)
        GRF = GaussianRandomField(**kwargs)
        self.phases  = [ GRF for iPhase in range(self.nPhases) ]
        self.structure  = [ torch.tensor(0.) for iPhase in range(self.nPhases) ]

        actfc = kwargs.get('actfc', 'argmax') ### 'argmax', 'heaviside'
        if actfc in ('argmax',):
            # self.actfc = lambda x: torch.argmax(x)
            self.actfc = lambda x: ( torch.softmax(x * self.shape.max(), dim=-1) * torch.arange(x.shape[-1]) ).sum(dim=-1)
        elif actfc in ('heaviside',):
            # self.actfc = lambda x: torch.heaviside(x, torch.tensor(0.))
            self.actfc = lambda x: torch.sigmoid(x * self.shape.max()).sum(dim=-1)

        self.alpha = kwargs.get('alpha', 1.)
        m, n = self.nPhases, self.nPhases-1
        self.PhaseCorrelation = nn.Linear(m, n, bias=False)
        self.PhaseCorrelation.weight.data = torch.tensor([[1.,0.,0.],[0.,1.-self.alpha,self.alpha]])
        # self.PhaseCorrelation.weight.data = torch.eye(self.nPhases)
        # self.PhaseCorrelation.weight.data = torch.eye(self.nPhases) - torch.eye(self.nPhases).roll(-1,1)
        # self.PhaseCorrelation.weight.data = torch.eye(self.nPhases-1,self.nPhases) # - torch.eye(self.nPhases-1,self.nPhases).roll(1,1)
        # self.PhaseCorrelation.weight.data = 2*torch.eye(self.nPhases) - torch.triu(torch.ones(self.nPhases,self.nPhases))
        # self.PhaseCorrelation.weight.data = 1/2*torch.eye(self.nPhases)*0 - torch.ones(self.nPhases,self.nPhases)
        # self.PhaseCorrelation.weight.data = 2*torch.eye(self.nPhases) - torch.eye(1,self.nPhases).matmul(torch.ones(self.nPhases,1))
        # self.PhaseCorrelation.weight.data = torch.tensor([[-1.,1.,-1.],[1.,-1.,1.],[-1.,-1.,-1.]])
        # self.PhaseCorrelation.weight.data = torch.tensor([[1.,-1.],[-1.,1.]])

        # self.PhaseCorrelation.bias.data = 0*torch.tensor([-1.,1.])
        # self.PhaseCorrelation.bias.data   = torch.zeros()
        # self.PhaseCorrelation.bias.data = torch.ones(self.nPhases)

    
    # def actfc(self, x):
    #     return torch.heaviside(x, torch.tensor(0.))
        

        


    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------
    
    ### Generate a realization

    def sample(self, noise=None):
        IntensityVector = torch.stack([ phase.sample(noise) for phase in self.phases ], dim=-1)
        IntensityVectorStructure = torch.stack([ phase.sample(noise) for phase in self.structure ], dim=-1)
        IntensityVector = 0.5*IntensityVector + IntensityVectorStructure
        # IntensityVector = self.PhaseCorrelation(IntensityVector)
        PhaseField      = self.actfc(IntensityVector)
        # PhaseField      = IntensityVector.sum(dim=-1)
        # PhaseField_grad = interface(PhaseField).abs().detach().unsqueeze(dim=-1) * IntensityVector
        # PhaseVector     = PhaseField_grad + (PhaseField - PhaseField_grad).detach()
        # PhaseField      = PhaseVector.sum(dim=-1)
        # PhaseVector_grad= interface(PhaseVector, vector=True).abs().detach() * IntensityVector
        # PhaseVector     = PhaseVector_grad + (PhaseVector - PhaseVector_grad).detach()
        # PhaseField      = PhaseVector.sum(dim=-1)
        return PhaseField

#######################################################################################################


        
