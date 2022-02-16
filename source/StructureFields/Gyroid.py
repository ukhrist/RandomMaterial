

from math import pi, sqrt
import numpy as np
from scipy.optimize import fsolve
from time import time

import torch
from torch import nn
# import pytorch3d
from scipy.spatial.transform import Rotation

from ..RandomField import RandomField

class Gyroid(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.par_thickness = nn.Parameter(torch.tensor([0.]))
        self.thickness     = kwargs.get('thickness', 0.1)

        axes = [2*pi*torch.arange(n)/n for n in self.Window.shape]
        self.coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), axis=-1).detach()

        self.fg_periodic = kwargs.get('periodic', True)

        
    #--------------------------------------------------------------------------
    #   Properties
    #--------------------------------------------------------------------------

    @property
    def thickness(self):
        return self.par_thickness.square()

    @thickness.setter
    def thickness(self, thickness):
        self.par_thickness.data[:] = torch.tensor(float(thickness)).sqrt()
    
    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample(self):

        x = self.coordinates[...,0]
        y = self.coordinates[...,1]
        z = self.coordinates[...,2]

        field = torch.sin(x) * torch.cos(y) + torch.sin(y) * torch.cos(z) + torch.sin(z) * torch.cos(x)
        field = self.thickness - field**2

        return field



    