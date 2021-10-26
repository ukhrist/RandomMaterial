

from math import pi
import torch
import numpy as np
from scipy.optimize import fsolve
from collections.abc import Iterable, Callable
from time import time

from ..RandomField import RandomField

class MatrixStructure(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.tau   = kwargs.get('tau', 0.)
        self.field = torch.ones(*self.Window.shape) * self.tau

    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample(self, noise=None):     
        return self.field






