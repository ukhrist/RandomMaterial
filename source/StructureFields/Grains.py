

from math import *
import numpy as np
from scipy.optimize import fsolve
from time import time
import torch

from ..RandomField import RandomField

class GrainStructure(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # self.min_particles_number = min_particles_number
        # self.max_particles_number = max_particles_number

        self.angle = kwargs.get('angle', pi/3)
        self.shift = kwargs.get('shift', 20)
        self.mask  = kwargs.get('mask', False)
        self.scale = 1
        self.shift_layers = kwargs.get('shift_layers', np.random.randint(low=0, high=2))

        
        SizeCell = kwargs.get('SizeCell', [80,120])
        self.SizeCell = [SizeCell]*self.ndim if np.isscalar(SizeCell) else SizeCell[:self.ndim]
        self.SizeCell = np.array(self.SizeCell)

        SizeVoid = kwargs.get('SizeVoid', 0.)
        self.SizeVoid = [SizeVoid]*self.ndim if np.isscalar(SizeVoid) else SizeVoid[:self.ndim]
        self.SizeVoid = np.array(self.SizeVoid)

        assert( np.all(self.SizeCell >= self.SizeVoid) )

        self.RemoveLines = kwargs.get('RemoveLines', [False]*self.ndim)
        self.RemoveLines = np.array(self.RemoveLines)[:self.ndim]

        self.margin = np.rint( (self.SizeCell - self.SizeVoid) / 2 ).astype(np.int)
        assert( np.all(2*self.margin + self.SizeVoid == self.SizeCell) )

        self.nCells = np.rint(self.shape / self.SizeCell).astype(np.int)
        # assert( np.all(self.nCells*self.SizeCell == self.N) )



        
    #--------------------------------------------------------------------------
    #   Properties
    #--------------------------------------------------------------------------

    @property
    def semiaxes(self):
        return self.__semiaxes

    @semiaxes.setter
    def semiaxes(self, semiaxes):
        _semiaxes = np.array(semiaxes)
        if _semiaxes.size == self.ndim:
            _semiaxes = _semiaxes / np.prod(_semiaxes)**(1/self.ndim)
        elif _semiaxes.size == self.ndim-1:
            _semiaxes = np.append(_semiaxes, 1/np.prod(_semiaxes))
        else:
            _semiaxes = _semiaxes*np.ones(self.ndim)
        self.__semiaxes = _semiaxes
        assert np.isclose(np.prod(self.semiaxes), 1)
        self.isotropic = all([a==self.semiaxes[0] for a in self.semiaxes])

     
    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample(self, noise=None):
        if self.ndim == 2:
            return self.sample2d(noise)
        elif self.ndim == 3:
            return self.sample3d(noise)
        else:
            raise Exception('Dimension is not supported.')


    def sample2d(self, noise=None):
        axes = [np.arange(n) for n in self.shape]
        x = np.stack(np.meshgrid(*axes, indexing="xy"), axis=0)

        angle = (-1)**(x[1] // self.SizeCell[1]).astype(np.int) * self.angle

        x0 = x.copy()
        x0[0] = x[0] + x[1] / np.tan(angle) + self.shift

        xi = x0 % self.SizeCell[:,None,None]
        ix = self.SizeCell[:,None,None] - xi

        xi = np.minimum(xi[0], xi[1])
        ix = np.minimum(ix[0], ix[1])
        xi = np.minimum(xi, ix)

        field = self.scale * torch.tensor(xi)

        if self.mask:
            mask  = (1 + (-1)**((x[1] // self.SizeCell[1]) + self.shift_layers).astype(np.int)) * 0.5
            mask  = torch.tensor(mask)
            field = field * mask     

        return field


    def sample3d(self, noise=None):
        axes = [np.arange(n) for n in self.shape]
        x = np.stack(np.meshgrid(*axes, indexing="xy"), axis=0)

        angle = (-1)**(x[1] // self.SizeCell[1]).astype(np.int) * self.angle
        mask  = (1 + (-1)**((x[1] // self.SizeCell[1]) + self.shift_layers).astype(np.int)) * 0.5
        mask  = torch.tensor(mask)

        x0 = x.copy()
        x0[0] = x[0] + x[1] / np.tan(angle) + self.shift

        xi = x0 % self.SizeCell[:,None,None]
        ix = self.SizeCell[:,None,None] - xi

        xi = np.minimum(xi[0], xi[1])
        ix = np.minimum(ix[0], ix[1])
        xi = np.minimum(xi, ix)

        field = self.scale * torch.tensor(xi) * mask        
        return field

    