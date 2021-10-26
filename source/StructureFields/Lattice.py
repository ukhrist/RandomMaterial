

from math import inf, pi
from matplotlib.pyplot import axes
from numpy.core.numeric import Inf
import torch
import numpy as np
from scipy.optimize import fsolve
from collections.abc import Iterable, Callable
from time import time

from torch.functional import norm

from ..RandomField import RandomField

class LatticeStructure(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # self.SizeCell = kwargs.get('SizeCell', self.Window.shape / 4)
        # if np.isscalar(self.SizeCell):
        #     self.SizeCell = [self.SizeCell]*self.ndim
        # self.SizeCell = np.array(self.SizeCell)[:self.ndim]

        # self.SizeVoid = kwargs.get('SizeVoid', 0.)
        # if np.isscalar(self.SizeVoid):
        #     self.SizeVoid = [self.SizeVoid]*self.ndim
        # self.SizeVoid = np.array(self.SizeVoid)[:self.ndim]

        # assert( np.all(self.SizeCell >= self.SizeVoid) )

        # self.RemoveLines = kwargs.get('RemoveLines', [False]*self.ndim)
        # self.RemoveLines = np.array(self.RemoveLines)[:self.ndim]

        # self.margin = np.rint( (self.SizeCell - self.SizeVoid) / 2 ).astype(np.int)
        # assert( np.all(2*self.margin + self.SizeVoid == self.SizeCell) )

        # self.nCells = np.rint(self.Window.shape / self.SizeCell).astype(np.int)
        # assert( np.all(self.nCells*self.SizeCell == self.Window.shape) )

        # x = [ np.abs(np.linspace(-1,1, self.SizeCell[j])) for j in range(self.ndim) ]
        # x = [ x[j]*(1-self.RemoveLines[j])  for j in range(self.ndim) ] ### Remove lines of the grid
        # x = np.array(np.meshgrid(*x, indexing='ij'))
        # # x = torch.tensor(x)
        # x = x.max(axis=0)
        # self.cell = x

        # self.field = np.tile(self.cell, self.nCells )
        # # self.field = torch.tile(self.cell, tuple(self.nCells) )
        # self.field = torch.tensor(self.field)

        # self.angle = kwargs.get('angle', pi/3)
        # self.shift = kwargs.get('shift', 20)
        # self.scale = 0.1
        # self.shift_layers = kwargs.get('shift_layers', np.random.randint(low=0, high=2))

        
        SizeCell = kwargs.get('SizeCell', [40,40,40])
        nCells   = kwargs.get('nCells', None)
        if nCells is not None:
            self.nCells   = np.array(nCells)
            self.SizeCell = self.Window.shape / self.nCells
        else:
            self.SizeCell = [SizeCell]*self.ndim if np.isscalar(SizeCell) else SizeCell[:self.ndim]
            self.SizeCell = np.array(self.SizeCell)

        SizeVoid = kwargs.get('SizeVoid', [20,20,20])
        self.SizeVoid = [SizeVoid]*self.ndim if np.isscalar(SizeVoid) else SizeVoid[:self.ndim]
        self.SizeVoid = np.array(self.SizeVoid)

        assert( np.all(self.SizeCell >= self.SizeVoid) )

        self.RemoveLines = kwargs.get('RemoveLines', [False]*self.ndim)
        self.RemoveLines = np.array(self.RemoveLines)[:self.ndim]

        self.margin = np.rint( (self.SizeCell - self.SizeVoid) / 2 ).astype(np.int)
        assert( np.all(2*self.margin + self.SizeVoid == self.SizeCell) )

        self.nCells = np.rint(self.Window.shape / self.SizeCell).astype(np.int)
        # assert( np.all(self.nCells*self.SizeCell == self.N) )

     
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
        axes = [np.arange(n) for n in self.Window.shape]
        x = 1.*np.stack(np.meshgrid(*axes), axis=-1)

        xi = 2 * (x % self.SizeCell) / self.SizeCell - 1.
        xi = torch.tensor(xi)

        norm = xi.max()

        field = xi.norm(p=inf,dim=-1)
        return field


    def sample3d(self, noise=None):     
        axes = [np.arange(n) for n in self.Window.shape]
        x = 1.*np.stack(np.meshgrid(*axes), axis=-1)

        xi = 2 * (x % self.SizeCell) / self.SizeCell - 1.
        xi = torch.tensor(xi)

        y = torch.zeros_like(xi)
        for j in range(self.ndim):
            y[...,j] = xi.roll(j,-1)[...,1:].norm(p=inf,dim=-1)

        field = y.norm(p=-inf,dim=-1)
        return field






