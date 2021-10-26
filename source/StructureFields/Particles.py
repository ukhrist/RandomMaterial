

from math import *
import numpy as np
from scipy.optimize import fsolve
from time import time

import torch
from torch import nn

from ..RandomField import RandomField

class ParticlesCollection(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.min_particles_number, self.max_particles_number = kwargs.get('Uniform_range', [1, 10])
        self.Poisson_mean = kwargs.get('Poisson_mean', None)

        # self.Poisson_mean = kwargs.get('Poisson_mean', 10)
        # self.Poisson_mean = nn.Parameter(torch.tensor(float(self.Poisson_mean)))

        # semiaxes = kwargs.get('semiaxes', 1) ### can be vector
        # self.isotropic = np.isscalar(semiaxes)
        # semiaxes = [semiaxes]*self.ndim if np.isscalar(semiaxes) else semiaxes[:self.ndim]
        # self.semiaxes = np.array(semiaxes)

        self.par_radius = nn.Parameter(torch.tensor(0.))
        self.radius = kwargs.get('radius', 0.1)

        axes = [torch.arange(n)/n for n in self.Window.shape]
        self.coordinates = torch.stack(torch.meshgrid(*axes), axis=-1).detach()


        
    #--------------------------------------------------------------------------
    #   Properties
    #--------------------------------------------------------------------------

    @property
    def radius(self):
        # return torch.exp(self.__logRadius)
        return self.par_radius.square()

    @radius.setter
    def radius(self, radius):
        # self.__logRadius.data = torch.log(torch.tensor(float(radius)))
        self.par_radius.data = torch.tensor(float(radius)).sqrt()

    # @semiaxes.setter
    # def semiaxes(self, semiaxes):
    #     _semiaxes = np.array(semiaxes)
    #     if _semiaxes.size == self.ndim:
    #         _semiaxes = _semiaxes / np.prod(_semiaxes)**(1/self.ndim)
    #     elif _semiaxes.size == self.ndim-1:
    #         _semiaxes = np.append(_semiaxes, 1/np.prod(_semiaxes))
    #     else:
    #         _semiaxes = _semiaxes*np.ones(self.ndim)
    #     self.__semiaxes = _semiaxes
    #     assert np.isclose(np.prod(self.semiaxes), 1)
    #     self.isotropic = all([a==self.semiaxes[0] for a in self.semiaxes])

     
    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample(self):
        nParticles = self.draw_nParticles()
        # centers    = np.random.uniform(size=[nParticles, self.ndim])
        centers    = torch.rand(self.ndim, nParticles)
        radius     = torch.tensor( np.random.lognormal(mean=np.log(self.radius.item()), sigma=np.sqrt(3*self.radius.item()), size=nParticles)  )

        # axes = [torch.arange(n)/n for n in self.Window.shape]
        # x = torch.stack(torch.meshgrid(*axes), axis=-1)
        x = self.coordinates.unsqueeze(-1)

        r = (x-centers).norm(dim=-2)
        field, _ = self.cone(r, radius).max(dim=-1)

        # field = torch.tensor(-inf)
        # for i in range(nParticles):
        #     c = centers[i,:]
        #     r = x-c
        #     # if not self.isotropic: ### 2D only !!! TODO: rewrite using PyTorch
        #     #     alpha = np.random.uniform(0, 2*pi)
        #     #     R = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
        #     #     r = np.einsum("...i,ij->...j", r, R)
        #     #     r = r/self.semiaxes
        #     # r = np.sqrt(np.sum(r**2, axis=-1))
        #     r = r.norm(dim=-1)
        #     field_loc = self.cone(r)
        #     field = torch.maximum(field, field_loc)

        # field = torch.tensor(field)        
        return field


    def draw_nParticles(self):
        if self.Poisson_mean:
            nParticles = np.random.poisson(self.Poisson_mean)
            # nParticles = torch.poisson(torch.tensor((self.Poisson_mean)))
        else:
            # nParticles = np.random.randint(self.min_particles_number, self.max_particles_number)
            nParticles = torch.randint(self.min_particles_number, self.max_particles_number)
        if self.verbose: print(nParticles, "particles")
        return nParticles


    def cone(self, r, R):
        return 1.-r/R

    