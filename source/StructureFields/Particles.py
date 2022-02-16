

from math import *
import numpy as np
from scipy.optimize import fsolve
from time import time

import torch
from torch import nn
# import pytorch3d
from scipy.spatial.transform import Rotation

from ..RandomField import RandomField

class ParticlesCollection(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.min_particles_number, self.max_particles_number = kwargs.get('Uniform_range', [1, 10])
        self.Poisson_mean = kwargs.get('Poisson_mean', None)

        # self.Poisson_mean = kwargs.get('Poisson_mean', 10)
        # self.Poisson_mean = nn.Parameter(torch.tensor(float(self.Poisson_mean)))

        self.set_radius(**kwargs)

        axes = [torch.arange(n)/n for n in self.Window.shape]
        self.coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), axis=-1).detach()


    def set_radius(self, **kwargs):
        self.fg_anisotrop = kwargs.get('anisotrop', False)
        if self.ndim==3 and self.fg_anisotrop:
            self.par_radius = nn.Parameter(torch.tensor([0.]*self.ndim))
        else:
            self.par_radius = nn.Parameter(torch.tensor([0.]))
        radius = kwargs.get('radius', 0.1)
        if np.isscalar(radius):
            self.radius = radius
        else:
            self.radius = radius[0]

        # semiaxes = kwargs.get('semiaxes', 1) ### can be vector
        # self.isotropic = np.isscalar(semiaxes)
        # semiaxes = [semiaxes]*self.ndim if np.isscalar(semiaxes) else semiaxes[:self.ndim]
        # self.semiaxes = np.array(semiaxes)

        
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
        self.par_radius.data[:] = torch.tensor(radius, dtype=torch.double).sqrt()

    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample(self):
        nParticles = self.draw_nParticles()
        centers    = 0.1 + 0.8*torch.rand(self.ndim, nParticles)
        # centers    = torch.rand(self.ndim, nParticles)
        x          = self.coordinates.unsqueeze(-1)

        if self.ndim==3 and self.fg_anisotrop:
            euler_angles      = (2*pi) * torch.rand(nParticles, self.ndim).detach()
            rotation_matrices = torch.tensor( Rotation.from_euler("xyz", euler_angles.detach().numpy()).as_matrix() ).detach() ### numpy version
            # rotation_matrices = pytorch3d.transforms.euler_angles_to_matrix(euler_angles, "XYZ") ### torch version (needs last version of torch3d)
            rotation_matrices = torch.moveaxis(rotation_matrices, 0, -1)
            radius            = self.radius.unsqueeze(-1) * torch.zeros(self.ndim, nParticles).log_normal_(mean=0, std=1.e-2) ### 1%-deviation
            v   = (x-centers)
            Rv  = torch.zeros_like(v).detach()
            for iParticle in range(nParticles): ### number of particles is not big, so for loop is ok
                R  = rotation_matrices[...,iParticle]
                vi = v[...,iParticle].unsqueeze(-2)
                Rv[...,iParticle] = (R * vi).sum(dim=-1)
            # if np.max(v.shape)>2**7:
            #     i   = v.shape[0]//2
            #     Rv1 = (rotation_matrices * v[:i,...].unsqueeze(-3)).sum(dim=-2)
            #     Rv2 = (rotation_matrices * v[i:,...].unsqueeze(-3)).sum(dim=-2)
            #     Rv  = torch.stack([Rv1, Rv2], dim=0)
            # else:
            #     Rv  = (rotation_matrices * v.unsqueeze(-3)).sum(dim=-2)
            DRv = Rv / radius
            r   = DRv.square().sum(dim=-2).sqrt()    
            R,_ = radius.min(dim=-2)
            field, _ = (R*(1-r)).max(dim=-1)    ### maximum cone
        else: ### isotropic particles
            radius = self.radius * torch.zeros(nParticles).log_normal_(mean=0, std=1.e-2) ### 1%-deviation
            r = (x-centers)
            # r = r.abs()               ### periodicity of distance
            # r = torch.minimum(r, 1-r) ### periodicity of distance
            r = r.norm(dim=-2)        
            field, _ = (radius-r).max(dim=-1)    ### maximum cone
  
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

    