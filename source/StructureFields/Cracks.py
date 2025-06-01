
from math import pi, sqrt
import numpy as np
from numpy.core.numeric import ones_like
from scipy.optimize import fsolve
from time import time

import torch
from torch import nn

from ..RandomField import RandomField

class CracksCollection(RandomField):

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

        self.par_thickness = nn.Parameter(torch.tensor([0.]))
        self.thickness     = kwargs.get('thickness', 0.1)

        axes = [torch.arange(n)/n for n in self.Window.shape]
        self.coordinates = torch.stack(torch.meshgrid(*axes), axis=-1).detach()

        self.tau = kwargs.get('tau', 0)

        self.fg_periodic = kwargs.get('periodic', True)
        self.fixed_centers = None


        
    #--------------------------------------------------------------------------
    #   Properties
    #--------------------------------------------------------------------------

    @property
    def thickness(self):
        # return torch.exp(self.__logRadius)
        return self.par_thickness.square()

    @thickness.setter
    def thickness(self, thickness):
        # self.__logRadius.data]:] = torch.log(torch.tensor(float(radius)))
        self.par_thickness.data[:] = torch.tensor(float(thickness)).sqrt()

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

    def sample(self, periodic=None):
        if periodic is None:
            fg_periodic = self.fg_periodic
        else:
            fg_periodic = periodic
            
        if self.fixed_centers is None:
            nParticles = self.draw_nParticles()
            nParticles = int(np.maximum(nParticles, 2))
            centers    = torch.rand(self.ndim, nParticles) ### uniform on [0,1]
            print(f"Number of particles : {nParticles}")
        else:
            centers = self.fixed_centers
        self.centers = centers
        thickness = self.thickness # * torch.zeros(nParticles).log_normal_(mean=0, std=sqrt(3))

        x = self.coordinates.unsqueeze(-1)

        r = torch.abs(x-centers)    ### distance vector
        if fg_periodic: r = torch.minimum(r, 1-r)   ### periodicity of distance 
        distances_to_centers = r.norm(dim=-2) ### distances to the centers

        top2, indx = torch.topk(distances_to_centers, 2, dim=-1, largest=False)  ### two minimum distances

        a = top2[...,0]     ### distance from x to c1
        b = top2[...,1]     ### distance from x to c2
        # distance_to_boundaries = 0.5*torch.abs(a-b) ###  distance to the Voronoi tessalation boundaries

        ### distance from c1 to c2
        c = torch.abs(centers[:,indx[...,0]] - centers[:,indx[...,1]])
        if fg_periodic: c = torch.minimum(c, 1-c)  ### periodicity of distance 
        c = c.norm(dim=0)

        distance_to_boundaries = 0.5*torch.abs((a**2 - b**2)/c) ###  distance to the Voronoi tessalation boundaries

        field = thickness - distance_to_boundaries

        ### not optimal
        # r = (x-centers).norm(dim=-2)
        # value = self.cone(r, thickness)
        # field1, ind = value.max(dim=-1)

        # if self.ndim==2:
        #     I, J = ind.shape
        #     for i in range(I):
        #         for j in range(J):
        #             value[i,j,ind[i,j]]=value[i,j,ind[i,j]]-100000000000
        # elif self.ndim==3:
        #     I, J, K = ind.shape
        #     for i in range(I):
        #         for j in range(J):
        #             for k in range(K):
        #                 value[i,j,k,ind[i,j,k]]=value[i,j,k,ind[i,j,k]]-100000000000

        # field2, _ = value.max(dim=-1) 

        # field = - 0.5 * abs(field1-field2)
        # field = field - self.tau
        ### end "not optimal"

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
            nParticles = torch.poisson(torch.tensor(self.Poisson_mean).to(torch.double)).to(torch.int).item()
            # nParticles = self.PRNG.poisson(self.Poisson_mean)
            # nParticles = np.random.poisson(self.Poisson_mean)
            # nParticles = torch.poisson(torch.tensor((self.Poisson_mean)))
        else:
            # nParticles = np.random.randint(self.min_particles_number, self.max_particles_number)
            nParticles = torch.randint(self.min_particles_number, self.max_particles_number)
        if self.verbose: print(nParticles, "particles")
        return nParticles


    def cone(self, r, R):
        return 1.-r/R #-r

# def dist(x, y):
#     return (x-y).norm(dim=-1)

# def dist_to_edge(x, e):
#     a = dist(x, e[0])
#     b = dist(x, e[1])
#     c = dist(e[0], e[1])
#     h = (a**2 + b**2)/2 - c**2/4 - ((a**2-b**2)/(2*c))**2
#     h = h.sqrt()
#     return h

# def dist_to_centerline(x, c1, c2):
#     h = dist_to_edge(x, [c1, c2])
#     L = (c1-c2).norm
#     d = torch.sqrt(L**2 - h**2)

    