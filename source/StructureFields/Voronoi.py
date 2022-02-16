

from math import *
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import Voronoi, voronoi_plot_2d
from time import time

import matplotlib.pyplot as plt

import torch
from torch import nn

from ..RandomField import RandomField

class VoronoiCollection(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.min_particles_number, self.max_particles_number = kwargs.get('Uniform_range', [1, 10])
        self.Poisson_mean = kwargs.get('Poisson_mean', None)

        self.par_radius = nn.Parameter(torch.tensor(0.))
        self.radius = kwargs.get('radius', 0.1)

        axes = [torch.arange(n)/n for n in self.Window.shape]
        self.coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), axis=-1).detach()


        
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

     
    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def sample(self):
        nParticles = self.draw_nParticles()
        # centers    = np.random.uniform(size=[nParticles, self.ndim])
        centers    = torch.rand(self.ndim, nParticles)
        radius     = torch.tensor( np.random.lognormal(mean=np.log(self.radius.item()), sigma=np.sqrt(3*self.radius.item()), size=nParticles)  )

        vor = Voronoi(centers.T)

        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)

        finite_segments = []
        infinite_segments = []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if (vor.furthest_site):
                    direction = -direction
                far_point = vor.vertices[i] + direction * ptp_bound.max()

                infinite_segments.append([vor.vertices[i], far_point])

        fig = voronoi_plot_2d(vor)
        plt.show()

        x = self.coordinates.unsqueeze(-1)

        r = (x-centers).norm(dim=-2)
        field, _ = self.cone(r, radius).max(dim=-1)
 
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

    