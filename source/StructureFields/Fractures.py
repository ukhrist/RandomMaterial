

from math import *
import numpy as np
from scipy.optimize import fsolve
from time import time

import torch
from torch import nn

from sympy import Point3D, Line3D, Plane
from pyrr import rectangle, vector, vector3, plane


from ..RandomField import RandomField

class FracturesCollection(RandomField):

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
        self.coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), axis=-1).detach()

        self.tau        = kwargs.get('tau', 0)


        
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
        # centers    = torch.rand(self.ndim, nParticles) # couple of points to define the 
        P1         = torch.rand(self.ndim, nParticles)
        x1 = P1[0,:]
        y1 = P1[1,:]
        P2         = torch.rand(self.ndim, nParticles)
        x2 = P2[0,:]
        y2 = P2[1,:]


        radius     = torch.tensor( np.random.lognormal(mean=np.log(self.radius.item()), sigma=np.sqrt(3*self.radius.item()), size=nParticles)  ) # opening = 2*radius

        x = self.coordinates.unsqueeze(-1)
        x0=x[:,:,0,:]
        y0=x[:,:,1,:]

        if self.ndim==3:
            z0 = x[:,:,2,:]
            z1 = P1[2,:]
            z2 = P2[2,:]
            P3         = torch.rand(self.ndim, nParticles)
            x3 = P3[0,:]
            y3 = P3[1,:]
            z3 = P3[2,:]



        dx12 = x2 - x1
        dx01 = x1 - x0
        dy01 = y1 - y0
        dy12 = y2 - y1
        l12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        l012 = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))

        # r = (x-centers).norm(dim=-2) #define here distance point to segment


            # vx = xy0(1)-xyP(1);->dx01
    # vy = xy0(2)-xyP(2);->dy01
    # ux = xy1(1)-xy0(1);->dx12
    # uy = xy1(2)-xy0(2);->dy12
        lenSqr= (dx12**2+dy12**2)
        detP= -dx01*dx12 - dy01*dy12

        # if detP<0:
        #     r = (P1-x).norm(dim=-2)
        # elif detP > lenSqr:
        #     r = (P2-x).norm(dim=-2)
        # else:
        #     r = l012/l12

        if self.ndim == 2:
            r =(P1-x).norm(dim=-2) * (detP<0) + (P2-x).norm(dim=-2) * (detP>lenSqr) + l012/l12 * (detP>0) * (detP<lenSqr)


        # r = l012/l12
        if self.ndim == 3:
            u,v,w, g,h = x.shape
            r = np.zeros([u,v,w,nParticles])
            for i in range(nParticles):
                pl = plane.create_from_points(P1[:,i],P2[:,i],P3[:,i])
                n = plane.normal(pl)
                p = n * plane.distance(pl)
                d = np.dot(p, n)
                for j in range(u): 
                    for k in range(v):
                        for l in range(w):
                            point = x[j,k,l,:,:]
                            qn = np.dot(point.T, n)
                            pp = point.T + (n * (d - qn))
                            dx = point - pp.T
                            r[j,k,l,i] = sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)

                            # # bounded plane
                            # pa = P1[:,i] - pp

                            # #test if coplanar point is in bounds
                            # if not np.dot(P1[:,i],P2[:,i]-P1[:,i]) <= np.dot(pa,P2[:,i]-P1[:,i]) and np.dot(pa,P2[:,i]-P1[:,i]) <= np.dot(P2[:,i],P2[:,i]-P1[:,i]) and np.dot(P1[:,i],P3[:,i]-P1[:,i]) <= np.dot(pa,P3[:,i]-P1[:,i])  and np.dot(pa,P3[:,i]-P1[:,i])  <= np.dot(P3[:,i],P3[:,i]-P1[:,i]):

                            #     #/* draw lines to test shaded area is in bounds */

                            #     #// distance to closest edge
                            #     P4 = P3[:,i] + P2[:,i] - P1[:,i] # // 4th corner of rect

                            #         #// closest point on each edge
                            #     labp = self.measure_dist(point, self.closestPoint(P1[:,i], P2[:,i], point.T))
                            #     lacp = self.measure_dist(point, self.closestPoint(P1[:,i], P3[:,i], point.T))
                            #     lbdp = self.measure_dist(point, self.closestPoint(P2[:,i], P4, point.T))
                            #     lcdp = self.measure_dist(point, self.closestPoint(P3[:,i], P4, point.T))

                            #         # find minimum closest edge point
                            #     dist = min(labp, lacp)
                            #     dist = min(dist, lbdp)
                            #     dist = min(dist, lcdp)
                            #     r[j,k,l,i] = dist



        # pl = Plane(Point3D(x1, y1, z1), Point3D(x2, y2, z2), Point3D(x3, y3, z3))
        # xx = Point3D(x0, y0, z0)
        # a.distance(b)

        # c = Line3D(Point3D(2, 3, 1), Point3D(1, 2, 2))
        # a.distance(c)

        # d = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))



#         from sympy import Point3D, Line3D, Plane
# a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
# b = Point3D(1, 2, 3)
# a.intersection(b)
# [Point3D(1, 2, 3)]
# c = Line3D(Point3D(1, 4, 7), Point3D(2, 2, 2))
# a.intersection(c)
# [Point3D(2, 2, 2)]
# d = Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
# e = Plane(Point3D(2, 0, 0), normal_vector=(3, 4, -3))
# d.intersection(e)
# [Line3D(Point3D(78/23, -24/23, 0), Point3D(147/23, 321/23, 23))]

# from sympy import Plane, Point3D
# a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
# a.perpendicular_line(Point3D(9, 8, 7))
# Line3D(Point3D(9, 8, 7), Point3D(11, 12, 13))



        field, _ = self.cone(r, radius).max(dim=-1)
        field = field - self.tau

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


    def draw_nParticles(self):  # particles --> segments 
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

    def closestPoint(self,x,y,p):
        xp = (x-p).T
        yp = (y-p).T
        dxp = xp[0]**2 + xp[1]**2 + xp[2]**2
        dyp = yp[0]**2 + yp[1]**2 + yp[2]**2
        if dxp < dyp:
            return x
        else:
            return y

    def measure_dist(self,x,y):
        dxy = (x.T-y).T
        return sqrt(dxy[0]**2 + dxy[1]**2 + dxy[2]**2)
    