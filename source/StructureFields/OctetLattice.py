
from math import inf, pi
import torch
from torch import nn
import numpy as np

from ..RandomField import RandomField

class OctetLatticeStructure(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
               
        SizeCell = kwargs.get('SizeCell', [40,40,40])
        nCells   = kwargs.get('nCells', None)
        if nCells is not None:
            self.nCells   = np.array(nCells)
            self.SizeCell = self.Window.shape / self.nCells
        else:
            self.SizeCell = [SizeCell]*self.ndim if np.isscalar(SizeCell) else SizeCell[:self.ndim]
            self.SizeCell = np.array(self.SizeCell)

        self.par_thickness = nn.Parameter(torch.tensor([0.], dtype=float))
        self.thickness = kwargs.get('thickness', 0.1)

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

        self.subcell_size = (self.SizeCell // 2).astype(np.int)

        self.fg_pores = kwargs.get("nodal_pores", False)

        if self.ndim == 2:
            self.init_2d()
        elif self.ndim == 3:
            self.init_3d()
        else:
            raise Exception('Dimension is not supported.')


    @property
    def thickness(self):
        return torch.exp(self.par_thickness)
        # return self.par_thickness.square()

    @thickness.setter
    def thickness(self, thickness):
        self.par_thickness.data[:] = torch.log(torch.tensor(float( thickness )))
        # self.par_thickness.data[:] = torch.tensor(float( thickness )).sqrt()

    
    def init_2d(self):
        pass


    def init_3d(self):

        subcell_scale = 0.5/self.nCells

        ### prepare subcell       
        grid = [ (0.5 + np.arange(n))/n for n in self.subcell_size ]
        x = 1.*np.stack(np.meshgrid(*grid), axis=-1)
        x = torch.tensor(x, dtype=torch.float).detach()
        x = x * subcell_scale

        nodes = torch.tensor([[0,0,0], [0,1,1], [1,0,1], [1,1,0]], dtype=torch.float).detach()
        nodes = nodes * subcell_scale
        edges = [ 
            [ nodes[0], nodes[1] ],
            [ nodes[0], nodes[2] ],
            [ nodes[0], nodes[3] ],
            [ nodes[1], nodes[2] ],
            [ nodes[1], nodes[3] ],
            [ nodes[2], nodes[3] ]
        ]

        # subcell = torch.ones(*self.subcell_size).detach() * 100.
        # for e in edges:
        #     val_loc = dist_to_edge(x, e)
        #     subcell = torch.minimum(subcell, val_loc)


        ls_dist_edges = [dist_to_edge(x, e) for e in edges]
        # ls_dist_nodes = [0.45*dist(x, n) for n in nodes]
        ls_dist = ls_dist_edges #+ ls_dist_nodes
        subcell, indices = torch.stack(ls_dist,  dim=0).min(dim=0)

        if self.fg_pores:  ### Nodal pores
            ls_dist_nodes = [dist(x, n) for n in nodes]
            node_dist, indices = torch.stack(ls_dist_nodes,  dim=0).min(dim=0)

            r_bubble = 0.15 * subcell_scale[0]
            R_bubble = 0.2 * subcell_scale[0]

            # bubble1 = torch.relu(r_bubble-node_dist)
            bubble1 = (r_bubble-node_dist).relu()
            bubble2 = (R_bubble-node_dist).abs()

            subcell = torch.minimum(subcell, bubble2)
            subcell = torch.maximum(subcell, 10*bubble1)
            
        self.location_field = x
        self.subcell_nodes  = nodes
        self.subcell_edges  = edges
        self.subcell_dists  = subcell

     
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
        cell  = self.generate_cell()
        field = torch.tile(cell, dims=tuple(self.nCells))
        return field


    def generate_cell(self):
        cell = self.generate_subcell()
        for i in range(self.ndim):
            cell = torch.cat([cell, cell.flip(dims=(i,))], dim=i)
        return cell


    def generate_subcell(self):
        return 0.5*self.thickness - self.subcell_dists


def dist(x, y):
    return (x-y).norm(dim=-1)


def dist_to_edge(x, e):
    a = dist(x, e[0])
    b = dist(x, e[1])
    c = dist(e[0], e[1])
    h = (a**2 + b**2)/2 - c**2/4 - ((a**2-b**2)/(2*c))**2
    h = h.sqrt()
    return h






