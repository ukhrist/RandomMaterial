

from math import *
import numpy as np
from scipy.optimize import fsolve
from time import time
import torch

from ..RandomField import RandomField
from ..GaussianRandomField import GaussianRandomField

class GrainClustersStructure(RandomField):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        SizeCell = kwargs.get('SizeCell', [80,120])
        self.SizeCell = [SizeCell]*self.ndim if np.isscalar(SizeCell) else SizeCell[:self.ndim]
        self.SizeCell = np.array(self.SizeCell)

        # layer_dir = 0
        self.SizeLayer = self.SizeCell[1]
        self.nLayers   = np.floor(self.shape[0] / self.SizeLayer).astype(np.int)

        ### NOTE: probably, the axes are messed up (owing to 'ij' or 'xy' indexing), but given a square it does not matter..

        self.nClustersPerLayer = kwargs.get('nClustersPerLayer', 1)
        self.ClusterRadius     = kwargs.get('ClusterRadius', 0.1)

        self.nClusters = self.nLayers * self.nClustersPerLayer

        self.xClusterCenters = [ [i*self.SizeLayer]*self.nClustersPerLayer for i in range(self.nLayers) ]
        self.xClusterCenters = np.array(self.xClusterCenters).flatten()

        self.ClusterCenters = np.zeros([self.nClusters, 2])
        self.ClusterCenters[:,1] = self.xClusterCenters
        self.ClusterCenters[:,0] = np.random.randint(self.shape[1], size=self.nClusters)

        ClusterFunctionType = kwargs.get("ClusterFunctionType", "Gauss") ### "Heaviside", "Gauss"
        if ClusterFunctionType == 'Heaviside':
            self.ClusterFunction = lambda R: np.heaviside(self.ClusterRadius-R, 0)
        elif ClusterFunctionType == 'Gauss':
            self.ClusterFunction = lambda R: 1/self.ClusterRadius * np.exp(-0.5*(R/self.ClusterRadius)**2)

        GRF_config = dict(
            ndim                = self.ndim,
            grid_level          = kwargs.get('grid_level', 7),
            nu                  = kwargs.get('nu', 2.5),
            correlation_length  = kwargs.get('correlation_length', 0.01),
            Folded              = kwargs.get('Folded', False)
        )
        self.GRF = GaussianRandomField(**GRF_config)
        self.scale = kwargs.get('scale', 1)

     
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
        X = np.stack(np.meshgrid(*axes, indexing="xy"), axis=0)

        ### Normilize coordinates (image=[0,1]^d)
        X = X / self.shape[:,None,None]
        ClusterCenters = self.ClusterCenters / self.shape

        R = np.ones(self.shape)
        for i, c in enumerate(ClusterCenters):
            ri = np.linalg.norm(X-c[:,None,None], axis=0)
            R  = np.minimum(R, ri)

        # Clusters = np.where(R<self.ClusterRadius, 1, 0)
        Clusters = self.ClusterFunction(R)
        Clusters = torch.tensor(Clusters)

        Field = Clusters * self.GRF.sample()
        Field = self.scale * Field
        return Field


    def sample3d(self, noise=None):
        ### Not implemented
        return None

    