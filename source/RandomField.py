
import torch
from torch import nn
import numpy as np
from math import ceil
import copy

import os, sys, csv
from tqdm import tqdm
from time import time

from imageio import imwrite
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt

# from utilities.Exports import exportVTK
# from utilities.ErrorMessages import *


"""
==================================================================================================================
Abstract Random Field class
==================================================================================================================
"""

class RandomField(nn.Module):

    def __init__(self, **kwargs):
        super(RandomField, self).__init__()
        self.verbose = kwargs.get('verbose', False)

        self.dtype = kwargs.get('dtype', torch.float64)
        torch.set_default_dtype(self.dtype)

        self.ndim  = int(kwargs.get('ndim', 2))   ### dimension 2D or 3D
        if self.ndim not in (2,3):
            raise Exception('Unsuppoerted dimension: {0:d}'.format(self.ndim))

        grid_level   = kwargs.get('grid_level', 7)
        grid_level   = [grid_level]*self.ndim if np.isscalar(grid_level) else grid_level[:self.ndim]
        self.shape   = np.round(2**np.atleast_1d(grid_level)).astype(np.intc)
        assert(len(self.shape) == self.ndim)

        # self.nvoxels = np.prod(self.shape)
        # self.mgrid   = np.array(np.meshgrid(*(np.arange(n) for n in self.shape)))
        # self.Volume  = 1

        ### Extended window domain
        self.Window = ExtendedWindow(domain_shape=self.shape, margin_length=kwargs.get('window_margin', 0))

        # Pseudo-random number generator
        self.PRNG = np.random.RandomState()

        # Settings flags
        self.FLAGS = {}

        # Sample/material label
        self.LABEL = {}

    #--------------------------------------------------------------------------
    #   Updates
    #--------------------------------------------------------------------------
    
    ### Reseed pseudo-random number generator
    def seed(self, seed=None):
        self.PRNG.seed(seed)
        torch.manual_seed(seed)

    def update(self):
        pass

    #--------------------------------------------------------------------------
    #   Sampling
    #--------------------------------------------------------------------------

    def forward(self, noise=None):
        return self.sample(noise)
    
    ### Generate a realization

    def sample(self, noise=None):
        raise NotImplementedError()

    def sample_numpy(self, noise=None):
        X = self.sample(noise)
        if torch.is_tensor(X): X = X.detach().numpy()
        return X

    def sample_labeled(self, noise=None):
        X = self.sample_numpy(noise)
        L = self.LABEL
        return X, L


    ### Generate a family of realizations

    def samples_generator(self, nsamples=10):
        for isample in range(nsamples):
            yield self.sample()

    def generate_samples(self, nsamples=1, path=None, output_format="png", append=False):
        output = False if path is None else True

        if output:
            if not append or not hasattr(self, 'sample_count'):
                os.system('rm -f ' + path + 'sample_*')
                self.sample_count = 0

        time_start = time()

        expected_vf = 0
        for isample in tqdm(range(nsamples)):
            phase = self.sample()
            expected_vf += phase.mean()
            if output:
                self.sample_count += 1
                filename = path + 'sample_{0:d}'.format(self.sample_count)
                if   self.ndim==2 and output_format == "png": self.save_png(phase, filename)
                elif self.ndim==3 or  output_format == "vtk": self.save_vtk(phase, filename)
        expected_vf /= nsamples

        print('All samples generation time: {0} s'.format(time()-time_start))
        print('Volume fraction: {0}'.format(expected_vf))


    #--------------------------------------------------------------------------
    #   Utils
    #--------------------------------------------------------------------------

    def copy(self):
        return copy.deepcopy(self)

    #--------------------------------------------------------------------------
    #   EXPORTS
    #--------------------------------------------------------------------------

    ### Save as an image (2D only)
    def save_png(self, filename, Sample=None, binary=False):
        assert(self.ndim==2)
        if Sample is None: Sample = self.sample_numpy()
        if binary:
            plt.rcParams['image.cmap']='binary_r'
            imsave(filename, Sample, format='png', vmin=0, vmax=1)   
        else:
            plt.rcParams['image.cmap']='jet'
            # plt.rcParams['image.cmap']='rainbow'
            imsave(filename, Sample, format='png', vmin=Sample.min(), vmax=Sample.max())            

    ### Save in vtk format
    def save_vtk(self, filename, Sample=None, noise=None, seed=None, field_name='field'):
        if type(Sample) is dict:
            cellData = Sample
        else:
            if Sample is None:
                if seed is not None: self.seed(seed)
                Sample_numpy = self.sample_numpy(noise)
            else:
                Sample_numpy = Sample
            cellData = {field_name : Sample_numpy}    

        for i, x in cellData.items():
            if torch.is_tensor(x):
                cellData[i] = x.detach().numpy()

        exportVTK(filename, cellData=cellData)


    ### Save as numpy array
    def save_numpy(self, filename, Sample=None):
        if Sample is None: Sample = self.sample_numpy()
        np.save(filename, Sample)


    #--------------------------------------------------------------------------
    #   Visualiziation
    #--------------------------------------------------------------------------

    def plot(self, Sample=None, show=True, ax=None, noise=None, seed=None):
        assert(self.ndim==2)
        if Sample is None:
            if seed is not None: self.seed(seed)
            Sample = self.sample_numpy(noise)
        if ax is None:
            ax = plt.imshow(Sample)
        else:
            ax.imshow(Sample)
        if show: plt.show()
        return ax

    """
    ==================================================================================================================
    Learnable parameters
    ==================================================================================================================
    """

    def set_parameters(self, named_parameters_dict, verbose=None):
        if verbose is None: verbose = self.verbose

        for name, p in self.named_parameters():
            if name in named_parameters_dict.keys():
                p.data[:] = named_parameters_dict[name]
                if verbose: print(name+' : ', p.item())
        self.update()

    def export_parameters(self, filename):
        named_parameters_dict = { name : p for name, p in self.named_parameters()}
        torch.save(named_parameters_dict, filename)

    def import_parameters(self, filename):
        named_parameters_dict = torch.load(filename)
        self.set_parameters(named_parameters_dict)
        



"""
==================================================================================================================
Extended window class
==================================================================================================================
"""

class ExtendedWindow:

    def __init__(self, domain_shape, margin_length=0):
        self.ndim = len(domain_shape)
        self.domain_shape = np.array(domain_shape).astype(np.intc)
        step   = 1/self.domain_shape
        margin = np.ceil(margin_length/step).astype(np.intc)
        self.shape = self.domain_shape + 2*margin
        self.scale = self.shape * step
        self.domain_slice = [slice(margin[i], margin[i] + self.domain_shape[i]) for i in range(self.ndim)]

    def crop_to_domain(self, field):
        return field[self.domain_slice]


"""
==================================================================================================================
Export to VTK format
==================================================================================================================
"""

from pyevtk.hl import imageToVTK

def exportVTK(FileName, cellData):
    shape   = list(cellData.values())[0].shape
    ndim    = len(shape)
    spacing = (1./min(shape), ) * 3

    if ndim==3:
        imageToVTK(FileName, cellData = cellData, spacing = spacing)

    elif ndim==2:
        cellData2D = {}
        for key in cellData.keys(): cellData2D[key] = np.expand_dims(cellData[key], axis=2)
        imageToVTK(FileName, cellData = cellData2D, spacing = spacing)

    else:
        raise Exception('Dimension must be 2 or 3.')


"""
==================================================================================================================
"""