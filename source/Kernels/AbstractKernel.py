
from math import pi, sqrt, sin, cos
import torch
from torch import nn
import numpy as np
from scipy.spatial.transform import Rotation


"""
==================================================================================================================
Abstract Kernel class
==================================================================================================================
"""

class Kernel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.verbose = kwargs.get('verbose', False)
        self.ndim    = int(kwargs.get('ndim', 2))   ### dimension 2D or 3D
        if self.ndim not in (2,3):
            raise Exception('Unsuppoerted dimension: {0:d}'.format(self.ndim))

        self.dtype = kwargs.get('dtype', torch.float64)
        torch.set_default_dtype(self.dtype)

    def forward(self, freq):
        return self.eval_spec(freq)

    def eval_func(self, x):
        return 0

    def eval_spec(self, freq):
        return 0




"""
==================================================================================================================
Construct Metric matrix
==================================================================================================================
"""

def set_Metric(ndim=2, length=1, angle=0, axle=None, out='matrix'): ### out: 'matrix', 'vectors'
    length = [length]*ndim if np.isscalar(length) else length[:ndim]
    length = np.array(length)

    if ndim==2:
        if angle:
            R = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)],
                ])
            A = R.T @ np.diag(length**2) @ R
        else:
            A = np.diag(length**2)

    elif ndim==3:
        if axle is not None:
            R = Rotation.from_rotvec(axle).as_matrix()
            A = R.T @ np.diag(length**2) @ R
        else:
            A = np.diag(length**2)

    else:
        raise Exception('Unsuppoerted dimension: {0:d}'.format(ndim))

    if out=='matrix':
        return A
    elif out=='vectors':
        if axle is None: axle = np.zeros_like(length)
        return length, axle

"""
==================================================================================================================
Rotation matrix
==================================================================================================================
"""
#TODO: rotate vector
def rotate(v, axle=None):
    # angles = 0, 0, 0
    # # angles = pi/4, pi/4, 0

    # a = angles[0]
    # Rx1 = v[...,0]
    # Rx2 = cos(a) * v[...,1] - sin(a) * v[...,2]
    # Rx3 = sin(a) * v[...,1] + cos(a) * v[...,2]

    # a = angles[1]
    # Ry1 = cos(a) * Rx1 - sin(a) * Rx3
    # Ry2 = Rx2
    # Ry3 = sin(a) * Rx1 + cos(a) * Rx3

    # a = angles[2]
    # Rz1 = cos(a) * Ry1 - sin(a) * Ry2
    # Rz2 = sin(a) * Ry1 + cos(a) * Ry2
    # Rz3 = Ry3

    # v = torch.stack([Rz1, Rz2, Rz3], dim=-1)
    return v