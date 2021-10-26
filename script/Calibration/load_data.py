
import torch
import numpy as np
import matplotlib.pyplot as plt

from source.RMNet.StatisticalDescriptors.common import interpolate

def load_data_from_npy(filename, downsample_shape=None):
    Data_full = np.load(filename)
    n = np.min(Data_full.shape)
    # Data = torch.zeros([n]*Data_full.ndim, dtype=torch.float)
    Data = torch.tensor( Data_full[:n,:n,:n], dtype=float )
    if downsample_shape is not None:
        Data = interpolate(Data, downsample_shape)
    Data = [ Data ]
    return Data