
import numpy as np
from math import *
from time import time
import matplotlib.pyplot as plt
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.GaussianRandomField import GaussianRandomField


#######################################################

config = {
    'grid_level'        : 9,
    'ndim'              : 2,
### Covariance (default: Matern)
    'nu'                :   0.5, 
    'correlation_length':   0.03,
    'Folded_GRF'        :   False,
### other
    'verbose'           : True,
}
EXPORTDIR = "./"


#######################################################

### Create Random Material
GRF = GaussianRandomField(**config)
# GRF.seed(0)

### Create a sample
time_start = time()
nsamples = 3
for i in range(nsamples):
    Y = GRF.sample_numpy()
    K = np.exp(Y)
    filename = os.path.abspath(os.path.join(EXPORTDIR, f"sample_{i}"))
    GRF.save_vtk(filename, K)
print('Sample runtime: ', time()-time_start)


### Info
print()
print('vf = ', K.mean())
print('shape: ', K.shape)
print()


#######################################################

plt.show()