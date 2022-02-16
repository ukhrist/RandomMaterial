
from ntpath import join
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

from source.TwoPhaseMaterial import GaussianMaterial
from source.Kernels import MaternKernel


#######################################################

config = {
    'grid_level'        : 7,
    'ndim'              : 3,
    'vf'                : 0.3, #0.4,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   2, #1.62,
    'correlation_length':   [1,0.05,0.05], #0.08, #[1,1,0.05],
    'Folded_GRF'        :   False,
### other
    'verbose'           : True,
}
EXPORTDIR = "./"
filename  = os.path.abspath(os.path.join(EXPORTDIR, "sample_Simple2PhaseMat"))


#######################################################

### Create Random Material
time_start = time()
RM = GaussianMaterial(**config)
# RM.seed(0)
print('Init runtime: ', time()-time_start)

### Create a sample
time_start = time()
X = RM.sample_numpy()
print('Sample runtime: ', time()-time_start)


### Info
print()
print('vf = ', X.mean())
print('shape: ', X.shape)
print()


### Show 2D image
if config['ndim'] == 2:
    plt.figure()
    plt.imshow(X)


### Export as vtk
RM.save_vtk(filename, X)
print(f"Successfully saved as {filename}.vti")


#######################################################

plt.show()