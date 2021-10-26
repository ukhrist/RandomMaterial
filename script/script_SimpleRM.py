
import numpy as np
from math import *
from time import time
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/khristen/Projects/Paris/random_material/code/source/")
# sys.path.append("/home/khristen/Projects/Paris/RandomMaterialCode/source/")

from SurrogateMaterialModeling.RandomMaterial import RandomMaterial
from SurrogateMaterialModeling.CovarianceKernels import MaternCovariance


#######################################################

config = {
    'grid_level'        : 7,
    'ndim'              : 3,
    'vf'                : 0.3, #0.4,
    'nu'                : 2, #1.62,
    'corrlen'           : [1,0.05,0.05], #0.08, #[1,1,0.05],
    'angle_anis'        : pi/4,
    'sampling_method'   : 'fftw',
    # 'window_margin'     : -0.1,
}
config['Covariance'] = MaternCovariance(**config)


#######################################################

### Create Random Material
time_start = time()
RM = RandomMaterial(**config)
# RM.seed(0)
print('Create runtime: ', time()-time_start)

### Create a sample
time_start = time()
X = RM.sample()
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


### Exports

RM.save_np(X)
print('Successfully saved as npy')

RM.save_vtk(X)
print('Successfully saved as vtk')


#######################################################

plt.show()