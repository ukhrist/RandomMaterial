
import numpy as np
from math import *
from time import time
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/khristen/Projects/Paris/random_material/code/source/")


from SurrogateMaterialModeling.MultiPhaseMaterial import MultiPhaseMaterial
from SurrogateMaterialModeling.CovarianceKernels import MaternCovariance


#######################################################

# a = 0.01
# tau = 0.99
nPhases=2

config = {
    'grid_level'        : 10,
    'ndim'              : 2,
    'vf'                : 0.1,
    'nu'                : 10,
    'corrlen'           : 0.05, #[0.02,0.06],
    'angle_anis'        : pi/4,
    'sampling_method'   : 'fftw',
    'nPhases'           : nPhases
}
config['Covariance'] = MaternCovariance(**config)

seed = np.random.randint(100)


#######################################################


MPM = MultiPhaseMaterial(**config)

X = MPM.sample()
print(MPM.Covariance.corrlen)
print(X.astype(np.bool).astype(np.int).mean())

plt.figure()
plt.imshow(X)
plt.show()

MPM.save_png(X, "/home/khristen/Projects/random_material/paper2/Draft_SIAM/figures/example_multiphase.png")


#######################################################

plt.show()