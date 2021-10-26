
import numpy as np
from math import *
from time import time
import matplotlib.pyplot as plt


import sys
sys.path.append("/home/khristen/Projects/Paris/RandomMaterialCode/source")

from SurrogateMaterialModeling.HybridMaterial import HybridMaterial
from SurrogateMaterialModeling.Particles import ParticlesCollection
from SurrogateMaterialModeling.CovarianceKernels import MaternCovariance


#######################################################

a = 0.01
tau = 0.99

config = {
    'grid_level'        : 9,
    'ndim'              : 2,
    'tau'               : 0.9,
    'nu'                : 1,
    'corrlen'           : 0.02,
    'sampling_method'   : 'fftw'
}
config['Covariance'] = MaternCovariance(**config)

seed = np.random.randint(100)


#######################################################


PC = ParticlesCollection(**config, min_particles_number=10, max_particles_number=40, rho=10)
HM = HybridMaterial(alpha=a, ParticlesModel=PC, **config)

HM.sample()

# HM.save_png(HM.field, "/home/khristen/Projects/random_material/paper2/Draft_SIAM/figures/example_ellipses_smooth3.png")

print(np.amax(HM.field_0))
print(np.mean(HM.field_1))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(HM.field_0)
plt.subplot(1,3,2)
plt.imshow(HM.field_2)
plt.subplot(1,3,3)
plt.imshow(HM.field)
plt.show()

# save_png(phase, "/home/khristen/Projects/random_material/paper2/Draft_SIAM/figures/example_ellipses_smooth3.png")

#######################################################

### Output

# plt.figure()
# plt.subplot(1,3,1)
# plt.title("Angle field")
# if A is not None: plt.imshow(A)
# plt.subplot(1,3,2)
# plt.title("ODE-based loc.anis")
# plt.imshow(X)
# plt.subplot(1,3,3)
# plt.title("Classic")
# plt.imshow(X0)

# save_png(X, "./apps/test_Anis.png")


# plt.figure()
# err = RF.GRF.test_Covariance(nsamples=1000)

# plt.figure()
# err = RM0.GRF.test_Covariance(nsamples=1000)


#######################################################

plt.show()