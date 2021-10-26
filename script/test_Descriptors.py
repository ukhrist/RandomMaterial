
import numpy as np
from math import *
from time import time
import matplotlib.pyplot as plt


from SurrogateMaterialModeling.HybridMaterial import HybridMaterial
from SurrogateMaterialModeling.Particles import ParticlesCollection
from SurrogateMaterialModeling.CovarianceKernels import MaternCovariance
from SurrogateMaterialModeling.utilities.common import Matern_function
from SurrogateMaterialModeling.Reconstruction.Descriptors import compute_nPointProbability
from SurrogateMaterialModeling.Reconstruction.Inference import MaterialEncoder
import SurrogateMaterialModeling.Reconstruction.Parameters as Parameters


#######################################################

# conf = lambda: None
# conf.grid_level = 8
# print(conf.__dict__)

R_def = 0.05
tau = np.exp(-R_def)

config = {
    'grid_level'        : 8,
    'ndim'              : 2,
    'vf'                : 0.1,
    'sampling_method'   : 'fftw',
    'window_margin=0'   : 0,

    'nu'                : 1.2,
    'corrlen'           : 0.05,

    'alpha'             : 0.1,
    'tau'               : tau,

    'rho'               : 5,
    'semiaxes'          : [1, 1],

}
config['Covariance'] = MaternCovariance(**config)

seed = np.random.randint(100)


#######################################################


PC = ParticlesCollection(**config)
HM = HybridMaterial(ParticlesModel=PC, **config)

# t = time()
# S1 = compute_nPointProbability(HM, n=1)
# print('vf(predict) =', 1-S1)
# S2 = compute_nPointProbability(HM, n=2, nbins=20)
# print('Total time', time()-t)
# plt.plot(S2)

X = HM.sample()
plt.figure()
plt.subplot(1,3,1)
plt.imshow(HM.field_0)
plt.subplot(1,3,2)
plt.imshow(HM.field_2)
plt.subplot(1,3,3)
plt.imshow(HM.field)
plt.show()
print('vf =', X.mean())

nsamples = 10
Samples = ( HM.sample() for _ in range(nsamples) )


HM0 = HybridMaterial(ParticlesModel=PC, **config)
Parameters.set_DesignParameters(HM0, np.log([50, 0.5, 0.1]) )
MaterialEncoder(HM, Samples)


#######################################################

plt.show()