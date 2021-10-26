
import numpy as np
from math import *
from time import time
import matplotlib.pyplot as plt


from SurrogateMaterialModeling.HybridMaterial import HybridMaterial
from SurrogateMaterialModeling.GridSupportField import GridSupportField
from SurrogateMaterialModeling.CovarianceKernels import MaternCovariance


#######################################################

a = 0.02
tau = 0.5

config = {
    'grid_level'        : 9,
    'ndim'              : 2,
    'vf'                : 0.35,
    'nu'                : 10,
    'corrlen'           : 0.2,
    'sampling_method'   : 'fftw',

    'SizeCell'          : 2**7,
    'SizeVoid'          : 2*int(2**6 * 0.8),
}
config['Covariance'] = MaternCovariance(**config)

seed = np.random.randint(100)


#######################################################


Support = GridSupportField(**config)
HM = HybridMaterial(alpha=a, ParticlesModel=Support, **config)

HM.sample()

HM.save_png(HM.field, "/home/khristen/Projects/Additive_Manufacturing/ParticleSizeModel/ParticleSize_Model/code/script/Results/GausPerturbOfGrid6.png")

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

# save_png(phase, "/home/khristen/Projects/Additive_Manufacturing/ParticleSizeModel/ParticleSize_Model/code/script/Results")

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