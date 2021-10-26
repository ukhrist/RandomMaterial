


from RandomMaterial2.RandomMaterial import RandomMaterial
from RandomMaterial2.CovarianceKernels import MaternCovariance
from time import time
import matplotlib.pyplot as plt
import numpy as np


config = {
    'grid_level'        : 9,
    'ndim'              : 2,
    'vf'                : 0.05,
    'nu'                : 1.369,
    'corrlen'           : [0.08],
    'verbose'           : 1
}
cov = MaternCovariance(**config)

seed = 0*np.random.randint(100)


### TEST FIELD

t0 = time()
config['sampling_method'] = 'Rational'
RM = RandomMaterial(**config, Covariance=cov)
RM.reseed(seed)
print('Build(SPDE) = ', time()-t0)

t0 = time()
X = RM.sample()
print('Runtime(SPDE) = ', time()-t0)

RM.reseed(seed)
t0 = time()
X = RM.sample()
print('Runtime(SPDE) = ', time()-t0)


### REFERENCE FIELD

config['sampling_method'] = 'fftw'
RM_ref = RandomMaterial(**config, Covariance=cov)
RM_ref.reseed(seed)

t0 = time()
X_ref = RM_ref.sample()
print('Runtime(FFT) = ', time()-t0)



### COMPARE

print('vf:', X.mean(), X_ref.mean())

plt.figure()
plt.subplot(1,2,1)
plt.title("SPDE")
plt.imshow(X)
plt.subplot(1,2,2)
plt.title("Reference")
plt.imshow(X_ref)



### CHECK COVARIANCE

# plt.figure()
# err = RM.GRF.test_Covariance(nsamples=10000)
# RM.generate_samples(nsamples=100)


#######################################################

plt.show()