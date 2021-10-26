
from source.RMNet.StatisticalDescriptors import compute_from_samples
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
==================================================================================================================
Test the marginal moments of the field
==================================================================================================================
"""

def test_MarginalMoments(self, **kwargs):
    print()
    print('==================================================================================================================')
    print('   Test the marginal moments of the field')
    print('==================================================================================================================\n')
    nsamples = kwargs.get('nsamples', 1000)
    nmoments = kwargs.get('moments', 2)
    M = [0.]*nmoments
    for isample in tqdm(range(nsamples)):
        X = self.sample()
        Xn= 1.
        for n in range(nmoments):
            Xn = Xn * X
            M[n] += Xn.mean()
    for n in range(nmoments):
        M[n] /= nsamples
        print('Moment {0:d} = {1:f}'.format(n+1, M[n]))
    return M



"""
==================================================================================================================
Test the covariance function
==================================================================================================================
"""

def test_Correlation(self, **kwargs):
    print()
    print('==================================================================================================================')
    print('   Test the correlation function')
    print('==================================================================================================================\n')
    
    nsamples = kwargs.get('nsamples', 1000)
    samples_generator = ( self.sample_numpy() for isample in range(nsamples) )
    Cov = compute_from_samples(samples_generator, nsamples=nsamples, iso=True)
    nbins_tot = Cov.size
    nbins     = nbins_tot // 2
    Cov       = Cov[:nbins]

    r = np.arange(nbins)/nbins_tot
    Cov_ref = self.S2(r)
    if torch.is_tensor(Cov_ref): Cov_ref = Cov_ref.detach().numpy()

    error = np.linalg.norm(Cov-Cov_ref, np.inf)
    print("Correlation error = ", error)

    plt.figure()
    plt.plot(r, Cov, "b-")
    plt.plot(r, Cov_ref, "r--")
    plt.legend(["Sampled", "Analythic"])
    plt.show()

    return error




# """
# ==================================================================================================================
# Test Gaussian curvature of the field
# ==================================================================================================================
# """

# def test_Curvature(self, **kwargs):
#     print()
#     print('==================================================================================================================')
#     print('   Test Gaussian curvature of the field')
#     print('==================================================================================================================\n')

#     nsamples = kwargs.get('nsamples', 1000)
#     samples_generator = ( self.sample_numpy() for isample in range(nsamples) )
#     K = compute_from_samples(samples_generator, nsamples=nsamples, descriptor='Curvature')
#     return K



# """
# ==================================================================================================================
# Test specific area
# ==================================================================================================================
# """

# def test_SpecificArea(self, **kwargs):
#     print()
#     print('==================================================================================================================')
#     print('   Test specific area of the field')
#     print('==================================================================================================================\n')
    
#     nsamples = kwargs.get('nsamples', 1000)
#     samples_generator = ( self.sample_numpy() for isample in range(nsamples) )
#     s = compute_from_samples(samples_generator, nsamples=nsamples, descriptor='SpecArea')
#     return s