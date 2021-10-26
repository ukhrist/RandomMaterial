
import torch
import numpy as np
from tqdm import tqdm


"""
==================================================================================================================
Cumpute statistical desciptor from samples
==================================================================================================================
"""

def compute_from_samples(samples_generator, **kwargs):
    nsamples = kwargs.get('nsamples', 1000)
    if hasattr(samples_generator, 'sample_numpy'):
        Samples = ( samples_generator.sample_numpy() for isample in range(nsamples) )
    else:
        Samples = samples_generator

    descriptor = kwargs.get('descriptor', 'Covariance')
    if descriptor == 'Covariance':
        print("Computing covariance...")
        from .correlation import autocorrelation
        descriptor = autocorrelation
    elif descriptor == 'Curvature':
        print("Computing curvature...")
        from .curvature import spec_curvature
        descriptor = spec_curvature
    elif descriptor == 'SpecArea':
        if not (samples_generator.__class__.__name__ == 'TwoPhaseMaterial'):
            raise Exception('Speific area can be computed only for a two-phase media !')
        print("Computing specific area...")
        from .interface import spec_area
        descriptor = spec_area
        
    D = 0.
    for n, X in enumerate(tqdm(Samples, total=nsamples)):
        D += descriptor(X)
    D /= n+1

    ### Assume isotropy: compute the radial profile
    fg_iso = kwargs.get('iso', False)
    if fg_iso:
        nbins = np.amin(D.shape)
        D = radial_profile(D)[:nbins]

    return D



"""
==================================================================================================================
Cumpute radial profile of a desciptor
==================================================================================================================
"""

def radial_profile(X, center=None):
    if center is None: center = torch.zeros(X.ndim)
    
    with torch.no_grad():
        x = [torch.arange(n) for n in X.shape]
        x = torch.stack(torch.meshgrid(x), dim=-1)
        r = (x - center).norm(dim=-1)

    r = r.detach().numpy()
    if torch.is_tensor(X): X = X.detach().numpy()

    bins = 0.5+np.arange(np.amin(X.shape))
    radialprofile, bins = np.histogram(r, weights=X, bins=bins)
    norm, bins = np.histogram(r, bins=bins)
    radialprofile = radialprofile/norm
    radialprofile = np.insert(radialprofile, 0, X.flat[0])
    return radialprofile



"""
==================================================================================================================
Get frequensies tensor
==================================================================================================================
"""

def get_frequencies(X):    
    axes = [torch.arange(n) for n in X.shape]
    axes[-1] = torch.arange(int(np.ceil((X.shape[-1]+1)/2)))
    k = 1.0 * torch.stack(torch.meshgrid(axes), dim=0).detach()
    return k



"""
==================================================================================================================
Compute gradient of the field
==================================================================================================================
"""

def gradient(X, fft=False, diff=False, normalized=False, scheme='central'): ### 'central', 'backward', 'forward'

    if fft:
        k = get_frequencies(X)
        F = torch.fft.rfftn(X, s=[n for n in X.shape], norm='ortho')
        G = torch.zeros((X.dim(),) + X.shape)
        for j in range(X.dim()):
            G[j] = torch.fft.irfftn(1j*k[j] * F, s=[n for n in X.shape], norm='ortho')
        G *= 1/np.prod(X.shape)**(1/2)

    else:
        G = torch.zeros((X.dim(),) + X.shape)
        for j in range(X.dim()):
            if scheme=='central':
                G[j] = (X.roll(1, dims=j) - X.roll(-1, dims=j))
            elif scheme=='backward':
                G[j] = (X - X.roll(-1, dims=j))
            elif scheme=='forward':
                G[j] = (X.roll(1, dims=j) - X)
            if not diff:
                 G[j] = G[j] * X.shape[j]
                 if scheme=='central':
                    G[j] = G[j] /2

    if normalized:
        G = normalize(G)
        # normG = G.norm(dim=0)
        # interface = torch.where(normG>0.5, True, False).detach()
        # G = G.clone()
        # G[:,interface] = G[:,interface].clone()/normG[interface]

    return G


"""
==================================================================================================================
Normalize the vector field
==================================================================================================================
"""

def normalize(V):
    normV = V.norm(dim=0)
    interface = torch.where(normV>0.5, True, False).detach()
    NV = V.clone()
    NV[:,interface] = V[:,interface]/normV[interface]    
    return NV




"""
==================================================================================================================
Down/up-sampling of the field (interpolation)
==================================================================================================================
"""

def interpolate(X, new_shape, mode='linear'):
    Y = torch.nn.functional.interpolate(X.unsqueeze(0).unsqueeze(0), size=tuple(new_shape)).squeeze()
    return Y