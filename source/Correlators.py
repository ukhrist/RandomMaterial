
from math import pi
import torch.fft
import numpy as np
import scipy.fftpack as fft
from time import time

#######################################################################################################

def set_Correlator(obj, CorrelatorName):

    if CorrelatorName in ('torch', 'Torch', 'pytorch', 'pyTorch', 'PyTorch'):
        return Correlator_Torch(obj)

    elif CorrelatorName in ('fftw', 'FFTW'):
        return Correlator_FFTW(obj)

    else:
        raise Exception('Unknown sampling method "{0}".'.format(CorrelatorName))


#######################################################################################################
#	Abstract Correlator class
#######################################################################################################

class AbstractCorrelator:

    def __init__(self, RandomField):
        self.obj     = RandomField
        self.verbose = RandomField.verbose
        self.ndim    = RandomField.ndim
        self.L       = RandomField.Window.scale
        self.shape   = RandomField.Window.shape
        self.fg_Initialized = False

    def __call__(self, noise):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def check_init(self):
        if not self.fg_Initialized:
            self.initialize()
            self.fg_Initialized = True


#######################################################################################################
#	PyTorh FFT
#######################################################################################################
### - Only stationary covariance
### - Uses pytorch

class Correlator_Torch(AbstractCorrelator):

    def __init__(self, RandomField):
        super().__init__(RandomField)

    def initialize(self):
        freq = [None]*self.ndim
        for i in range(self.ndim):
            n, L = self.obj.Window.shape[i], self.obj.Window.scale[i]
            freq_gen = torch.fft.rfftfreq if i==self.ndim-1 else torch.fft.fftfreq
            freq[i]  = 2*pi*n*freq_gen(n, d=L)
        self.freq = torch.stack(list(torch.meshgrid(*freq, indexing="ij")), dim=-1).detach()
        self.KernelSpectrum = self.obj.Covariance.eval_spec(self.freq).sqrt()
        self.TransformScale = np.prod(self.obj.Window.shape/self.obj.Window.scale)**(1/2)

    def __call__(self, noise):
        self.check_init()
        if not torch.is_tensor(noise): noise = torch.tensor(noise).detach()
        spec  = self.KernelSpectrum * torch.fft.rfftn(noise, norm='ortho')
        field = torch.fft.irfftn(spec, norm='ortho')
        return field * self.TransformScale


#######################################################################################################
#	Fourier Transform (FFTW)
#######################################################################################################
### - Only stationary covariance
### - Uses the Fastest Fourier Transform on the West

class Correlator_FFTW(AbstractCorrelator):

    def __init__(self, RandomField):
        super().__init__(RandomField)

    def initialize(self):
        import pyfftw
        Window = self.obj.Window
        L, N, d = Window.scale[0], Window.shape[0], Window.ndim
        self.Frequences = (2*pi/L)*(N*fft.fftfreq(N))
        self.TransformNorm = L**(d/2)
        shpR = Window.shape
        shpC = shpR.copy()
        shpC[-1] = int(shpC[-1] // 2)+1
        axes = np.arange(self.ndim)
        flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED')
        self.fft_x     = pyfftw.empty_aligned(shpR, dtype='float64')
        self.fft_y 	   = pyfftw.empty_aligned(shpC, dtype='complex128')
        self.fft_plan  = pyfftw.FFTW(self.fft_x, self.fft_y, axes=axes, direction='FFTW_FORWARD',  flags=flags)
        self.ifft_plan = pyfftw.FFTW(self.fft_y, self.fft_x, axes=axes, direction='FFTW_BACKWARD', flags=flags)
        self.shpC = shpC
        self.Spectrum = self.obj.Covariance.precompute_Spectrum(self.Frequences)
        self.Spectrum_half = self.Spectrum[...,:self.shpC[-1]] * N**(d/2)

    def __call__(self, noise):
        self.check_init()
        self.fft_x[:] = noise
        self.fft_plan()
        self.fft_y[:] *= self.Spectrum_half 
        self.ifft_plan()
        return self.fft_x[self.DomainSlice] / self.TransformNorm



#######################################################################################################
#	Fourier Transform of Gaussian Noise
#######################################################################################################

def FourierOfGaussian(noise):
    a, b = noise, noise
    for j in range(noise.ndim):
        b = np.roll(np.flip(b, axis=j), 1, axis=j)
    n = int(np.ceil((noise.shape[-1]+1)/2))
    a, b = a[...,:n], b[...,:n]
    noise_hat = 0.5*( (a + b) + 1j*(a-b) ) 
    return noise_hat