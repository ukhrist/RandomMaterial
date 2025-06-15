
from math import pi
import numpy as np
from torch.autograd import backward

from torch.nn.functional import softplus
import torch
from torch import nn
from torch import random

from ..StatisticalDescriptors import autocorrelation, spec_area, interface, correlation_curvature, curvature, num_curvature
from ..StatisticalDescriptors.common import gradient, interpolate

import matplotlib.pyplot as plt




"""
==================================================================================================================
Distance measure between a sample and the data
==================================================================================================================
"""

class DistanceMeasure(nn.Module):

    def __init__(self, Data=None, **kwargs):
        super(DistanceMeasure, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.relu     = torch.nn.ReLU()

        if Data is not None:
            self.Data = list(Data)
            self.preprocess_data(self.Data)
            # print('Data descriptors :', [sd.data.view(-1) for sd in self.data_descriptors[:-1]])
            # exit()

        self.obj = kwargs.get('Model', None)
        if self.obj is not None:
            self.ndim = self.obj.ndim
            freq = [None]*self.ndim
            for i in range(self.ndim):
                n, L = self.obj.Window.shape[i], self.obj.Window.scale[i]
                freq_gen = torch.fft.rfftfreq if i==self.ndim-1 else torch.fft.fftfreq
                freq[i]  = 2*pi*n*freq_gen(n, d=L)
            self.freq = torch.stack(list(torch.meshgrid(*freq, indexing="ij")), dim=-1).detach()
        

    def forward(self, Sample, Data=None):
        if Data is not None:
            target = self.preprocess_data(Data)
        else:
            target = self.data_descriptors

        if Sample.shape != self.data_sample_shape:
            self.downsample_data(Sample.shape)

        predict = self.statistial_descriptors(Sample)
        # self.print(target,  label='Target')
        # self.print(predict, label='Predict')

        # x = np.arange(Sample.shape[0])
        # plt.subplot(1,2,1)
        # plt.plot(target[-1][0,x,x].detach())
        # plt.plot(predict[-1][0,x,x].detach())
        # plt.subplot(1,2,2)
        # plt.plot(target[-2][0,0,:].detach())
        # plt.plot(predict[-2][0,0,:].detach())

        # # plt.subplot(2,2,1)
        # # plt.plot(target[-4][0,0,:].detach())
        # # plt.plot(predict[-4][0,0,:].detach())
        # # plt.subplot(2,2,2)
        # # plt.plot(target[-1][0,0,:].detach())
        # # plt.plot(predict[-1][0,0,:].detach())
        # # plt.subplot(2,2,3)
        # # plt.plot(target[-3][0,0,:].detach())
        # # plt.plot(predict[-3][0,0,:].detach())
        # # plt.subplot(2,2,4)
        # # plt.plot(target[-2][0,0,:].detach())
        # # plt.plot(predict[-2][0,0,:].detach())
        # plt.show()

        self.misfit = torch.zeros(self.nDescriptors)
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        # weights[-1] = 10
        for i in range(self.nDescriptors):
            self.misfit[i] = weights[i] * self.compute_misfit(predict[i],target[i], i)

        return self.misfit.sum()

    def compute_misfit(self, predict, target, i=None):
        return (predict-target).square().mean() / (target).square().mean()
        # if predict.dim() > 1:
        #     return (predict-target).square().mean() / (target).square().mean()
        #     # return torch.log((predict/target).abs()).square().sum()
        # else:
        #     return (predict-target).square().mean() / (target).square().mean()
        #     # return torch.log((predict/target).abs()).square().mean() / torch.log((target).abs()).square().mean()
            

    def preprocess_data(self, Data):
        self.nDescriptors = None
        nData = 0
        for Sample in Data:
            nData += 1
            sd_Sample = self.statistial_descriptors(Sample)
            if self.nDescriptors is None:
                self.nDescriptors = len(sd_Sample)
                sd_list = [ torch.zeros_like(sd_Sample[i]).detach() for i in range(self.nDescriptors) ]
            for i in range(self.nDescriptors):
                sd_list[i] += sd_Sample[i]
        self.data_descriptors  = [ sd.detach()/nData for sd in sd_list]
        self.data_sample_shape = Sample.shape
        return self.data_descriptors

    def downsample_data(self, shape):
        for i, sd in enumerate(self.data_descriptors):
            if sd.dim() > 1:
                self.data_descriptors[i] = interpolate(sd, shape)

    def statistial_descriptors(self, Sample):
        l = 20

        ### Volume fraction and specific surface area
        vf = Sample.mean()
        sa = spec_area(Sample ) #* Sample.numel()
        # sphericity = (36*pi)**(1/3) / sa

        ### Two-point correlation 
        S2 = autocorrelation(Sample - Sample.mean()) #[:l,:l,:l]
        # S2 = self.softplus(S2)
        # H1 = torch.zeros([3])
        # H2 = torch.zeros([3])
        # H1 = torch.tensor([ S2[1,0,0]-S2[0,0,0], S2[0,1,0]-S2[0,0,0], S2[0,0,1]-S2[0,0,0] ]).mean()
        # H2 = torch.tensor([ S2[2,0,0], S2[0,2,0], S2[0,0,2] ]).mean()

        # d1S2 = autocorrelation(Sample-vf, deriv=1) #/ vf
        # d2S2 = autocorrelation(Sample-vf, deriv=2) #/ vf
        # H = d2S2[0,0,0]

        # H2 = torch.tensor([ d1S2[1,0,0], d1S2[0,1,0], d1S2[0,0,1] ]).mean()

        ### Curvature
        # H, K, g = num_curvature(Sample)
        # # Chi = K[K!=0.].mean() / (2*pi)
        # Havg = H[H!=0.].abs().mean()
        # Kavg = K[K!=0.].abs().mean()
        # # Havg = H.square().mean()
        # # Kavg = K.square().mean()
        # # H2 = autocorrelation(H)
        # # K2 = autocorrelation(K)

        ### Interface correlation
        # I  = interface(Sample)
        # J2 = autocorrelation(I-sa) #[:l,:l,:l]
        # # J2 = J2/sa #[:l,:l]
        # J2 = self.relu(J2)
        # d1J2 = autocorrelation(I-sa, deriv=1) #/ vf
        # d2J2 = autocorrelation(I-sa, deriv=2) #/ vf
        I  = interface(Sample)
        I2 = autocorrelation(I - I.mean()) #[:l,:l,:l]
        # G  = gradient(Sample, fft=True).norm(dim=0)
        # GG = autocorrelation(G)
        # G  = gradient(Sample, fft=True) #), diff=True, scheme='backward') #.norm(dim=0)
        # G1 = autocorrelation(G[0]) # - sa) / sa #[:l,:l,:l] 
        # G2 = autocorrelation(G[1]) # - sa) / sa #[:l,:l,:l] 
        # G3 = autocorrelation(G[2]) # - sa) / sa #[:l,:l,:l] 
        # G21 = autocorrelation(G[0])
        # G22 = autocorrelation(G[1])
        # G23 = autocorrelation(G[2])


        # H = gradient(G, diff=True).norm(dim=0).mean()

        # axes = [torch.arange(n)/n for n in Sample.shape]
        # x = 1.0 * torch.stack(torch.meshgrid(axes), dim=0).detach()
        # r = x.norm(dim=0)
        # w = torch.exp(-10*r)
        # w = 1

        

        ### Vector of statistical descriptors
        sd = []
        sd.append( vf )
        sd.append( sa )
        # sd.append( H )
        # sd.append( H1  )
        # sd.append( H2  )
        # sd.append( sphericity )
        # sd.append( L )
        # sd.append( C )
        # sd.append( Havg )
        # sd.append( Kavg )
        # sd.append( Chi )
        #===============
        sd.append( S2 )
        # sd.append( d1S2 )
        # sd.append( d2S2 )
        sd.append( I2 )
        # sd.append( GG )
        # sd.append( G1 )
        # sd.append( G2 )
        # sd.append( G3 )
        # sd.append( G21 )
        # sd.append( G22 )
        # sd.append( G23 )
        # sd.append( J2 )
        # sd.append( d1J2 )
        # sd.append( d2J2 )
        # sd.append( H2 )
        # sd.append( K2 )
        return sd

    
    def print(self, SDs, label='Statistical descriptors'):
        ls_print = []
        for sd in SDs:
            if sd.dim() < 1:
                ls_print.append(sd.item())
                # ls_print.append(sd.data)
        print(label + ': ', ls_print)
