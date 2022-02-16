
import numpy as np
import torch
from torch import nn
from torch.autograd import backward
from torch.nn.utils import parameters_to_vector

from .DistanceMeasure import DistanceMeasure
from .Discriminator import Discriminator

from ..StatisticalDescriptors import autocorrelation, spec_area, interface, correlation_curvature, curvature, num_curvature
from ..StatisticalDescriptors.common import gradient, interpolate

import matplotlib.pyplot as plt

"""
==================================================================================================================
Conditional Value-at-Risk loss function
==================================================================================================================
"""

class CVaR(nn.Module):

    def __init__(self, MaterialModel, Data, **kwargs):
        super(CVaR, self).__init__()
        self.Material = MaterialModel(**kwargs)
        self.dist     = kwargs.get('distance_measure', DistanceMeasure(Data=Data) )
        self.beta     = kwargs.get('beta', 0.9)
        self.quantile = nn.Parameter(torch.tensor([0.]))
        self.reg_coef = kwargs.get('regularization', 0.1)
        self.actfc    = nn.Softplus()
        # self.actfc    = nn.ReLU()

        self.dist.Material = self.Material
        self.fg_mean_only  = kwargs.get('mean_only', False)

        ### GAN
        self.GAN = kwargs.get('GAN', False)
        if self.GAN:
            self.D = Discriminator(int(self.Material.Window.shape.prod()))
            self.DataSample = torch.as_tensor(Data[0]).detach()




    def update(self):
        self.Material.correlate.initialize()
        if self.GAN:
            self.logD0 = (self.D(self.DataSample)).log()

    def sample(self, w, Data=None, Discriminator=False):
        beta, t = self.beta, self.quantile
        self.Material.seed(w)
        MaterialSample = self.Material.sample()
        # if MaterialSample.mean()
        # isempty = self.empty_sample_assert(MaterialSample)
        if self.GAN:
            # if Discriminator:
            #     for param in self.D.parameters():
            #         param.requires_grad = True
            #     G = MaterialSample.detach()
            #     out = self.logD0 + (1-self.D(G)).log()
            #     out = -1*out
            # else:
                # for param in self.D.parameters():
                #     param.requires_grad = False
            out = (1-self.D(MaterialSample)).log()
            # n = 2
            # # if n_iter<n:
            # #     # out = -1*self.logD0
            # #     out = self.logD0 + (1-self.D(MaterialSample.detach())).log() #.detach()
            # #     out = -1*out
            # # if n_iter % n == 0:
            # # if self.D(self.DataSample)<0.9:
            # if n_iter % n == 0: #in np.arange(n//2):
            #     for param in self.D.parameters():
            #         param.requires_grad = True
            #     out_d = self.logD0 + (1-self.D(MaterialSample.detach())).log() #.detach()
            #     out_d = -1*out_d
            #     out = out_d
            # else:
            #     for param in self.D.parameters():
            #         param.requires_grad = False
            #     out_g = (1-self.D(MaterialSample)).log()
            #     # out_g = -1*out_g
            #     out = 10*out_g
        else:
            rho = self.dist(MaterialSample, Data)
            if self.fg_mean_only: ### mean
                out = 0*t + rho
            else: ### CVaR itself
                out = t + self.actfc(rho - t) / (1-beta)

        ### Regularization
        # reg = 0.5 * parameters_to_vector([self.Material.Covariance.log_nu]).norm()**2
        # reg = 0.5 * parameters_to_vector([self.quantile]).norm()**2
        # reg = 0.5 * parameters_to_vector(self.parameters()).norm()**2
        # out = out + self.reg_coef * reg

        return out.squeeze()

    def val(self, Batch, Discriminator=False):
        val = torch.tensor(0.)
        for w in Batch:
            val += self.sample(w, Discriminator=Discriminator).item()
        val /= len(Batch)
        return val

    def forward(self, w, Discriminator=False):
        return self.sample(w, Discriminator=Discriminator)



    def empty_sample_assert(self, MaterialSample):
        if MaterialSample.mean().abs() < 1.e-3:
            print('EMPTY SAMPLE !')
            print('Mean = ', MaterialSample.mean())
            # print([p.data for p in self.Material.parameters()])
            print(self.Material.tau.data, self.Material.Covariance.nu.data, self.Material.Covariance.corrlen.data)
            return True
        else:
            return False



    def update_Discriminator(self, Batch):
        # for param in self.D.parameters():
        #     param.requires_grad = True

        optimizer = torch.optim.LBFGS(
            self.D.parameters(),
            lr=0.5,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-1,
            max_iter=4
            )
        # optimizer = torch.optim.SGD(self.D.parameters(), lr=1)

        # self.D.vf.data[:] = self.DataSample.mean().log()
        # self.D.sa.data[:] = spec_area(self.DataSample).log()

        def closure():
            self.update()
            loss = -self.D(self.DataSample).log() - self.val(Batch, Discriminator=True)
            print('loss = ', loss.item())
            # print('prms = ', self.D.vf.exp().item(), self.D.sa.exp().item())
            # print('vf, sa = ', self.DataSample.mean().item(), spec_area(self.DataSample).item())
            return loss

        k_max = 1
        for k in range(k_max):
            optimizer.zero_grad()
            loss = closure()
            loss.backward()
            optimizer.step(closure)
            # if self.D(self.DataSample) > 0.99: break

        
        print('     D(G) = ', self.D(self.Material.sample()).item())
        print('     D(Y) = ', self.D(self.DataSample).item())
        print()


