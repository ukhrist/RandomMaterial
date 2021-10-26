
import torch
from torch import nn
import numpy as np
from contextlib import suppress

from torch.nn.utils.convert_parameters import parameters_to_vector

from .LossFunctions import CVaR

class calibrate():

    def __init__(self, Model, Data, **kwargs):
        self(Model, Data, **kwargs)

    def __call__(self, Model, Data, **kwargs):
        self.verbose = kwargs.get('verbose', True)

        nepochs = kwargs.get('nepochs', 100)
        lr      = kwargs.get('lr',  1e-1)
        tol     = kwargs.get('tol', 1e-3)

        self.fg_adapt = kwargs.get('adapt', False)

        self.LossFunc = kwargs.get('Loss', CVaR ) ### 'CVaR', 'Mean'
        self.LossFunc = self.LossFunc(Model, Data, **kwargs)

        self.Optimizer = kwargs.get('Optimizer', torch.optim.LBFGS)
        self.Optimizer = self.Optimizer([self.LossFunc.quantile, self.LossFunc.Model.Covariance.log_nu], lr=lr)#, line_search_fn='strong_wolfe')
        # self.Optimizer = self.Optimizer([self.LossFunc.Model.Covariance.nu], lr=lr)
        # self.Optimizer = self.Optimizer([{'params': self.LossFunc.quantile, 'lr': 0.5},
        #                                  {'params': self.LossFunc.Model.Covariance.log_nu, 'lr': 100},
        #                                 #  {'params': self.LossFunc.Model.par_tau,   'lr': 0.5},
        #                                  {'params': self.LossFunc.Model.par_alpha, 'lr': 0.5},
        #                                 #  {'params': self.LossFunc.Model.Structure.par_radius, 'lr': 0.5} ],
        #                                  {'params': self.LossFunc.Model.Covariance.log_corrlen, 'lr': 0.5} ],
        #                                  lr=lr)
        # self.Optimizer = self.Optimizer(self.LossFunc.parameters(), lr=lr, line_search_fn='strong_wolfe')
        # self.Optimizer = self.Optimizer([self.LossFunc.quantile, self.LossFunc.Model.Covariance.log_corrlen], lr=lr)

        self.batch_size    = kwargs.get('init_batch_size', 1)
        self.batch_overlap = kwargs.get('batch_overlap', 0)

        def closure():
            self.Optimizer.zero_grad()
            self.Batch = self.adapt_Batch()
            p = parameters_to_vector(self.LossFunc.parameters())
            self.loss  = self.LossFunc(self.Batch) #+ 1.e0 * p.square().mean()
            self.loss.backward()
            self.line_search()
            if self.verbose:
                print('loss = ', self.loss.item())
                self.print_grad()
                self.print_parameters()
                # self.LossFunc.Model.plot()
            return self.loss

        nepochs = 1
        for epoch in range(nepochs):
            if self.verbose:
                print()
                print('=================================')
                print('-> Epoch {0:d}'.format(epoch))
                print('=================================')
            self.Optimizer.step(closure)
            if self.verbose:
                self.print_grad()
                self.print_parameters()
                print()
            if self.loss.item() < tol: break

        if self.verbose:
            print()
            print('=================================')
            print('Calibration terminated.')
            print('=================================')
            print('loss = {0}'.format(self.loss.item()))
            print('tol  = {0}'.format(tol))
            print()
            self.print_parameters()

        return 0

    def get_Batch(self, BatchSize):
        Batch = self.draw(BatchSize)
        return Batch

    def draw(self, size):
        return torch.randint(1000, (int(size),))

    def adapt_Batch(self):
        Batch = self.draw(self.batch_size)
        if self.fg_adapt:
            #TODO: Batch overlap
            bound = self.compute_batch_size_bound(Batch)
            if bound:
                while len(Batch) < bound:
                    Batch_plus = self.draw(np.ceil(bound)-len(Batch))
                    Batch      = torch.cat([Batch, Batch_plus]).unique()
                self.batch_size = len(Batch)
        return Batch

    def compute_batch_size_bound(self, Batch):
        # return len(Batch)
        b = self.LossFunc.batch_size_bound
        print('S = ', self.batch_size)
        return b
    

    def line_search(self):
        pass



    def print_parameters(self):
        print('---------------------------------')
        with suppress(AttributeError): print('{0:7s}  |  {1: 2.3f}  |  {2: 3.4f}  |'.format('tau', self.LossFunc.Model.tau.item(), self.LossFunc.Model.par_tau.grad.view(-1).item()) )
        with suppress(AttributeError): print('{0:7s}  |  {1: 2.3f}  |  {2: 3.4f}  |'.format('alpha', self.LossFunc.Model.alpha.item(), self.LossFunc.Model.par_alpha.grad.view(-1).item()) )
        with suppress(AttributeError): print('{0:7s}  |  {1: 2.3f}  |  {2: 3.4f}  |'.format('radius', self.LossFunc.Model.Structure.radius.item(), self.LossFunc.Model.Structure.par_radius.grad.view(-1).item()) )
        with suppress(AttributeError): print('{0:7s}  |  {1: 2.3f}  |  {2: 3.4f}  |'.format('nu', self.LossFunc.Model.Covariance.nu.item(), self.LossFunc.Model.Covariance.log_nu.grad.view(-1).item()) )
        with suppress(AttributeError): print('{0:7s}  |  {1: 2.3f}  |  {2: 3.4f}  |'.format('t', self.LossFunc.quantile.item(), self.LossFunc.quantile.grad.view(-1).item()) )

        with suppress(AttributeError): print('vf = ', self.LossFunc.Model.vf)
        # with suppress(AttributeError): print('tau  = ', self.LossFunc.Model.tau.item())
        # with suppress(AttributeError): print('alpha  = ', self.LossFunc.Model.alpha.item())
        # with suppress(AttributeError): print('R  = ', self.LossFunc.Model.Structure.radius.item())
        # print('nu = ', self.LossFunc.Model.Covariance.nu.item())
        print('l  = ', self.LossFunc.Model.Covariance.corrlen.data)
        # print('t  = ', self.LossFunc.quantile.item())
        print('---------------------------------')

    def print_grad(self):
        self.grad = torch.cat([ param.grad.view(-1) for param in self.LossFunc.parameters() ]).detach().numpy()
        print('---------------------------------')
        print('grad = ', self.grad)
        print('---------------------------------')
