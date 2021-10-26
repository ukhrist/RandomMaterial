
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from .DistanceMeasure import DistanceMeasure

"""
==================================================================================================================
Conditional Value-at-Risk loss function
==================================================================================================================
"""

class CVaR(nn.Module):

    def __init__(self, Model, Data, **kwargs):
        super(CVaR, self).__init__()
        self.Model    = Model(**kwargs)
        self.dist     = kwargs.get('distance_measure', DistanceMeasure(Data=Data) )
        self.beta     = kwargs.get('beta', 0.9)
        self.quantile = nn.Parameter(torch.tensor(1.))
        self.actfc    = nn.Softplus()
        # self.actfc    = nn.ReLU()
        self.batch_size_bound = None

    def forward(self, Batch, Data=None):
        self.Model.correlate.initialize()
        beta, t = self.beta, self.quantile
        self.StoredTerms = []
        self.BatchSize   = len(Batch)
        CVaR = torch.tensor(0.)
        for w in Batch:
            self.Model.seed(w)
            Sample = self.Model.sample()
            # print('\n s = {}'.format(spec_area(Sample))+'\n')
            rho    = self.dist(Sample, Data)
            # print('\n rho = {}'.format(rho)+'\n')
            # CVaR   = CVaR + self.actfc(rho - t) / (1-beta)
            CVaR  = CVaR + rho
            self.StoredTerms.append(rho) 
        CVaR = CVaR / len(Batch) + t*0
        print('\n CVaR = {}\n'.format(CVaR))
        self.batch_size_bound = self.compute_batch_size_bound()
        return CVaR

    def compute_batch_size_bound(self):
        if hasattr(self, 'StoredTerms'):
            list_gi = [ torch.autograd.grad(Fi, self.Model.parameters(), allow_unused=True, retain_graph=True) for Fi in self.StoredTerms ]
            list_gi = [torch.cat([gij.view(-1) for gij in gi]) for gi in list_gi]
            g = torch.mean(torch.stack(list_gi, dim=0), dim=0)
            g2 = g.square().sum(dim=-1)
            Var = torch.tensor(0.)
            for gi in list_gi:
                Var = Var + (torch.inner(gi, g) - g2).square()
            Var = Var / (self.BatchSize-1)
            b = Var / g2**2 if g2 else 0
            return b / 10
        else:
            return 0
