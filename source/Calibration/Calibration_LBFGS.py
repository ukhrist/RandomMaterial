
from operator import ne
from numpy.core.numeric import indices
import torch
from torch import is_tensor, nn
import numpy as np
import pandas as pd
from contextlib import suppress
from torch._C import dtype
from torch.autograd import grad
from torch.nn import parameter
from torch.nn.functional import batch_norm
from torch.nn.utils.convert_parameters import parameters_to_vector
# import intel_pytorch_extension as ipex

from .CVaR import CVaR

import sys
sys.path.append('/home/khristen/ThirdPartyCode/PyTorch-LBFGS/functions/')

from LBFGS import LBFGS


class calibrate():

    def __init__(self, MaterialModel, Data, **kwargs):
        self(MaterialModel, Data, **kwargs)

    def initialize(self):
        return 0


    def __call__(self, MaterialModel, Data, **kwargs):
        self.verbose = kwargs.get('verbose', True)
        self.debug   = kwargs.get('debug', False)

        ### Parameters for L-BFGS training
        max_iter = kwargs.get('max_iter', 300)  ### note each iteration is NOT an epoch
        lr       = kwargs.get('lr',  1)
        tol      = kwargs.get('tol', 1e-3)
        self.SGD = kwargs.get('SGD', False)


        self.batch_size    = kwargs.get('init_batch_size', 1)
        self.overlap_ratio = kwargs.get('overlap_ratio', 0.25)         ### should be in (0, 0.5)
        self.overlap_size  = int(self.overlap_ratio * self.batch_size)

        line_search        = kwargs.get('line_search', 'Armijo')
        Powell_dumping     = kwargs.get('Powell_dumping', False)
        history_size       = kwargs.get('history_size', 10)
        curvature_eps      = kwargs.get('curvature_eps', 0.2)

        output_folder = kwargs.get('output_folder', './calibration/')
        self.output_folder = output_folder
        
        ### Check cuda availability
        cuda = torch.cuda.is_available()
            
        ### Create neural network model
        self.Model = kwargs.get('Loss', CVaR ) ### 'CVaR', 'Mean'
        if cuda:
            torch.cuda.manual_seed(2021)
            self.Model = self.Model(MaterialModel, Data, **kwargs, calibration_regime=True).cuda()
        else:
            torch.manual_seed(2021)
            self.Model = self.Model(MaterialModel, Data, **kwargs, calibration_regime=True)


        ### Define optimizer
        not_require_grad = kwargs.get('not_require_grad', [])
        self.set_not_require_grad(not_require_grad)
        params  = self.Model.Material.parameters()
        self.Optimizer = LBFGS(params, lr=lr, history_size=history_size, line_search=line_search, debug=self.debug)

        nparams = len([p for p in self.Model.Material.parameters()])

        def closure():
            self.Optimizer.zero_grad()
            self.Model.update()
            return self.Model.val(self.Batch)

        def get_grad(Batch):
            vals, grads = [], []
            for w in Batch:
                self.Optimizer.zero_grad()
                self.Model.update()
                val = self.Model.forward(w)
                val.backward()
                grad = self.Optimizer._gather_flat_grad()
                grads.append(grad)
                vals.append(val)
            return grads, vals

        # def func(theta):
        #     self.Model
        #     self.Model.forward(w)

        # def get_hess(Batch):
        #     hess = []
        #     for w in Batch:
        #         self.Model.update()
        #         H = torch.autograd.functional.hessian(self.Model.forward, w)
        #         hess.append(H)
        #     return hess

        Batch_Ok_prev = np.array([], dtype=np.int)
        grads_Ok_prev, vals_Ok_prev = [], []
        self.g_hist = []

        ### store initial state
        # self.Batch  = self.draw(self.batch_size)
        # grads, vals = get_grad(self.Batch)
        # self.loss   = torch.stack(vals).mean(dim=0)
        # self.g_Sk   = torch.stack(grads).mean(dim=0)
        # self.crit   = self.g_Sk.norm(p=np.infty)
        self.store_iteration_results()

        ### MAIN LOOP
        for n_iter in range(max_iter):
            self.n_iter = n_iter

            self.print_parameters()
            
            Batch = Batch_Ok_prev
            Batch, Batch_plus = self.extend_batch(Batch, self.batch_size)

            if self.Model.GAN: self.Model.update_Discriminator(Batch)

            grads, vals = get_grad(Batch_plus)
            # hess = get_hess(Batch_plus)

            grads.extend(grads_Ok_prev)
            vals.extend(vals_Ok_prev)

            if self.update_batch_size(grads):

                Batch, Batch_plus = self.extend_batch(Batch, self.batch_size)
            
                grads_plus, vals_plus = get_grad(Batch_plus)

                grads.extend(grads_plus)
                vals.extend(vals_plus)


            indices = np.arange(self.batch_size)
            Ok = indices[-self.overlap_size:]

            grads = torch.stack(grads)            
            g_Ok  = grads[Ok].mean(dim=0)
            g_Sk  = grads.mean(dim=0)
            loss  = torch.stack(vals).mean(dim=0)
        
            ### two-loop recursion to compute search direction
            if self.SGD:
                lr = 0.1
                p = -lr*g_Sk
            else:
                p = self.Optimizer.two_loop_recursion(-g_Sk)                  
                alpha_init = self.init_line_search(grads)
                p = alpha_init * p


            if not self.Model.GAN:
                print('g_Sk    = ', g_Sk.data.tolist())
                print('Hg_Sk   = ', p.data.tolist())
                print('Misfits = ', self.Model.dist.misfit.data.tolist())
    
            ### perform line search step
            self.Batch = Batch
            options = { 'closure'       : closure, 
                        'current_loss'  : loss,
                        'max_ls'        : 10,
                        'inplace'       : True,
                        'interpolate'   : True,
                        'ls_debug'      : self.debug,
                    }
            result  = self.Optimizer.step(p, g_Ok, g_Sk=g_Sk, options=options)
            if line_search == 'None':
                lr = result
            elif line_search == 'Armijo':
                loss, lr, *other = result
            elif line_search == 'Wolfe':
                loss, _, lr, *other = result
            else:
                raise Exception('Unsupported line search method.')
            inc = lr*p

            ### criterion
            # crit = (g_Sk*parameters_to_vector(self.Model.parameters()).exp()).norm(p=np.infty)
            # crit = inc.norm(p=np.infty) #/loss.item()
            crit = torch.inner(g_Sk, inc).abs() #/loss.item()
            if self.Model.GAN:
                crit = g_Sk.norm(p=np.infty) #/loss.item()
                # print('     D(G) = ', self.Model.D(self.Model.Material.sample()).item())
                # print('     D(Y) = ', self.Model.D(self.Model.DataSample).item())
            self.crit = crit

            ### compute gradient
            if line_search != 'Armijo': loss = closure()
            loss.backward()
            g_Sk = self.Optimizer._gather_flat_grad()
            self.g_Sk = g_Sk
            self.loss = loss
            self.g_hist.append(g_Sk)

            ### compute previous overlap gradient for next sample
            # Batch_Ok_prev = Batch[Ok]
            # grads_Ok_prev, vals_Ok_prev = get_grad(Batch_Ok_prev)
            # g_Ok_prev = torch.stack(grads_Ok_prev).mean(dim=0)

            Batch_Ok_prev = np.array([], dtype=np.int)
            grads_Ok_prev, vals_Ok_prev = [], []

            
            ### curvature update
            # self.Optimizer.curvature_update(g_Ok_prev, eps=curvature_eps, damping=Powell_dumping)
            self.Optimizer.curvature_update(g_Sk, eps=curvature_eps, damping=Powell_dumping)


            

            ### print data
            if self.verbose:               
                print()
                print('=========================================================================')
                print('Iter:', n_iter + 1, 'lr:', lr, 'Loss:', loss.item(), 'Grad:', g_Sk.norm(p=np.infty).item(), 'Inc:', inc.norm(p=np.infty).item(), 'Crit:', crit.item())
                print('=========================================================================')
                print()

                if self.Model.Material.ndim == 3:
                    # self.Model.Material.calibration_regime = False
                    self.Model.Material.save_vtk(output_folder+'current_sample', seed=0)
                    self.Model.Material.calibration_regime = True
                    self.Model.Material.export_parameters(output_folder + 'inferred_parameters')

            ### store current state 
            self.store_iteration_results()
               
            ### stop criterion
            if crit.item() < tol: break
            # if loss < tol: break
            # if lr < tol: break
            # if g_Sk.norm() < tol: break


        ### print final data
        if self.verbose:
            print()
            print('=================================')
            print('Calibration terminated.')
            print('=================================')
            print('loss = {0}'.format(loss.item()))
            print('grad = {0}'.format(g_Sk.norm()))
            print('Hg = {0}'.format(p.norm()))
            print('lr = {0}'.format(lr))
            print('tol  = {0}'.format(tol))
            print()
            self.print_parameters()

        return 0

    ##############################################################

    def draw(self, size):
        # return torch.randint(1000, (int(size),))
        return np.random.randint(1000, size=size)

    def extend_batch(self, Batch, size):
        size_ini = len(Batch)
        while len(Batch) < size:
            Batch_plus = self.draw(size-len(Batch))
            Batch      = np.append(Batch,Batch_plus)
            Batch      = pd.unique(Batch)
        Batch_plus = Batch[size_ini:]
        return Batch, Batch_plus

    def update_batch_size(self, grads):
        if self.Model.GAN:
            return False

        S  = len(grads)
        gs = grads if torch.is_tensor(grads) else torch.stack(grads, dim=0)
        g  = gs.mean(dim=0)
        if self.SGD:
            Hg = g
            HHg= g
        else:
            Hg = self.Optimizer.two_loop_recursion(1.*g)
            HHg= self.Optimizer.two_loop_recursion(1.*Hg)
        
        Var = torch.tensor(0.)
        for gi in gs:
            Var += (torch.inner(gi, HHg) - Hg.norm()**2).square()
        Var = Var / Hg.norm()**4 / (S-1)


        # batch_size = len(grads)
        # gs = torch.stack(grads, dim=0)
        # g  = gs.mean(dim=0)
        # Hg = self.Optimizer.two_loop_recursion(-g)
        # HHg= self.Optimizer.two_loop_recursion(-Hg/Hg.norm())
        # Hg2= Hg.square().sum()
        # Var = torch.tensor(0.)
        # for gi in gs:
        #     # Var += (torch.inner(gi, HHg) - Hg2).square()
        #     # Var += torch.inner(gi-g, HHg).square()
        #     Var += (torch.inner(gi/Hg.norm(), HHg) - 1).square()
        # Var /= (batch_size-1)
        # # bound = Var / Hg2.square() if Hg2 else 0


        kappa = 1
        bound = Var
        bound /= kappa**2


        # Var2 = torch.tensor(0.)
        # for gi in gs:
        #     Hgi  = self.Optimizer.two_loop_recursion(1.*gi)   
        #     Var2 += (Hgi - Hg).square().sum()
        # Var2 = Var2 / (S-1) / Hg.norm()**2

        # kappa2 = 2
        # bound2 = Var2
        # bound2 /= kappa2**2

        print('Std of g: ', (gs-g).norm().item())




        print('S, b = {}, {}'.format(self.batch_size, bound.item()))
        # print('S, b = {}, {}, {}'.format(self.batch_size, bound.item(), bound2.item()))

        if bound > 2*self.batch_size:
        # if bound > 10:
            # raise Exception('Batch size increment is too large !')
            print('\n\nBatch size increment is too large !\n\n')
            # exit()
            return False

        if self.batch_size < bound:
            self.batch_size_prev = self.batch_size
            self.batch_size      = int(np.ceil(bound))
            self.overlap_size    = int(self.overlap_ratio * self.batch_size)
            return True
        # elif self.sample_control(g, gs):
        #     return True
        else:
            return False

        # if self.batch_size < bound2:
        #     self.batch_size_prev = self.batch_size
        #     self.batch_size      = int(np.ceil(bound2))
        #     self.overlap_size    = int(self.overlap_ratio * self.batch_size)
        #     return True
        # else:
        #     return False


    def init_line_search(self, grads):
        S  = len(grads)
        gs = grads if torch.is_tensor(grads) else torch.stack(grads, dim=0)
        g  = gs.mean(dim=0)
        g2 = g.square().sum()
        Var= (gs-g).square().sum() / (S-1)
        a  = 1/(1 + Var / (S * g2))
        return a

        # S  = len(grads)
        # gs = grads if torch.is_tensor(grads) else torch.stack(grads, dim=0)
        # g  = gs.mean(dim=0)
        # Hg  = self.Optimizer.two_loop_recursion(1.*g)        
        # Var = torch.tensor(0.)
        # for gi in gs:
        #     Hgi  = self.Optimizer.two_loop_recursion(1.*gi)   
        #     Var += (Hgi - Hg).square().sum()
        # Var = Var / (S-1)
        # a   = 1/(1 + Var / (S * Hg.norm()**2))
        # return a

    def sample_control(self, g, gs):
        # if g is None: g=self.g_Sk
        r_max, w = 4, 10
        r = len(self.g_hist)
        if r<r_max:
            return False
        if r>r_max:
            self.g_hist = self.g_hist[-10:]
            r = r_max
        g_avg = torch.stack(self.g_hist).mean(dim=0)
        gamma = r**(-1/2) + (1-r**(-1/2)) * 1/w
        if g_avg.norm() < gamma*g.norm():

            kappa1, kappa2 = 1, 1

            S = len(gs)

            Var = torch.tensor(0.)
            for gi in gs:
                Var += torch.inner(gi-g, g_avg).square()
            Var = Var / (S-1)

            S1 = Var / g_avg.norm()**4 
            S1 = S1 / kappa1**2


            Var = torch.tensor(0.)
            for gi in gs:
                e = g_avg / g_avg.norm()
                p = torch.inner(gi-g, e)
                Var += ((gi-g) - p * e).norm()**2
            Var = Var / (S-1)

            S2 = Var / g_avg.norm()**2 
            S2 = S2 / kappa2**2

            new_batch_size = np.maximum(S1, S2)
            new_batch_size = np.maximum(new_batch_size, self.batch_size)

            self.batch_size_prev = self.batch_size
            self.batch_size      = int(np.ceil(new_batch_size))
            self.overlap_size    = int(self.overlap_ratio * self.batch_size)

            print('Batch size = ', new_batch_size)

            return True
        else:
            return False


    ##############################################################

    def print_parameters(self):
        print('---------------------------------')
        for name, p in self.Model.named_parameters():
            if p.data.ndim<2 and len(p.data)<4:
                print('{0:33s}  |  {1:5s}  |  {2}  '.format(name, str(p.requires_grad), p.data.tolist()))

    def print_grad(self):
        pass

    ##############################################################
    ### store the iteration state
    def store_iteration_results(self):

        if not hasattr(self, "stored_batch_size"):              self.stored_batch_size = []
        if not hasattr(self, "stored_grad_norm"):               self.stored_grad_norm = []
        if not hasattr(self, "stored_loss"):                    self.stored_loss = []
        if not hasattr(self, "stored_parameters_quantile"):     self.stored_parameters_quantile = []
        if not hasattr(self, "stored_parameters_thickness"):    self.stored_parameters_thickness = []
        if not hasattr(self, "stored_parameters_alpha"):        self.stored_parameters_alpha = []
        if not hasattr(self, "stored_parameters_nu"):           self.stored_parameters_nu = []
        if not hasattr(self, "stored_parameters_corrlen"):      self.stored_parameters_corrlen = []
        if not hasattr(self, "stored_parameters_noise_q"):      self.stored_parameters_noise_q = []

        ### 1) Batch size
        self.stored_batch_size.append(self.batch_size)
        save_as_csv(self.stored_batch_size, self.output_folder + 'stored_batch_size')

        ### 2) Grad norm (convergence), MSE
        if hasattr(self, "g_Sk"):
            self.stored_grad_norm.append([self.g_Sk.norm(p=np.infty).item(), self.crit.item()])
            save_as_csv(self.stored_grad_norm, self.output_folder + 'stored_grad_norm')
        if hasattr(self, "loss"):
            self.stored_loss.append(self.loss.item())
            save_as_csv(self.stored_loss, self.output_folder + 'stored_loss')

        ### 3) Parameters evolution
        self.stored_parameters_quantile.append(self.Model.quantile.item())
        self.stored_parameters_thickness.append(self.Model.Material.Structure.thickness.item())
        self.stored_parameters_alpha.append(self.Model.Material.alpha.item())
        self.stored_parameters_nu.append(self.Model.Material.Covariance.nu.item())
        self.stored_parameters_corrlen.append(self.Model.Material.Covariance.log_corrlen.exp().item())
        self.stored_parameters_noise_q.append(self.Model.Material.noise_quantile.item())
        save_as_csv(self.stored_parameters_quantile, self.output_folder + 'stored_parameters_quantile')
        save_as_csv(self.stored_parameters_thickness, self.output_folder + 'stored_parameters_thickness')
        save_as_csv(self.stored_parameters_alpha, self.output_folder + 'stored_parameters_alpha')
        save_as_csv(self.stored_parameters_nu, self.output_folder + 'stored_parameters_nu')
        save_as_csv(self.stored_parameters_corrlen, self.output_folder + 'stored_parameters_corrlen')
        save_as_csv(self.stored_parameters_noise_q, self.output_folder + 'stored_parameters_noise_q')


    ##############################################################

    def set_not_require_grad(self, not_require_grad):
        assert( isinstance(not_require_grad, list) )

        if len(not_require_grad) == 0:
            return

        if isinstance(not_require_grad[0], str):
            for name, p in self.Model.named_parameters():
                if name in not_require_grad:
                    p.requires_grad = False
            return

        if isinstance(not_require_grad[0], int):
            for num, p in enumerate(self.Model.parameters()):
                if num in not_require_grad:
                    p.requires_grad = False
            return



        # self.Model.Material.Structure.thickness.requires_grad = False
        # params = [self.Model.quantile, self.Model.Material.Covariance.log_nu, self.Model.Material.Covariance.log_corrlen]

##############################################################
##############################################################


    
def save_as_csv(data, fileneame):
    a = np.asarray(data)
    if a.ndim==1: a = a.reshape([-1, 1])
    np.savetxt(fileneame+".csv", a, delimiter=",")




