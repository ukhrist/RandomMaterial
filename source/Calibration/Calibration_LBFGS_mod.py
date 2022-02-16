
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

from LBFGS import LBFGS ### requires LBFGS modul for PyTorch from https://github.com/hjmshi/PyTorch-LBFGS


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
        # params  = self.Model.Material.parameters()
        params  = self.Model.parameters()
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

        def update_batch():
            self.Batch, self.vals, self.grads = [], [], []
            test = False
            while True:
                self.Batch, Batch_plus = self.extend_batch(self.Batch, self.batch_size)            
                grads_plus, vals_plus = get_grad(Batch_plus)
                self.grads.extend(grads_plus)
                self.vals.extend(vals_plus)
                i_test = self.inner_product_test(self.grads, kappa=1)
                if self.SGD:
                    o_test = self.orthogonality_test(self.grads, kappa=1)
                else:
                    o_test = True
                print('tests :', i_test, o_test)
                if i_test and o_test: break
                self.batch_size = self.batch_size + 1
            return self.grads, self.vals

        ### store initial state
        self.store_iteration_results()
        self.g_hist = []

        ### MAIN LOOP
        for n_iter in range(max_iter):
            self.n_iter = n_iter

            self.print_parameters()

            grads, vals = update_batch()
            print('Batch size =', len(self.Batch))

            grads = torch.stack(grads)
            g_Sk  = grads.mean(dim=0)
            loss  = torch.stack(vals).mean(dim=0)
        
            if self.SGD:
                p = -g_Sk
            else:
                ## two-loop recursion to compute search direction
                p = self.Optimizer.two_loop_recursion(-g_Sk)
                alpha_init = self.init_line_search(grads)
                p = alpha_init * p

            print('g_Sk    = ', g_Sk.data.tolist())
            print('Hg_Sk   = ', (-p.data).tolist())
            print('Misfits = ', self.Model.dist.misfit.data.tolist())
    
            ### perform line search step
            options = { 'closure'       : closure,
                        'current_loss'  : loss,
                        'max_ls'        : 20,
                        'inplace'       : True,
                        'interpolate'   : True,
                        'ls_debug'      : self.debug,
                    }
            result  = self.Optimizer.step(p, g_Sk, g_Sk=g_Sk, options=options)
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
            self.crit = inc.norm(p=np.infty) #torch.inner(g_Sk, inc).abs() #/loss.item()

            if not self.SGD:
                ### recompute gradient
                grads, vals = get_grad(self.Batch)
                grads = torch.stack(grads)
                g_Sk  = grads.mean(dim=0)
                loss  = torch.stack(vals).mean(dim=0)
                self.g_Sk = g_Sk
                self.loss = loss
                self.g_hist.append(g_Sk)

                ### curvature update
                self.Optimizer.curvature_update(g_Sk, eps=curvature_eps, damping=Powell_dumping)


            

            ### print data
            if self.verbose:               
                print()
                print('=========================================================================')
                print('Iter:', n_iter + 1, 'lr:', lr, 'Loss:', loss.item(), 'Grad:', g_Sk.norm(p=np.infty).item(), 'Inc:', inc.norm(p=np.infty).item(), 'Crit:', self.crit.item())
                print('=========================================================================')
                print()

                if self.Model.Material.ndim == 3:
                    self.Model.Material.calibration_regime = False
                    self.Model.Material.save_vtk(output_folder+'current_sample', seed=0)
                    self.Model.Material.calibration_regime = True
                    self.Model.Material.export_parameters(output_folder + 'inferred_parameters')

            ### store current state 
            self.store_iteration_results()
               
            ### stop criterion
            if self.crit.item() < tol: break
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
        return np.random.randint(10000000, size=size)

    def extend_batch(self, Batch, size):
        size_ini = len(Batch)
        while len(Batch) < size:
            Batch_plus = self.draw(size-len(Batch))
            Batch      = np.append(Batch,Batch_plus).astype(np.int)
            Batch      = pd.unique(Batch)
        Batch_plus = Batch[size_ini:]
        return Batch, Batch_plus

    

    def inner_product_test(self, grads, kappa=1):

        S  = len(grads)
        gs = grads if torch.is_tensor(grads) else torch.stack(grads, dim=0)
        g  = gs.mean(dim=0)
        if self.SGD:
            Hg  = g
            HHg = g
        else:
            Hg = self.Optimizer.two_loop_recursion(1.*g)
            HHg= self.Optimizer.two_loop_recursion(1.*Hg)
        
        Var = torch.tensor(0.)
        for gi in gs:
            Var += (torch.inner(gi, HHg) - Hg.norm()**2).square()
        Var = Var / max(S-1,1)
        test  = ( Var/S  <= kappa**2 * Hg.norm()**4 )
        self.batch_size_bound = Var / (kappa**2 * Hg.norm()**4)
        return test


    def orthogonality_test(self, grads, kappa=1):

        S  = len(grads)
        gs = grads if torch.is_tensor(grads) else torch.stack(grads, dim=0)
        g  = gs.mean(dim=0)
        d  = g/g.norm()
        
        Orth = torch.tensor(0.)
        for gi in gs:
            Orth += (gi - torch.inner(gi, d)*d).norm().square()
        Orth = Orth / max(S-1,1)
        test  = ( Orth/S  <= kappa**2 * g.norm()**2 )
        return test



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




