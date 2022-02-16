"""
==================================================================================================================
Warning: NOT FINISHED !!!
==================================================================================================================
"""




# from dolfin import *

import pyfftw
from .utilities import *
# from utilities.ErrorMessages import *
from math import *
import numpy as np
from scipy import ndimage, misc
import scipy.fftpack as fft
import scipy.integrate as integrate
from petsc4py import PETSc
import sys
import importlib
from tqdm import tqdm
import itertools
from time import time
import multiprocessing as mp
from . import cpplib
from ..utils import exportVTK
import torch

import torch



# Linear solver settings
def create_Solver(A, tol=1.e-4):
	ksp = PETSc.KSP().create()
	ksp.setOperators(A)
	ksp.setType('cg')
	pc = ksp.getPC()
	pc.setType('none')
	ksp.setFromOptions()
	ksp.setTolerances(rtol=tol, divtol=1.e6, max_it=1000)
	ksp.setInitialGuessNonzero(True)
	return ksp




"""
==================================================================================================================
Linear Elastisity problem class (Lippmann-Schwinger solver -- Fourier-based)
==================================================================================================================
"""

class LinearElastisityProblem_Fourier(object):

	ProblemName = 'Linear Elastisity'

	def __init__(self, **kwargs):
		self.verbose 	= kwargs.get("verbose", False)

		self.ndim 		= kwargs.get("ndim", 3)
		self.grid_level = kwargs.get("grid_level")
		self.mesh_level = kwargs.get("mesh_level")

		self.N	  = int(2**self.grid_level)
		self.shape = self.N * np.ones(self.ndim, dtype=np.intc)
		self.Nd   = self.N * np.ones(self.ndim, dtype=np.intc)
		self.Nd_c = self.N * np.ones(self.ndim, dtype=np.intc)
		self.Nd_c[-1] = int(self.Nd_c[-1] // 2) + 1
		self.nPoints  = np.prod(self.Nd)
		self.nVoxels  = np.prod(self.Nd)
		self.nTensElt = numberTensorComponents(self.ndim)
		self.shpTc = list(self.Nd_c) + [self.nTensElt, self.nTensElt]

		self.frq = fft.fftfreq(self.N)
		self.frq = torch.tensor(self.frq)
		self.freqs  = torch.stack(torch.meshgrid(*([self.frq]*self.ndim), indexing="ij"))
		self.Ntrunc = kwargs.get("n_terms_trunc_PeriodizedGreen")

		self.nticks_qoi = kwargs.get("nticks_qoi")
		self.nqois 		= self.nticks_qoi
		self.neighborhood_radius = kwargs.get("neighborhood_radius")

		self.plane_stress_strain = kwargs.get("plane_stress_strain")

		self.h = 1/float(self.N)
		self.dV = self.h**self.ndim

		### Material moduli
		self.E  = np.array([ kwargs.get("Young_modulus_M"), kwargs.get("Young_modulus_I") ])
		self.nu = np.array([ kwargs.get("Poisson_ratio_M"), kwargs.get("Poisson_ratio_I") ])
		self.Scale = max(self.E)
		self.E /= self.Scale
		self.lmbda, self.mu = transfer_YoungPoisson_to_Lame(self.E, self.nu)
		self.K, self.mu = transfer_YoungPoisson_to_BulkShear(self.E, self.nu, d=self.ndim)
		self.lmbda_M, self.lmbda_I = self.lmbda
		self.mu_M, self.mu_I = self.mu

		self.C_M = set_StiffnessMatrix(self.E[0], self.nu[0], d=self.ndim, plane_stress_strain=self.plane_stress_strain)
		self.C_I = set_StiffnessMatrix(self.E[1], self.nu[1], d=self.ndim, plane_stress_strain=self.plane_stress_strain)

		if self.verbose: print('Contrast =', self.mu_I/self.mu_M)

		### Reference stiffness matrix

		factor = kwargs.get("ref_stiff_factor")
		if factor=='HS':
			HS = HashinShtrikmanBounds(kwargs.get("fI"), bulk=self.K, shear=self.mu, d=self.ndim)
			self.K_ref, self.mu_ref = HS[0][1], HS[1][1]
		else:
			self.K_ref, self.mu_ref = factor*max(self.K), factor*max(self.mu)
		self.E_ref, self.nu_ref = transfer_BulkShear_to_YoungPoisson(self.K_ref, self.mu_ref, d=self.ndim)
		self.lmbda_ref, self.mu_ref = transfer_YoungPoisson_to_Lame(self.E_ref, self.nu_ref)

		self.C_ref = set_StiffnessMatrix(self.E_ref, self.nu_ref, d=self.ndim, plane_stress_strain=self.plane_stress_strain)
		self.Cinv_ref = set_ComplianceMatrix(self.E_ref, self.nu_ref, d=self.ndim, plane_stress_strain=self.plane_stress_strain)

		### Modified compliance matrix
		E_diff, nu_diff = transfer_Lame_to_YoungPoisson(self.lmbda[0]-self.lmbda_ref, self.mu[0]-self.mu_ref)
		self.CC_M = set_StiffnessMatrix(E_diff, nu_diff, d=self.ndim, plane_stress_strain=self.plane_stress_strain)
		self.CCinv_M = set_ComplianceMatrix(E_diff, nu_diff, d=self.ndim, plane_stress_strain=self.plane_stress_strain)
		E_diff, nu_diff = transfer_Lame_to_YoungPoisson(self.lmbda[1]-self.lmbda_ref, self.mu[1]-self.mu_ref)
		self.CC_I = set_StiffnessMatrix(E_diff, nu_diff, d=self.ndim, plane_stress_strain=self.plane_stress_strain)
		self.CCinv_I = set_ComplianceMatrix(E_diff, nu_diff, d=self.ndim, plane_stress_strain=self.plane_stress_strain)

		if self.verbose:
			C1 = set_StiffnessMatrix(self.E_ref, self.nu_ref, d=self.ndim, plane_stress_strain=self.plane_stress_strain)
			C1inv = set_ComplianceMatrix(self.E_ref, self.nu_ref, d=self.ndim, plane_stress_strain=self.plane_stress_strain)
			print("Test Stiffness:", np.linalg.norm(np.matmul(C1inv,C1)-np.eye(self.nTensElt)))

		### 4th order Green operator
		if self.verbose: print('Building 4th rank Green operator..')
		t0 = time()
		self.compute_PeriodizedGreenTensor(self.Ntrunc)
		# self.setGreenOperator(self.Ntrunc, kwargs.get("nproc", 1))
		if self.verbose: print('GreenOp time:', time()-t0)


		self.algebra_ready = False

		### Loading
		self.set_loading(type=kwargs.get("loading_type"), value=kwargs.get("MacroTensor"))

		### Init vectors
		nd   = list(self.Nd)   + [self.nTensElt]
		nd_c = list(self.Nd_c) + [self.nTensElt]
		self.tau     = np.zeros(nd)
		self.tau_hat = np.zeros(nd_c, dtype=np.complex)
		self.eps     = np.zeros_like(self.tau)
		self.eta     = np.zeros_like(self.tau)
		self.eta_hat = np.zeros_like(self.tau_hat)
		self.eta_hat_flat = np.zeros(self.eta_hat.size, dtype=np.complex)
		self.tau_flat = np.zeros(self.tau.size)


	#------------------------------------------------------------------------------------------
	# Initialization
	#------------------------------------------------------------------------------------------

	def compute_PeriodizedGreenTensor(self, Ntrunc=2):
		k = self.freqs

		factor0    = torch.where(k!=0, ( k * torch.sin(pi*k)/pi )**2, 1.).prod(dim=0)
		factor_loc_vec = torch.zeros([self.ndim] + list(self.shape))

		M_grid = torch.arange(2*Ntrunc+1) - Ntrunc
		M_meshgrid = torch.stack(torch.meshgrid(*( [M_grid]*self.ndim ), indexing="ij"))
		M_meshgrid = M_meshgrid.reshape([self.ndim, -1]).T

		G = torch.zeros([self.ndim]*4 + list(self.shape))
	
		for m in M_meshgrid:			
			k_mod = k + m.reshape([self.ndim] + [1]*(k.dim()-1))

			for i in range(self.ndim):
				factor_loc_vec[i] = torch.where(k[i]!=0, 1./k_mod[i]**2, 0. if m[i] else 1.)
			factor_loc = factor_loc_vec.prod(dim=0)
			
			k_mod_norm = k_mod.norm()
			if k_mod_norm:
				n = k_mod / k_mod_norm
				G_loc = self.compute_GreenTensor(n)
				G += factor_loc * G_loc
			print('m=', m)

		G = factor0 * G

		n_half = int(np.ceil(0.5*G.shape[-1]))
		G = G[..., :n_half]

		self.GreenTensor_hat = G
		return G


	def compute_GreenTensor(self, n):
		G = torch.zeros([self.ndim]*4 + list(self.shape))
		# coef = (self.lmbda_ref+self.mu_ref)/(self.lmbda_ref+2*self.mu_ref)
		coef1 = 0.25 / self.mu_ref
		coef2 = 0.5 / (1-self.nu_ref) / self.mu_ref
		for i in range(self.ndim):
			for j in range(self.ndim):
				for h in range(self.ndim):
					for l in range(self.ndim):						
						if i>j:
							G[i,j,h,l,...] = G[j,i,h,l,...]
						elif h>l:
							G[i,j,h,l,...] = G[j,i,l,h,...]
						else:
							N1 = delta(i,h)*n[j]*n[l] + delta(i,l)*n[j]*n[h] + delta(j,h)*n[i]*n[l] + delta(j,l)*n[i]*n[h]
							N2 = n[i]*n[j]*n[h]*n[l]
							G[i,j,h,l,...] = coef1 * N1 - coef2 * N2
		return G



	def set_loading(self, type=None, value=None, theta=0):
		if value is not None:
			MacroTensor = np.array(value[:self.nTensElt], dtype=np.float)
		else:
			T0, T1 = np.zeros(self.nTensElt), np.zeros(self.nTensElt)
			T0[:self.ndim] = 1/sqrt(self.ndim)
			T1[-1] = -1/sqrt(2)
			MacroTensor = cos(theta)*T0 + sin(theta)*T1

		if type is not None:
			self.loading_type = type

		if self.verbose>=3: print('Impose:', self.loading_type, MacroTensor.tolist())

		if self.loading_type=='stress':
			MacroTensor = np.dot(self.Cinv_ref, MacroTensor)

		self.MacroTensor = MacroTensor
		self.MacroTensor_array = np.tile(MacroTensor, self.nPoints)


	def set_Microstructure(self, Phase):
		if type(Phase) is str:
			Image = misc.imread(Phase, True, mode='L')
			Phase = 1.*(Image < Image.mean())
		self.Phase = Phase.astype(np.intc)


	#------------------------------------------------------------------------------------------
	# Solve the forward problem
	#------------------------------------------------------------------------------------------

	def forward(self, Phase=None):
		if Phase is not None:
			self.set_Microstructure(Phase)

		x = torch.zeros_like(self.Stress, requires_grad=False)
		b = torch.zeros_like(x)
		b[:] = self.Strain_hom.reshape([self.ndim]*2 + [1]*(b.shape[-1]-1))

		self.Polarization = self.solve(x, b)

		self.update_fields()
		self.update_QoIs()

		return self.Stiffness


	#------------------------------------------------------------------------------------------
	# Conjugate gradient solver
	#------------------------------------------------------------------------------------------

	def solve(self, x, b, **kwargs):
		tolerance = kwargs.get('tol', 1e-6)
		max_it    = kwargs.get('max_iter', 100)

		pbar = tqdm(total=max_it)
		pbar.set_description("iter / max_it");

		it = 0 # iteration counter
		diff = 1.0
		hist_error = []

		A = self.apply_Matrix

		r = b - A(x)
		p = 1*r
		r_norm_old = (r*r).sum(dim=[0,1]).detach()


		while (diff > tolerance):
			if it > max_it:
				print('\nSolution did not converged within the maximum number of iterations.')
				print(f'Last l2_diff was: {diff:.5e}\n')
				exit()
			
			Ap = A(p)
			alpha = r_norm_old / (p*Ap).sum(dim=[0,1])
			x = x + alpha*p
			r = r - alpha*Ap
			r_norm_new = (r*r).sum(dim=[0,1]).detach()
			p = r + (r_norm_new / r_norm_old) * p

			err        = r_norm_new
			r_norm_old = r_norm_new
			it += 1
			hist_error.append(err)
			pbar.update(1)

		else:
			print(f'\nThe solution converged after {it} Krylov iterations')

		del(pbar)
		return x

	#------------------------------------------------------------------------------------------
	# Operator action of the system matrix
	#------------------------------------------------------------------------------------------

	def apply_Matrix(self, tau):
		G_hat   = self.GreenTensor_hat
		CCinv   = self.CCinv
		tau_hat = torch.fft.rfft(tau, dim=[-i for i in range(self.ndim)])
		eta_hat = self.apply_4th_order_tensor(T4=G_hat, T2=tau_hat)
		eta     = torch.fft.irfft(eta_hat, n=[tau.shape[-i] for i in range(self.ndim)], dim=[-i for i in range(self.ndim)])
		Ax      = self.apply_4th_order_tensor(T4=CCinv, T2=tau) + eta
		return Ax


	def apply_4th_order_tensor(T4, T2):
		return (T4*T2).sum(dim=[2,3])

	#------------------------------------------------------------------------------------------
	# Fields
	#------------------------------------------------------------------------------------------
	
	def update_fields(self):
		self.Strain = np.zeros(self.nPoints*self.nTensElt)
		self.Stress = np.zeros(self.nPoints*self.nTensElt)

		self.Strain.flat[:] = self.MacroTensor_array - self.compute_eta(self.Solution.array)

		# cpplib.apply_Matrix_cpp(self.Phase.flatten(), self.CCinv_M.flatten(), self.CCinv_I.flatten(), self.Nd, self.nTensElt, self.Solution.array, self.Strain)
		cpplib.apply_Matrix_cpp(self.Phase,     self.C_M.flatten(),     self.C_I.flatten(), self.Nd, self.nTensElt,         self.Strain, self.Stress)

		self.Strain = self.Strain.reshape(self.tau.shape)
		self.Stress = self.Stress.reshape(self.tau.shape)

		self.Fatigue = self.FatigueCriteria(self.Stress)


	#------------------------------------------------------------------------------------------
	# Utils
	#------------------------------------------------------------------------------------------

	def Trace(self, Tensor):
		TensorTrace = np.sum(Tensor[...,:self.ndim],-1)
		if self.ndim==2 and self.plane_stress_strain is 'strain':
			TensorTrace *= (1 + self.nu[0]*(1-self.Phase) + self.nu[1]*self.Phase)
		return TensorTrace

	def TracePlus(self, Tensor):
		return np.maximum(self.Trace(Tensor), 0.) 

	def Deviator(self, Tensor, TensorTrace=None):
		if TensorTrace is None: TensorTrace = self.Trace(Tensor)
		Dev = Tensor.copy()
		Dev[...,:self.ndim] -= 1/3 * TensorTrace[...,None]
		return Dev

	def TensorNorm(self, Tensor):
		return np.sqrt(np.sum(Tensor**2, -1))

	def FatigueCriteria(self, Stress):
		t = self.Trace(Stress)
		s = self.Deviator(Stress, t)
		return self.TensorNorm(s) + 0.3*np.maximum(t, 0)







	#------------------------------------------------------------------------------------------
	# Post-treatment
	#------------------------------------------------------------------------------------------


	# def compute_Q0(self, subdomain=1):
	# 	Volume = np.sum(subdomain)		
	# 	if Volume:	
	# 		Q0 = 0.5*np.sum(self.Energy*subdomain) / np.sum(subdomain)
	# 	else: Q0 = 0
	# 	self.Q0, self.subdomain = Q0, subdomain
	# 	return Q0

	# def compute_Q1(self, subdomain=1):
	# 	Volume = np.sum(subdomain)
	# 	if Volume:
	# 		Q1 = np.sum(self.DevNorm2*subdomain) / self.mu_M / Volume
	# 	else: Q1 = 0
	# 	self.Q1, self.subdomain = Q1, subdomain
	# 	# Q1 = np.amax(self.DevNorm2)
	# 	return Q1

	# def compute_Q2(self, subdomain=1):
	# 	Volume = np.sum(subdomain)
	# 	if Volume:
	# 		Q2 = np.sum(self.traction*subdomain) / Volume
	# 	else: Q2 = 0
	# 	self.Q2, self.subdomain = Q2, subdomain
	# 	# Q2 = np.amax(self.traction)
	# 	return Q2


	def update_QoIs(self, Phase=None, nticks=None):
		if Phase is not None: self.set_Microstructure(Phase)
		if nticks is None: nticks = self.nticks_qoi

		t = time()
		self.set_loading(type='stress', theta=0) ### theta=0
		self.solve()
		Strain0 = self.Strain
		Stress0 = self.Stress

		self.set_loading(type='stress', theta=pi/2) ### theta=pi/2
		self.solve()
		Strain1 = self.Strain
		Stress1 = self.Stress
		if self.verbose>=2: print('Solve time:', time()-t)
		

		### Quantities of Interest (Q3)
		t = time()
		Q = {'Q3':[]}
		# Q = {}
		for i, theta in enumerate(np.linspace(0, pi/2, nticks)):
			Stress = cos(theta) * Stress0 + sin(theta) * Stress1
			q = self.FatigueCriteria(Stress)
			subdomain = self.vicinity_of_maximum(q)
			Volume = np.sum(subdomain)
			if Volume:
				Q_value = np.sqrt(np.sum((subdomain*q)**2) / Volume)
			else:
				raise Exception('Singular neighborhood of the maximum.')
			# Q['Q3_theta={}'.format(theta)] = Q_value
			# Q[('Q3',i)] = Q_value
			Q['Q3'].append(Q_value)
			if theta==0:
				self.Fatigue = q
				self.subdomain = subdomain
		if self.verbose>=2: print('Q3 time:', time()-t)

		### Quantities of Interest (E)
		if self.ndim==2:			
			E0 = Strain0.reshape([-1,self.nTensElt]).mean(axis=0)
			E1 = Strain1.reshape([-1,self.nTensElt]).mean(axis=0)
			if self.verbose:
				S0 = Stress0.reshape([-1,self.nTensElt]).mean(axis=0)
				S1 = Stress1.reshape([-1,self.nTensElt]).mean(axis=0)
				S0_imp = np.array([1, 1, 0])/sqrt(2)
				S1_imp = np.array([0, 0, -1])/sqrt(2)
				print('MeanStress check: ', np.linalg.norm(S0-S0_imp), np.linalg.norm(S1-S1_imp))
			Q['E0'] = E0.tolist()
			Q['E1'] = E1.tolist()
		return Q





#######################################################################################################
# Output
#######################################################################################################

	def export(self, FileName):
		cellData = {}
		try: cellData['phase'] = self.Phase
		except: pass
		try: cellData['subdomain'] = self.subdomain
		except: pass
		try: cellData['traction'] = self.traction
		except: pass
		try: cellData['dev_norm'] = np.sqrt(self.DevNorm2)
		except: pass
		try: cellData['fatigue'] = self.Fatigue
		except: pass
		# cellData = {'phase'  	: self.Phase,
		# 			'subdomain' : self.subdomain,
		# 			'traction' 	: self.traction,
		# 			'dev_norm' 	: np.sqrt(self.DevNorm2)}
		for i in range(self.Stress.shape[-1]):
			cellData['strain_{}'.format(i)] = np.array(self.Strain[...,0])
			cellData['stress_{}'.format(i)] = np.array(self.Stress[...,0])
		exportVTK(FileName, cellData = cellData)

#######################################################################################################
# Testing
#######################################################################################################

	def testFourierBased(self):

		from pymks.datasets import make_elastic_FE_strain_random

		elastic_modulus = (self.E[0], self.E[1])
		poissons_ratio = (self.nu[0], self.nu[1])
		macro_strain = self.MacroStrain
		size = (self.Nd[0], self.Nd[1])

		np.random.seed()
		X, strain = make_elastic_FE_strain_random(n_samples=1, elastic_modulus=elastic_modulus,
											poissons_ratio=poissons_ratio, size=size,
											macro_strain=macro_strain)

		print(X.shape)

		self.solve(X)

		print(np.linalg.norm(strain-self.Strain))



"""
==================================================================================================================
Utils
==================================================================================================================
"""

def delta(i, j):
	if i==j:
		return 1
	else:
		return 0


def Voigt2Index(I, d):
	if I<d:
		i = I
		j = I
	elif I==d:
		i = d-2
		j = d-1
	elif I==d+1:
		i = d-3
		j = d-1
	elif I==d+2:
		i = d-3
		j = d-2
	return i, j


"""
==================================================================================================================
Stiffness/Compliance Matrix
==================================================================================================================
"""

def numberTensorComponents(d):
	return int(d*(d+1) // 2)

def set_StiffnessMatrix(E, nu, d=3, plane_stress_strain='strain'):
	N = numberTensorComponents(d)
	C = np.zeros([N,N])
	if d==2 and plane_stress_strain=='stress':
		np.fill_diagonal(C, 1-nu)
		C[:d,:d] += nu
		C *= E/(1-nu**2)	
	else:
		np.fill_diagonal(C, 1-2*nu)
		C[:d,:d] += nu
		C *= E/(1+nu)/(1-2*nu)
	return C

def set_ComplianceMatrix(E, nu, d=3, plane_stress_strain='strain'):
	N = numberTensorComponents(d)
	if d==2 and plane_stress_strain=='strain':
		C = np.eye(N)
		C[:d,:d] += -nu
		C *= (1+nu)/E	
	else:
		C = np.zeros([N,N])
		np.fill_diagonal(C, 1+nu)
		C[:d,:d] += -nu
		C *= 1/E
	return C


#######################################################################################################
# Run as main (rather for testing)
#######################################################################################################

# if __name__ == "__main__":

# 	config = importlib.import_module(sys.argv[1])

# 	pb = LinearElastisityProblem_FB(config)

# 	from RandomMaterial import RandomMaterial
# 	RM = RandomMaterial(config)
# 	Phase = RM.sample()

# 	pb.solve(Phase)

# 	W = pb.compute_Q0()
# 	print('Energy =', W)















