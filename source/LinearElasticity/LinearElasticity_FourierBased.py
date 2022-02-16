
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
import pickle
import os.path



# Linear solver settings
def create_Solver(A, tol=1.e-6):
	ksp = PETSc.KSP().create()
	ksp.setOperators(A)
	ksp.setType('cg')
	pc = ksp.getPC()
	pc.setType('none')
	ksp.setFromOptions()
	ksp.setTolerances(atol=tol, divtol=1.e6, max_it=1000)
	ksp.setInitialGuessNonzero(True)
	return ksp



#######################################################################################################
#	Linear Elastisity problem class
#######################################################################################################

class LinearElastisityProblem_FB(object):

	ProblemName = 'Linear Elastisity'

	def __init__(self, **kwargs):
		self.verbose 	  = kwargs.get("verbose", False)
		self.outputfolder = kwargs.get("outputfolder", "./")
		self.export_vtk   = kwargs.get("export_vtk", False)

		self.ndim 		= kwargs.get("ndim", 3)
		self.grid_level = kwargs.get("grid_level")
		self.mesh_level = kwargs.get("mesh_level")

		self.N	  = int(2**self.grid_level)
		self.Nd   = self.N * np.ones(self.ndim, dtype=np.intc)
		self.Nd_c = self.N * np.ones(self.ndim, dtype=np.intc)
		self.Nd_c[-1] = int(self.Nd_c[-1] // 2) + 1
		self.nPoints  = np.prod(self.Nd)
		self.nTensElt = numberTensorComponents(self.ndim)
		self.shpTc = list(self.Nd_c) + [self.nTensElt, self.nTensElt]

		self.frq = fft.fftfreq(self.N)
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
		self.Ref_factor = factor
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
		self.setGreenOperator(self.Ntrunc, kwargs.get("nproc", 1))
		if self.verbose: print('GreenOp time:', time()-t0)


		self.algebra_ready = False
		self.tol = kwargs.get("tol", 1.e-6)

		### Loading
		self.loading_type = kwargs.get("loading_type", "strain")
		self.set_loading(type=self.loading_type, value=kwargs.get("MacroTensor"))

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

	### Green
	def setGreenOperator(self, Ntrunc, nproc=1):

		filename_GreenTensor = self.outputfolder + 'GreenTensor_grid{0:d}_M{1:d}_F{2:d}.pkl'.format(self.grid_level, Ntrunc, self.Ref_factor)
		if os.path.isfile(filename_GreenTensor):
			with open(filename_GreenTensor, 'rb') as filehandler:
				self.GreenOp = pickle.load(filehandler)
			return self.GreenOp
		
		nd = list(self.Nd) + [self.nTensElt, self.nTensElt]
		nd_c = list(self.Nd_c) + [self.nTensElt, self.nTensElt]

		if nproc==0: nproc = 1
		if nproc<0 or nproc is None: nproc = mp.cpu_count()

		if nproc>1:
			print('N proc=', nproc)
			nvoxels_per_proc   = int(self.Nd_c[0] / nproc)
			nvoxels_first_proc = self.Nd_c[0] - nvoxels_per_proc * (nproc-1)
			nvoxels_per_proc   = [nvoxels_first_proc] + [nvoxels_per_proc] * (nproc-1)
			cs = np.cumsum(np.array([0]+nvoxels_per_proc, dtype=np.intc))
			
			Idx_begin = np.array([0*self.Nd_c] * nproc)
			Idx_end   = np.array([  self.Nd_c] * nproc)
			Idx_begin[:,0] = cs[:-1]
			Idx_end[:,0]   = cs[1:]

			pool = mp.Pool(processes=nproc)
			results = [pool.apply_async(cpplib.construct_FourierOfPeriodizedGreenOperator, args=(self.lmbda_ref, self.mu_ref, Idx_begin[iproc], Idx_end[iproc], self.ndim, self.frq, Ntrunc)) for iproc in range(nproc)]
			pool.close()
			pool.join()

			self.GreenOp = []
			for p in results:
				self.GreenOp = np.append(self.GreenOp, p.get())

		else:
			### Serial
			self.GreenOp = cpplib.construct_FourierOfPeriodizedGreenOperator(	self.lmbda_ref, self.mu_ref,
																						0*self.Nd, self.Nd_c, self.ndim, 
																						self.frq, Ntrunc)	

		with open(filename_GreenTensor, 'wb') as filehandler:
			pickle.dump(self.GreenOp, filehandler)

		return self.GreenOp



	### Init PETSc Solver
	def set_Solver(self):
		d, N = self.ndim, self.nTensElt
		
		# comm = PETSc.COMM_WORLD
		# comm = PETSc.COMM_SELF
		# x = PETSc.Vec().create(comm=comm)
		# x.setSizes(self.nPoints*self.nTensElt, bsize=self.nPoints*self.nTensElt)

		x = PETSc.Vec().createSeq(self.nPoints*self.nTensElt)
		x.setUp()

		b = x.duplicate()

		A = PETSc.Mat().createPython([b.getSizes(), x.getSizes()], comm=x.comm)
		A.setPythonContext(self)
		A.setUp()

		self.A = A
		self.Solver = create_Solver(A, self.tol)
		self.Solution, self.RHS = x, b


	### Init FFTW
	def set_fft(self):
		nd = list(self.Nd) + [self.nTensElt]
		nd_c = list(self.Nd_c) + [self.nTensElt]
		# nd_c[self.ndim-1] = int(nd_c[self.ndim-1] // 2) +1

		axes = np.arange(self.ndim)
		flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED')
		# flags=('FFTW_EXHAUSTIVE',)#, 'FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED')
		self.fft_x     = pyfftw.empty_aligned(nd, dtype='float64')
		self.fft_y 	   = pyfftw.empty_aligned(nd_c, dtype='complex128')
		self.fft_plan  = pyfftw.FFTW(self.fft_x, self.fft_y, axes=axes, direction='FFTW_FORWARD',  flags=flags)#, threads=4)
		self.ifft_plan = pyfftw.FFTW(self.fft_y, self.fft_x, axes=axes, direction='FFTW_BACKWARD', flags=flags)#, threads=4)


	#------------------------------------------------------------------------------------------
	# Set components
	#------------------------------------------------------------------------------------------

	def prepare_algebra(self):
		if not self.algebra_ready:
			self.set_fft()
			self.set_Solver()
			self.algebra_ready = True


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
		elif torch.is_tensor(Phase):
			Phase = Phase.detach().numpy()
		self.Phase = Phase.astype(np.intc)


	#------------------------------------------------------------------------------------------
	# Solve the problem (in strains)
	#------------------------------------------------------------------------------------------


	def solve(self, Phase=None):
		if Phase is not None: self.set_Microstructure(Phase)
		self.prepare_algebra()

		### Initialize
		self.RHS.array[:] = self.MacroTensor_array
		# cpplib.apply_Matrix_cpp(self.Phase.flatten(),
		# 						self.CC_M.flatten(), self.CC_I.flatten(),
		# 						self.Nd, self.nTensElt,
		# 						self.MacroTensor_array, self.RHS.array)
		self.RHS.copy(self.Solution)
		# self.Solution.array[:] = 0
		self.Krylov_iter = 0

		### Solve
		t0 = time()
		self.Solver.solve(self.RHS, self.Solution)
		if self.verbose:
			print('FB-Solver runtime:', time()-t0)
			print('Number of Krylov iterations:', self.Krylov_iter)
			print('Residual error:', (self.A*self.Solution-self.RHS).norm())

		self.evaluate_fields()

		return self.Krylov_iter


	def mult(self, mat, X, Y):

		# eta_array = self.compute_eta(X.array)
		# cpplib.apply_Matrix_cpp(self.Phase, self.CC_M.flatten(), self.CC_I.flatten(),
		# 						self.Nd, self.nTensElt,
		# 						eta_array, Y.array)
		# Y.array[:] += X.array

		# cpplib.apply_Matrix_cpp(self.Phase.flatten(), self.CCinv_M.flatten(), self.CCinv_I.flatten(),
		# 						self.Nd, self.nTensElt,
		# 						X.array, Y.array)
		# Y.array[:] += self.compute_eta(X.array)

		cpplib.apply_Matrix_cpp(self.Phase.flatten(), self.CC_M.flatten(), self.CC_I.flatten(),
								self.Nd, self.nTensElt,
								X.array, Y.array)
		Y.array[:] = X.array + self.compute_eta(Y.array)

		# t0 = time()
		# cpplib.apply_Matrix_cpp(self.Phase.flatten(), self.CCinv_M.flatten(), self.CCinv_I.flatten(), self.Nd, self.nTensElt, X.array, Y.array)
		# if self.verbose>=10: print('Compliance',time()-t0)

		# # t1 = time()
		# # self.fft_x.flat[:] = X.array
		# # self.fft_plan()
		# # cpplib.apply_Green_cpp(self.GreenOp, self.fft_y, self.Nd_c, self.nTensElt, self.eta_hat_flat)
		# # if self.loading_type=='stress':
		# # 	tau_avg = self.fft_y.real.flat[:self.nTensElt] / self.nPoints
		# # 	self.eta_hat_flat[:self.nTensElt] += np.dot(self.Cinv_ref, tau_avg) * self.nPoints
		# # self.fft_y.flat[:] = self.eta_hat_flat
		# # self.ifft_plan()
		# # Y.array[:] += self.fft_x.flat
		# Y.array[:] += self.compute_eta(X.array)
		# if self.verbose>=10: print('Green_tot', time()-t1)
		
		if self.verbose>=10:
			print('Mult',time()-t0)
			print()

		self.Krylov_iter+=1

	def compute_eta(self, X):
		self.fft_x.flat[:] = X
		self.fft_plan()
		cpplib.apply_Green_cpp(self.GreenOp, self.fft_y, self.Nd_c, self.nTensElt, self.eta_hat_flat)
		# if self.loading_type=='stress':
		# 	tau_avg = self.fft_y.real.flat[:self.nTensElt]
		# 	self.eta_hat_flat[:self.nTensElt] += np.dot(self.Cinv_ref, tau_avg)
		self.fft_y.flat[:] = self.eta_hat_flat
		self.ifft_plan()
		eta = self.fft_x.flat
		if self.loading_type=='stress':
			tau_avg = X.reshape(self.tau.shape).mean(axis=(0,1,2))
			addon   = np.dot(self.Cinv_ref, tau_avg)
			addon_array = np.tile(addon, self.nPoints)
			eta = eta + addon_array
		return eta



	#------------------------------------------------------------------------------------------
	# Fields
	#------------------------------------------------------------------------------------------
	
	def evaluate_fields(self):
		self.Strain = np.zeros(self.nPoints*self.nTensElt)
		self.Stress = np.zeros(self.nPoints*self.nTensElt)

		# cpplib.apply_Matrix_cpp(self.Phase.flatten(), self.CCinv_M.flatten(), self.CCinv_I.flatten(), self.Nd, self.nTensElt, self.Solution.array, self.Strain)
		# self.Strain.flat[:] = self.MacroTensor_array - self.compute_eta(self.Solution.array)
		self.Strain.flat[:] = self.Solution.array

		cpplib.apply_Matrix_cpp(self.Phase,     self.C_M.flatten(),     self.C_I.flatten(), self.Nd, self.nTensElt,         self.Strain, self.Stress)

		self.Strain = self.Strain.reshape(self.tau.shape)
		self.Stress = self.Stress.reshape(self.tau.shape)

		self.Fatigue = self.FatigueCriteria(self.Stress)

		# self.TrStress = np.sum(self.Stress[...,:self.ndim],-1)
		# if self.ndim==2 and self.plane_stress_strain is 'strain':
		# 	self.TrStress = (1 + self.nu[0]*(1-self.Phase) + self.nu[1]*self.Phase) * self.TrStress

		# self.traction = np.maximum(self.TrStress, 0.)

		# self.Dev = self.Stress - 1/3 * self.TrStress[...,None]
		# self.DevNorm2 = np.sum(self.Dev*self.Dev, -1)

		# self.Energy = np.sum(self.Stress*self.Strain, -1)

		# self.MeanStrain = self.Strain.reshape([-1,self.nTensElt]).mean(axis=0)
		# self.MeanStress = self.Stress.reshape([-1,self.nTensElt]).mean(axis=0)

		# print('Mean verif:', np.sum(self.Strain, (0,1)) / np.prod(self.Nd))

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


	### Young's modulus as QoI
	def compute_MacroStiffness(self, Phase=None):
		if Phase is not None: self.set_Microstructure(Phase)

		self.set_loading(type=self.loading_type, value=[1,0,0,0,0,1])
		self.solve()

		LocalStrain = self.Strain
		LocalStress = self.Stress
		MacroStrain = LocalStrain.reshape([-1,self.nTensElt]).mean(axis=0)
		MacroStress = LocalStress.reshape([-1,self.nTensElt]).mean(axis=0)

		s1 =  MacroStrain[0]  / MacroStress[0]
		s2 = -MacroStrain[1]  / MacroStress[0]
		s3 =  MacroStrain[-1] / MacroStress[-1]

		### Homogenized Young's modulus
		K = 1/self.ndim * self.Trace(MacroStress) / self.Trace(MacroStrain)
		G = 1/2 * MacroStress[-1] / MacroStrain[-1]
		Young = 9*K*G/(3*K+G)

		vf = self.Phase.mean()

		if self.export_vtk: self.export(self.outputfolder + "result")

		return [s1, s2, s3, K, G, Young, vf]



		# t = time()
		# # self.set_loading(type='stress', theta=0) ### theta=0
		# self.set_loading(type='stress', value=[1.,0,0,0,0,0])
		# # self.set_loading(type='strain', value=[1,0,0,0,0,0])
		# self.solve()
		# Strain0 = self.Strain
		# Stress0 = self.Stress

		# if self.export_vtk: self.export(self.outputfolder + "load0")

		# self.set_loading(type='stress', theta=pi/2) ### theta=pi/2
		# self.solve()
		# Strain1 = self.Strain
		# Stress1 = self.Stress

		# if self.export_vtk: self.export(self.outputfolder + "load1")

		# if self.verbose>=2: print('Solve time:', time()-t)

		# ### Homogenized strains
		# E0 = Strain0.reshape([-1,self.nTensElt]).mean(axis=0)
		# E1 = Strain1.reshape([-1,self.nTensElt]).mean(axis=0)
		# S0 = Stress0.reshape([-1,self.nTensElt]).mean(axis=0)
		# S1 = Stress1.reshape([-1,self.nTensElt]).mean(axis=0)
		# if self.verbose:
		# 	if self.ndim==2:
		# 		S0_imp = np.array([1, 1, 0])/sqrt(2)
		# 		S1_imp = np.array([0, 0, -1])/sqrt(2)
		# 	else:
		# 		S0_imp = np.array([1, 1, 1, 0, 0, 0])/sqrt(3)
		# 		S1_imp = np.array([0, 0, 0, 0, 0, -1])/sqrt(2)
		# 	print('MeanStress check: ', np.linalg.norm(S0-S0_imp), np.linalg.norm(S1-S1_imp))

		# ### Homogenized Young's modulus
		# K = 1/self.ndim * self.Trace(S0) / self.Trace(E0)
		# G = 1/2 * S1[-1] / E1[-1]
		# Young = 9*K*G/(3*K+G)
		# return Young

	
	### Maximum Stress as QoI !!!	
	def compute_QoIs(self, Phase=None, nticks=None):
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
			# for i in range(E0.size):
			# 	Q['E0_{0:d}'.format(i)] = E0[i]
			# 	Q['E1_{0:d}'.format(i)] = E1[i]
			# for i in range(E0.size):
			# 	Q[('E0',i)] = E0[i]
			# 	Q[('E1',i)] = E1[i]
			Q['E0'] = E0.tolist()
			Q['E1'] = E1.tolist()
			# return np.array(Q + E0.tolist() + E1.tolist())
			# return Q
		# else:
		# 	return Q
		return Q


	def vicinity_of_maximum(self, Field, radius=None):
		if radius is None: radius = self.neighborhood_radius

		t = time()
		rh= ceil(radius*self.N)
		n = 2*rh+1
		if self.ndim==2:
			X, Y = np.ogrid[:n,:n]
			X, Y = X-rh, Y-rh
			dist_from_center = np.sqrt(X**2 + Y**2)
		elif self.ndim==3:
			X, Y, Z = np.ogrid[:n,:n,:n]
			X, Y, Z = X-rh, Y-rh, Z-rh
			dist_from_center = np.sqrt(X**2 + Y**2 + Z**2)
		else:
			msgDimError(self.ndim)

		CircularSubMask = dist_from_center<=rh

		center = np.array(np.unravel_index(Field.argmax(), Field.shape))
		start, end = (center-rh) % self.N, (center+rh) % self.N
		dom_slice = [slice(0, n)]*self.ndim

		Mask = np.zeros_like(self.Phase)
		Mask[dom_slice] = CircularSubMask
		for j in range(self.ndim): Mask = np.roll(Mask, start[j], axis=j)
		Mask *= np.logical_not(self.Phase)
		# print('Subdomain time:', time()-t)
		return Mask





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
			cellData['strain_{}'.format(i)] = np.array(self.Strain[...,i])
			cellData['stress_{}'.format(i)] = np.array(self.Stress[...,i])
		# cellData['strain'] = tuple( self.Strain[...,i].copy(order='F') for i in range(self.nTensElt) )
		# cellData['stress'] = tuple( self.Stress[...,i].copy(order='F') for i in range(self.nTensElt) )
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























