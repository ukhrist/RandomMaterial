
from dolfin import *
from LinearElasticity.utilities import *
from utilities.ErrorMessages import *
import numpy as np
import math
import logging
import sys
import importlib

# Options
# logging.getLogger('FFC').setLevel(logging.WARNING)
# logging.getLogger('UFL').setLevel(logging.WARNING)
# set_log_active(False)
expr_degree = 1

# Linear solver settings
def set_linSolver():
	# solver = PETScLUSolver("mumps")
	# solver = PETScKrylovSolver("cg", "icc")
	solver = PETScKrylovSolver("gmres", "amg")
	solver.parameters["maximum_iterations"] = 1000
	solver.parameters["relative_tolerance"] = 1.e-6
	solver.parameters["absolute_tolerance"] = 1.e-6
	solver.parameters["error_on_nonconvergence"] = True
	solver.parameters["nonzero_initial_guess"] = True
	# solver.parameters["monitor_convergence"] = True
	return solver


def set_coefficient(c, c_I, c_M, chi):
	chi = 1-chi
	try:
		c.vector().set_local(c_M + (c_I-c_M)*chi.flatten())
	except:
		c.vector().set_local(c_M + (c_I-c_M)*chi.vector().get_local())


# Local strains
def strain(u, E):
	return sym(grad(u)) + E

# Local stress
def sigma(u, lmbda, mu, E):
	return lmbda*tr(strain(u, E))*Identity(len(u)) + 2*mu*strain(u, E)

# Elastic energy
def energy(u, lmbda, mu, E):
	return 0.5*inner(sigma(u, lmbda, mu), strain(u))

# Deviatoric stress
def devStress(u, lmbda, mu, E):
	return sigma(u, lmbda, mu, E) - 1/3 * tr(sigma(u, lmbda, mu, E)) * Identity(len(u))

# Traction
def traction(u, lmbda, mu, E):
	TrStress = tr(sigma(u, lmbda, mu, E))
	return 0.5*(TrStress + abs(TrStress)) # max(TrStress, 0.)



# class used to define the periodic boundary map
# class PeriodicBoundary2D(SubDomain):

#     def inside(self, x, on_boundary):
#         # return True if on left or bottom boundary AND NOT on one of the
#         # bottom-right or top-left vertices
#         return bool( ( near(x[0], 0) or near(x[1], 0) ) and 
#             		 (not ( ( near(x[0], 0) and near(x[1], 1) ) or ( near(x[0], 1) and near(x[1], 0) ) ) ) and 
# 					 on_boundary)

#     def map(self, x, y):
#         if near(x[0], 1) and near(x[1], 1): # if on top-right corner
#             y[0] = x[0] - 1
#             y[1] = x[1] - 1
#         elif near(x[0], 1): # if on right boundary
#             y[0] = x[0] - 1
#             y[1] = x[1]
#         else:   # should be on top boundary
#             y[0] = x[0]
#             y[1] = x[1] - 1

# class PeriodicBoundary3D(SubDomain):

# 	def inside(self, x, on_boundary):
# 		return bool( ( near(x[0], 0) or near(x[1], 0) or near(x[2], 0) ) and 
#             		 (not ( near(x[0], 1) or near(x[1], 1) or near(x[2], 1) ) ) and 
# 					 on_boundary)

# 	def map(self, x, y):
# 		if near(x[0], 1):
# 			y[0] = x[0] - 1
# 		else:
# 			y[0] = x[0]
		
# 		if near(x[1], 1):
# 			y[1] = x[1] - 1
# 		else:
# 			y[1] = x[1]
			
# 		if near(x[2], 1):
# 			y[2] = x[2] - 1
# 		else:
# 			y[2] = x[2]


class PeriodicBoundary(SubDomain):

	def __init__(self, ndim):
		SubDomain.__init__(self)
		self.ndim = ndim

	def inside(self, x, on_boundary):
		return bool( 	any([ near(x[j], 0) for j in range(self.ndim) ]) and 
					not any([ near(x[j], 1) for j in range(self.ndim) ]) and 
						on_boundary
					)

	def map(self, x, y):
		for j in range(self.ndim):
			if near(x[j], 1):
				y[j] = x[j] - 1
			else:
				y[j] = x[j]

def Tensor2Voigt(T): # (lists) slow!
	m, n = size(T)
	assert(m==n)
	t = []
	for i in range(n):
		t = t.append(T[i,i])
	for i in range(n):
		for j in range(i+1,n):
			t = t.append(T[i,j])
	return t

def Voigt2Tensor(t): # (lists) slow!
	n = len(t)
	d = (sqrt(8*n+1)-1)/2
	assert(d.is_integer())
	d = int(d)
	T = np.zeros([d,d])
	k = 0
	for i in range(d):
		T[i,i] = t[i]
		k += 1
	for i in range(d):
		for j in range(i+1,d):
			T[i,j] = t[k]
			T[j,i] = t[k]
			k += 1
	return T

#######################################################################################################
#	Linear Elastisity problem class
#######################################################################################################

class LinearElastisityProblem(object):

	def __init__(self, config, mesh=None):
		self.verbose = config.verbose

		self.ndim = config.ndim
		self.N	  = int(2**config.mesh_level)
		self.Nd   = (self.N,) * self.ndim
		self.nPoints  = np.prod(self.Nd)
		self.nTensElt = int(self.ndim*(self.ndim+1) // 2)
		self.W, self.H = 1, 1
		self.h = self.W/float(self.N)

		### Material moduli
		self.E  = np.array([ config.Young_modulus_M, config.Young_modulus_I ])
		self.nu = np.array([ config.Poisson_ratio_M, config.Poisson_ratio_I ])
		self.lmbda, self.mu = transfer_YoungPoisson_to_Lame(self.E, self.nu)
		self.lmbda_M, self.lmbda_I = self.lmbda
		self.mu_M, self.mu_I = self.mu

		### Mesh
		self.mesh = self.generate_Mesh()

        ### Formulate FEM
		self.init_FEM(config)

		### Right hand side
		# self.source  = Expression(config.Force_volumic, degree=expr_degree)
		self.Source = Expression(('0.',)*self.ndim, degree=expr_degree)
		self.MacroStrain = Constant(Voigt2Tensor(config.MacroTensor[:self.nTensElt]))
		# vh = TestFunction(self.Vh)
		# self.rhs = assemble(inner(self.source, vh)*dx)
		# for bc in self.BCs: bc.apply(self.rhs)

		### Create solver
		self.Asolver = set_linSolver()

		### QoIs
		# if subdomain is not None:
		# 	self.dI = dx(1, domain=self.mesh, subdomain_data=subdomain)
		# else:
		# 	self.dI = dx(self.mesh)

	#------------------------------------------------------------------------------------------
	# FEM formulation
	#------------------------------------------------------------------------------------------


	def generate_Mesh(self):

		if self.ndim == 1:
			mesh = UnitIntervalMesh(*self.Nd)
		elif self.ndim == 2:
			# mesh = UnitSquareMesh(*self.Nd)#, cell = "quadrilateral")
			mesh = UnitSquareMesh.create(*self.Nd, CellType.Type_quadrilateral)
		elif self.ndim == 3:
			mesh = UnitCubeMesh.create(*self.Nd, CellType.Type_hexahedron)#, cell = "hexahedron")
		else:
			raise Exception("The case of Dimension={0:d} is not inplemented!".format(self.ndim))

		self.h = 1./float(self.Nd[0])

        # if verbose: print() 
        # if verbose: print('Constructed {0:d}D Mesh : level {1:d}, h = {2:f}'.format(ndim, mesh_level, h))

        # if mesh_file is not None:
        #     File(mesh_file) << mesh
        #     if verbose: print('Saved to {0:s}'.format(mesh_file))
        # if verbose: print()

		return mesh
		
		
	def init_FEM(self, config):

		# if self.ndim==2:
		# 	PeriodicBoundary = PeriodicBoundary2D()
		# elif self.ndim==3:
		# 	PeriodicBoundary = PeriodicBoundary3D()
		# else:
		# 	raise Exception("The case of Dimension={0:d} is not inplemented!".format(self.ndim))

        ### FE Spaces
		# VE = VectorElement("CG", self.mesh.ufl_cell(), config.element_degree)
		# self.Vh = FunctionSpace(self.mesh, VE)
		self.Vh = VectorFunctionSpace(self.mesh, "CG", config.element_degree, constrained_domain=PeriodicBoundary(self.ndim))
		self.Dh = FunctionSpace(self.mesh, "DG", 0)


		### Boundaries
		# if config.ndim == 2:
		# 	bottom = CompiledSubDomain('on_boundary && near(x[1], 0.0)')
		# 	top    = CompiledSubDomain('on_boundary && near(x[1], 1.0)')
		# 	left   = CompiledSubDomain('on_boundary && near(x[0], 0.0)')
		# 	right  = CompiledSubDomain('on_boundary && near(x[0], 1.0)')

		# 	u_top 	 = Constant((0.0, 0.0))
		# 	u_bottom = Constant((0.0, 0.0))
		# 	u_left   = Constant((0.0, 0.0))
		# 	u_right  = Constant((0.0, 0.0))

		# 	self.BCs = [ 	DirichletBC(self.Vh, u_top, 	top), 		\
		# 					DirichletBC(self.Vh, u_bottom, 	bottom),	\
		# 					DirichletBC(self.Vh, u_left, 	left),		\
		# 					DirichletBC(self.Vh, u_right, 	right) 		]

		# 	### Boundary markers
		# 	self.markers = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1, 0)
		# 	bottom.mark(self.markers, 1)
		# 	top.mark(self.markers, 2)
		# 	left.mark(self.markers, 3)
		# 	right.mark(self.markers, 4)
		# 	self.dGamma = ds(subdomain_data = self.markers)
		# else:
		# 	raise Exception("The case of Dimension={0:d} is not inplemented!".format(ndim))




	#------------------------------------------------------------------------------------------
	# Set microstructure
	#------------------------------------------------------------------------------------------

	def set_Microstructure(self, Phase):
		if type(Phase) is str:
			Image = misc.imread(Phase, True, mode='L')
			Phase = 1.*(Image < Image.mean())

		# Set Lamé coefficients
		lmbda, mu = Function(self.Dh), Function(self.Dh)
		set_coefficient(lmbda, self.lmbda_I, self.lmbda_M, Phase)
		set_coefficient(mu,    self.mu_I,    self.mu_M,    Phase)

		E = self.MacroStrain
		f = self.Source

		# Set operator
		uh, vh = TrialFunction(self.Vh), TestFunction(self.Vh)
		Form = inner(sigma(uh, lmbda, mu, E), grad(vh))*dx - inner(f, vh)*dx
		a, L = lhs(Form), rhs(Form)
		t0 = time()
		A = assemble(a)
		self.rhs = assemble(L)
		# if self.verbose: print('Assemble time :', time() - t0)
		# for bc in self.BCs: bc.apply(A)
		# A.view()
		# print(as_backend_type(A).mat().view())

		# b = as_backend_type(self.rhs).vec()

		# x = b.duplicate()
		# b.copy(x)

		# A = as_backend_type(A).mat()
		# A.setUp()
		
		# from petsc4py import PETSc

		# ksp = PETSc.KSP().create()
		# ksp.setOperators(A)
		# ksp.setType('cg')
		# pc = ksp.getPC()
		# pc.setType('gamg')
		# ksp.setFromOptions()
		# ksp.setTolerances(rtol=1.e-6, atol=1.e-6, divtol=1.e6, max_it=1000)
		# ksp.setInitialGuessNonzero(False)

		# ksp.solve(b, x)
		# exit()





		self.Asolver.set_operator(A)
		self.Phase = Phase
		self.lmbda, self.mu = lmbda, mu


	#------------------------------------------------------------------------------------------
	# Solve the problem (in displacements)
	#------------------------------------------------------------------------------------------

	def solve(self):
		# try:
		u = Function(self.Vh)
		niter = self.Asolver.solve(u.vector(), self.rhs)
		# 	print("FEM: Krylov iterations number =", niter)
		# except:
		# 	print("FEM: Krylov solver failed to converge.")
		return u


	#------------------------------------------------------------------------------------------
	# Quantities of Interest
	#------------------------------------------------------------------------------------------

	# Get Strains
	def compute_Strain(self, u):
		FE = FiniteElement("DG", self.mesh.ufl_cell(), 0)
		FE_list = [FE] * self.nTensElt
		ME = MixedElement(FE_list)
		Vt = FunctionSpace(self.mesh, ME)
		eps = strain(u, self.MacroStrain)
		if self.ndim==2:
			eps = as_vector([eps[0,0], eps[1,1], eps[0,1]])
		elif self.ndim==3:
			eps = as_vector([eps[0,0], eps[1,1], eps[2,2], eps[1,2], eps[0,2], eps[0,1]])
		else:
			msgDimError(self.ndim)
		eps = project(eps, Vt)
		return eps

	# Get Elastic energy over a subdomain
	def compute_Q0(self, u, subdomain):
		dO = dx(1, domain=self.mesh, subdomain_data=subdomain)
		Q0 = assemble(energy(u, self.lmbda, self.mu)*dO)
		return Q0

	# Get Deviatoric energy over a subdomain
	def compute_Q1(self, u, subdomain):
		dO = dx(1, domain=self.mesh, subdomain_data=subdomain)
		V  = assemble(1*dO)
		s  = devStress(u, self.lmbda, self.mu, self.MacroStrain)
		Q1 = assemble(tr(s.T*s) *dO) / V / self.mu_M
		return Q1

	# Get traction over a subdomain
	def compute_Q2(self, u, subdomain):
		dO = dx(1, domain=self.mesh, subdomain_data=subdomain)
		V  = assemble(1*dO)
		Q2 = assemble(traction(u, self.lmbda, self.mu, self.MacroStrain)*dO) / V
		return Q2


	def compute_MacroStiffness(self, Phase=None):
		if Phase is not None: self.set_Microstructure(Phase)

		t = time()
		self.set_loading(type='stress', theta=0) ### theta=0
		self.solve()
		Strain0 = self.Strain
		Stress0 = self.Stress

		self.export("./load0")

		self.set_loading(type='stress', theta=pi/2) ### theta=pi/2
		self.solve()
		Strain1 = self.Strain
		Stress1 = self.Stress

		self.export("./load1")

		if self.verbose>=2: print('Solve time:', time()-t)

		### Homogenized strains
		E0 = Strain0.reshape([-1,self.nTensElt]).mean(axis=0)
		E1 = Strain1.reshape([-1,self.nTensElt]).mean(axis=0)
		S0 = Stress0.reshape([-1,self.nTensElt]).mean(axis=0)
		S1 = Stress1.reshape([-1,self.nTensElt]).mean(axis=0)
		if self.verbose:
			if self.ndim==2:
				S0_imp = np.array([1, 1, 0])/sqrt(2)
				S1_imp = np.array([0, 0, -1])/sqrt(2)
			else:
				S0_imp = np.array([1, 1, 1, 0, 0, 0])/sqrt(3)
				S1_imp = np.array([0, 0, 0, 0, 0, -1])/sqrt(2)
			print('MeanStress check: ', np.linalg.norm(S0-S0_imp), np.linalg.norm(S1-S1_imp))

		### Homogenized Young's modulus
		K = 1/self.ndim * self.Trace(S0) / self.Trace(E0)
		G = 1/2 * S1[-1] / E1[-1]
		Young = 9*K*G/(3*K+G)
		return Young



#######################################################################################################
# Unified execution function for computing Quantities of Interest
#######################################################################################################
	
def compute_QoIs(config_file, sample_file, mesh=None, subdomain=None):

	config = importlib.import_module(config_file)

	pb = LinearElastisityProblem(config, mesh, subdomain)

	pb.set_Microstructure(sample_file)

	u = pb.solve()
	W = pb.compute_Q0(u)

	# print('Elastic energy = {0}'.format(W))

	return W


#######################################################################################################
# Run as main (rather for testing)
#######################################################################################################

if __name__ == "__main__":

	config = importlib.import_module(sys.argv[1])

	pb = LinearElastisityProblem(config)

	sample_file = config.samples_dir + config.samplename + '_0.xml'
	pb.solve(sample_file)























