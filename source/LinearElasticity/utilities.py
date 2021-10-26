
import numpy as np
from math import sqrt

#######################################################################################################
#	Elastic moduli
#######################################################################################################

def transfer_YoungPoisson_to_Lame(E, nu):
	lmbda = E*nu/((1+nu)*(1-2*nu))
	mu  = 0.5*E/(1+nu)
	return lmbda, mu

def transfer_Lame_to_YoungPoisson(lmbda, mu):
	if mu==0: return 0, 0
	E  = mu*(3*lmbda + 2*mu)/(lmbda + mu)
	nu = 0.5*lmbda/(lmbda + mu)
	return E, nu

def transfer_YoungPoisson_to_BulkShear(E, nu, d=3):
	# K = 1./3*E/(1-2*nu)
	K = E/((1+nu)*(1-2*nu)) * ((d-2)*nu+1)/d
	G = 0.5*E/(1+nu)
	return K, G

def transfer_BulkShear_to_YoungPoisson(K, mu, d=3):
	# E  = 9*K*mu / (3*K + mu)
	# nu = (3*K - 2*mu) / (6*K+ 2*mu)
	E  = (3*d*K - 2*(3-d)*mu)*mu / (d*K + (d-2)*mu)
	nu = (d*K - 2*mu) / (d*K + (d-2)*mu) / 2
	return E, nu



#######################################################################################################
#	Hashin-Shtrikman bounds
#######################################################################################################

def HashinShtrikmanBounds(vf, bulk=None, shear=None, Young=None, Poisson=None, d=3):
	try: n = vf.size
	except: n = 1
	K_bnd, G_bnd = np.zeros([2,n]), np.zeros([2,n])
	# print('size vf =', vf.size)

	K, G = np.zeros(2), np.zeros(2)
	vf = [1-vf, vf]

	if bulk is not None:
		K[:] = bulk
		G[:] = shear
	else:
		K[0], G[0] = transfer_YoungPoisson_to_BulkShear(Young[0], Poisson[0], d=d)
		K[1], G[1] = transfer_YoungPoisson_to_BulkShear(Young[1], Poisson[1], d=d)

	Km, Kf = vf[0]*K[0] + vf[1]*K[1], vf[1]*K[0] + vf[0]*K[1]
	Gm, Gf = vf[0]*G[0] + vf[1]*G[1], vf[1]*G[0] + vf[0]*G[1]
	# print('size Km =', Km.size)

	for i in range(2):
		if K[i] + 2*G[i] :
			Hi = G[i]*(0.5*d*K[i] + (d+1)*(d-2)/d *G[i])/(K[i] + 2*G[i])
		else:
			Hi = 0
		K_bnd[i,:] = Km - vf[0]*vf[1]*(K[1]-K[0])**2/(Kf + 2*(d-1)/d *G[i])
		G_bnd[i,:] = Gm - vf[0]*vf[1]*(G[1]-G[0])**2/(Gf + Hi)

	K_L, G_L = K_bnd.min(axis=0), G_bnd.min(axis=0)
	K_U, G_U = K_bnd.max(axis=0), G_bnd.max(axis=0)

	return [[K_L, K_U], [G_L, G_U]]


#######################################################################################################
#	Homogenized moduli
#######################################################################################################

def compute_HomogenizedModuli(E0, E1):
	### Only for 2D case
	d=2

	S0 = np.array([1, 1, 0])/sqrt(2)
	S1 = np.array([0, 0, -1])/sqrt(2)

	TrS0 = np.sum(S0[:d])
	TrE0 = np.sum(E0[:,:d], -1)

	K = TrS0/TrE0 / d
	G = 0.5*S1[-1]/E1[:,-1]

	return K, G





#######################################################################################################
#	Stiffness/Compliance Matrix
#######################################################################################################

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
	if E==0:
		return None
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






















