"""
This script is an example of calibration procedure of the surrogate cracks material model.
Parameters to be calibrated:
1. Correlation length of the Gaussian noise (ell)
2. Regularity of the Gaussian noise (nu)
2. Noise level (alpha)
3. Crack thickness (tau)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.StructureFields.SupportedMaterials import CracksMaterial
from source.Kernels.MaternKernel import MaternKernel
from source.Calibration.Calibration_Cracks_EGO import calibrate_material_cracks
# from source.Calibration.Calibration_Cracks_NLopt import calibrate_material_cracks
from source.DataManager import read_data3D
from script.Calibration.load_data import load_data_from_npy

###=============================================
### Configuration parameters

fg_verbose           = True
fg_fixed_seed        = True
EXPORTDIR            = "./" ###NOTE: set your own export folder

###=============================================
### Initialize material object
#NOTE: Here we configure and create an initial surrogate cracks material instance,
# which will be further clibrated.
# Parameters (ell, nu, alpha, tau) to be calibrated are also given here, 
# but the values are the placeholders and never used.

config = {
    'ndim'              :   2,
    'grid_level'        :   8,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   1.5,
    'correlation_length':   0.04,
    'Folded_GRF'        :   False,
### Support
    'alpha'             :   0.001,
    'thickness'         :   0.001,
    'Poisson_mean'      :   10,
}
RM = CracksMaterial(**config) ### Create a material instance to be calibrated

###=============================================
### Prepare the experimental data
#NOTE: In this example, a synthetic experimental data is used.
# User is free to load his own data.
# Format of "data_samples" : list of data samples, 
# where each sample is a binary (0 or 1 values) N-D array (image).

RM0 = CracksMaterial(**config)
n_data_samples = 4
data_samples = [ RM0.sample().detach() for i in range(n_data_samples) ]

#NOTE: Note that in this examples, the initial values 
# of parameters (ell, nu, alpha, tau) are the targets (known ground truth).


# Experimental volume fraction (porosity)
#NOTE: Using experimental porosity is an alternative to data samples
experimental_porosity = 0.0195


###=============================================
### Calibration
### Bayesian optimization: Efficent Global Optimization (EGO) from OpenTurns package

if fg_verbose:
    print()
    print("=======================================")
    print(" Surrogate crack material calibration")
    print("=======================================")

config_optim = {
    # Dictionary (keys: parameter names; values: parameter bounds).
    # Parameters are calibrated within given bounds.
    # Non-given parameters are not calibrated and fixed to the initialized values.
    'parameters_bounds' : {        
        # 'ell'   : [1.e-4, 0.1],
        # 'nu'    : [0.5, 3.],
        'alpha' : [0., 0.1],
        "tau"   : [0., 0.1],
    },
    'n_calls' : 200,                    ### Max number of calls of the objective function (beside initial DoE for EGO).
    'n_samples' : 50,                   ### Number of samples in the batch
    'discrepancy_tolerance' :   1.e-3,  ### stopping tolerance
    #
    # Experimental data
    # NOTE: calibration routine works in to modes, depending on data type passed to the 'data' argument:
    #   1. iterable :   data is understood as list of target image samples (2D or 3D)
    #   2. float    :   data is an experimental volume fraction
    # 'data'  :   data_samples,
    'data'  :   experimental_porosity,
}
                                
### Calibration routine
optimized_parameters = calibrate_material_cracks(RM, **config_optim)

if fg_verbose:
    print()
    print("=======================================")
    print("        Optimized parameters")
    print("=======================================")
    print(f"ell   = {list(RM.Covariance.corrlen.detach())}")
    print(f"nu    = {RM.Covariance.nu.item()}")
    print(f"alpha = {RM.alpha.item()}")
    print(f"tau   = {RM.Structure.thickness.item()}")
    print()




###=============================================
###  Expected crack apperture
###=============================================

import openturns as ot
def expected_apperture(alpha, tau):
    xi = (1-alpha)/alpha * tau
    phi = ot.Normal().computePDF(xi)
    Phi = ot.Normal().computeCDF(xi)
    T = 2*tau * (1 + phi / (xi*Phi))
    return T

alpha = RM.alpha.item()
tau   = RM.Structure.thickness.item()
T     = expected_apperture(alpha, tau)

if fg_verbose:
    print()
    print(f"Expected apperture size : {T}")
    print()


###=============================================
###  Verification
###=============================================

if fg_verbose:
    print()
    print("=======================================")
    print(" Verification")
    print("=======================================")

# RM.seed(0) ### Fix random seed

n_samples = config_optim['n_samples']
vf = np.mean([ RM.sample_numpy().mean() for _ in range(n_samples) ])

if fg_verbose:
    print()
    print(f"vf = {vf}")
    print()

filepath = os.path.abspath(os.path.join(EXPORTDIR, "verification_sample"))
sample = RM.sample_numpy()
RM.save_vtk(filepath, sample)
if config['ndim']==2:
    RM.save_png(filepath+".png", sample)

if fg_verbose:
    print()
    print(f"Samples are successfully saved in {filepath}")
    print()

#######################################################




