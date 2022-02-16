
from math import pi, sqrt
import numpy as np
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.Kernels.MaternKernel import MaternKernel



config = {
#################################################
#                   General
#################################################

    'dev'               :   True,
    'verbose'           :   True,
    'debug'             :   True,
    'export_vtk'        :   False,

    'fg_fixed_seed'         :   False,

#################################################
#            Surrogate Material
#################################################

    'ndim'              :   3,
    'grid_level'        :   7,
    'window_margin'     :   0,
    'vf'                :   0.2,

    ### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   2,
    'correlation_length':   0.04,
    'Folded_GRF'        :   False,

    ### Lattice
    'nCells'            :   1,
    'SizeVoid'          :   0,
    'alpha'             :   0.02,
    'thickness'         :   0.165,
    'nodal_pores'       :   False,


    'fg_random_parameters'  :   False,
    'model_parameters'  :   {   
                                'par_alpha'                 :   [-1.94, 0.046], ### mean, standard deviation (parametrization of alpha)
                                'Covariance.log_corrlen'    :   [-3.21, 0.043],
                                'Structure.par_thickness'   :   [-1.8, 0.018]
                            },

}