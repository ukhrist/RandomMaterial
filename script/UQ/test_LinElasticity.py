
from math import pi, sqrt
import numpy as np
import pickle
import torch
from tqdm import tqdm


import vtk
from vtk.util.numpy_support import vtk_to_numpy

from source.StructureFields.SupportedMaterials import LatticeMaterial
from source.Kernels.MaternKernel import MaternKernel
from source.RandomParameters import RandomParameters
from source.LinearElasticity.LinearElasticity_FourierBased import LinearElastisityProblem_FB as LinearElasticityProblem
# from source.LinearElasticity.LinearElasticity_Fourier_torch import LinearElastisityProblem_Fourier as LinearElasticityProblem_torch

inputfile    = "octettruss_surrogate_cell1.vtk"
inputfolder  = "./"
outputfolder = inputfolder

nsamples = 100


config = {
#################################################
#                   General
#################################################

    'dev'               :   True,
    'verbose'           :   True,
    'debug'             :   True,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   False,

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
    'alpha'             :   0, #0.02,
    'thickness'         :   0.17,
    'noise_sparsity'    :   0,
    'nodal_pores'       :   False,

#################################################
#                   Sampling
#################################################

    ### Number of samples
    'nsamples'  :   nsamples, 

    ### System paths for export
    'samples_dir'       :   outputfolder + 'samples/',
    'samples_img_dir'   :   outputfolder + 'samples_png/',
    'samples_vtk_dir'   :   outputfolder + 'samples_vtk/',
    'samplename'        :   'sample',


#################################################
#                   Forward problem
#################################################

    'Problem_to_solve'      :   'LinearElasticity',

    # Linear elasticity settings
    'Young_modulus_I'   :   0.,
    'Poisson_ratio_I'   :   0.2,
    'Young_modulus_M'   :   400.,
    'Poisson_ratio_M'   :   0.3,

    # Fourier solver settings
    'n_terms_trunc_PeriodizedGreen' :   100,

    # Reference stiffness factor
    'ref_stiff_factor'  :   2,

    # 2D stress-strain type
    # 'plane_stress_strain'   :   'strain',

    # Loading
    'loading_type'  :   'stress', #'strain' #'stress'
    # 'MacroTensor'   :   [1/sqrt(2), 1/sqrt(2), 0] + [0, 0, 0],
    # MacroTensor = [1, 1, 0] + [0, 0, 0]
    # Source      = ('0.', '-9.8')
    'nproc'         :   1,
    'tol'           :   1.e-4, ### Krylov solver tolerance

    ### FEM settings
    'enable_fem'        :   False,
    'element_degree'    :   1,
    'mesh_file'         :   None,


#################################################
#     UQ
#################################################

    'write_to_logfile'  :   True,
    'draw_plots'        :   False,

    'qois_dir'          :   outputfolder,
    'fit_type'          :   None, # 'Normal', 'LogNormal' or None

}



# parameters_config = {   ### (FORMAT)  parameter.name  :   [loc, std],
#     # 'tau', 
#     # 'par_tau', 
#     'par_alpha'                 :   [loc, std],
#     # 'Covariance.log_nu',
#     'Covariance.log_corrlen'    :   [loc, std],
#     'Structure.par_thickness'   :   [loc, std],
#     # 'noise_quantile'
# }



# pb_torch = LinearElasticityProblem_torch(**config)

RM = LatticeMaterial(**config)
pb = LinearElasticityProblem(**config)

reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(inputfolder+inputfile)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()


data = reader.GetOutput()
dim = data.GetDimensions()
shape = list(dim)
vtk_array = data.GetPointData().GetArray('field')

# VTK to Numpy
my_numpy_array = vtk_to_numpy(vtk_array)

Structure = my_numpy_array.reshape(shape)
Structure = RM.sample()
RM.save_vtk(outputfolder+"sample_test", Structure)

E = pb.compute_MacroStiffness(Structure)

strain = pb.Strain

if config["verbose"]: print(strain)

