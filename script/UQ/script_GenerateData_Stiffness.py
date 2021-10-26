
from math import pi, sqrt
import numpy as np
import pickle
import torch
from tqdm import tqdm


from source.RMNet.StructureFields.SupportedMaterials import LatticeMaterial
from source.RMNet.Kernels.MaternKernel import MaternKernel
from source.RMNet.RandomParameters import RandomParameters
from source.RMNet.LinearElasticity.LinearElasticity_FourierBased import LinearElastisityProblem_FB as LinearElasticityProblem
from source.RMNet.LinearElasticity.LinearElasticity_Fourier_torch import LinearElastisityProblem_Fourier as LinearElasticityProblem_torch



outputfile   = "data_stiffness"
outputfolder = "/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/UQ_MacroStiffness/"

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

    'path_to_original_samples'  :   'images/AluminumAlloy_A319-LFC/A319-LFC_alloy_binary',


#################################################
#                   Forward problem
#################################################

    'Problem_to_solve'      :   'LinearElasticity',

    # Linear elasticity settings
    'Young_modulus_I'   :   0.,
    'Poisson_ratio_I'   :   0.3,
    'Young_modulus_M'   :   1.e0,
    'Poisson_ratio_M'   :   0.3,

    # Fourier solver settings
    'n_terms_trunc_PeriodizedGreen' :   100,

    # Reference stiffness factor
    'ref_stiff_factor'  :   1,

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

### Parameters randomizer
random_parameters_config = torch.load(outputfolder+"random_parameters_config")
RP = RandomParameters(random_parameters_config)


# pb_torch = LinearElasticityProblem_torch(**config)

RM = LatticeMaterial(**config)
pb = LinearElasticityProblem(**config)
E_list = [] ### list of macro stiffness samples

### fix random seed if only one sample (for testing)
if nsamples==1: RM.seed(0)

### Generate data
pbar = tqdm(total=nsamples)
pbar.set_description("isample / nsamples");
for isample in range(nsamples):
    if "RP" in locals():
        print("Sample parameters..")
        RP.sample_parameters(RM)
    Structure = RM.sample()
    if nsamples<4:
        RM.save_vtk(outputfolder+"sample_{0:d}".format(isample), Structure)
    E_i = pb.compute_MacroStiffness(Structure)
    E_list.append(E_i)
    pbar.update(1)
del(pbar)

### Save data to file
datafile = open(outputfolder + outputfile + '.pkl', 'wb')
pickle.dump((E_list, config), datafile)

if config["verbose"]: print(E_list)

