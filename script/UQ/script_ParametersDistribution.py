
import numpy as np
import torch
import pickle



inputfolder  = "./"
outputfolder = "./"

ls_cells = 1+np.arange(6)

parameters_names = [
    # 'tau', 
    # 'par_tau', 
    'par_alpha',
    # 'Covariance.log_nu',
    'Covariance.log_corrlen',
    'Structure.par_thickness',
    # 'noise_quantile'
]


"""
==================================================================================================================
Load parameters data from files
==================================================================================================================
"""

random_parameters_dict = { name : [] for name in parameters_names }

for icell in ls_cells:
    filename = inputfolder + "cell_{0:d}/inferred_parameters".format(icell)
    named_parameters_dict = torch.load(filename)

    for name, value in named_parameters_dict.items():
        if name in parameters_names:
            random_parameters_dict[name].append(value.item())

"""
==================================================================================================================
Save statistics
==================================================================================================================
"""

for name, data in random_parameters_dict.items():
    x   = np.array(data)
    loc = x.mean()
    std = x.std()
    random_parameters_dict[name] = [loc, std]
    print(name, [loc, std])

torch.save(random_parameters_dict, outputfolder + "random_parameters_config")

