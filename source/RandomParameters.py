
import numpy as np



"""
==================================================================================================================
Random parameters (Gaussian, scalar)
==================================================================================================================
"""

class RandomParameters(object):

    def __init__(self, params_dict, Model=None) -> None:
        super().__init__()
        self.params_dict = params_dict
        self.params_dict_sample = { name : 0. for name in self.params_dict.keys() }
        self.Model = Model

    def draw_parameters(self):
        for item in self.params_dict.items():
            name     = item[0]
            loc, std = item[1]
            self.params_dict_sample[name] = np.random.normal(loc, std)
        return self.params_dict_sample

    def sample_parameters(self, Model=None):
        if Model is None: Model = self.Model
        named_parameters_dict = self.draw_parameters()
        Model.set_parameters(named_parameters_dict)
