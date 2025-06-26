import numpy as np
import torch
import openturns as ot
import nlopt
from source.Calibration.DistanceMeasure import DistanceMeasure
from source.StatisticalDescriptors import autocorrelation, spec_area, interface, correlation_curvature, curvature, num_curvature


import openturns.viewer as viewer
from matplotlib import pylab as plt

# Descriptors
def compute_descriptors(sample):
    X  = sample
    I  = interface(sample)
    vf = X.mean()
    sa = I.mean()
    S2 = autocorrelation(X - X.mean())
    I2 = autocorrelation(I - I.mean())
    _l = int(X.shape[0] * 0.04)
    d3 = 0.5*(S2[0,:_l]+S2[:_l,0])
    d4 = 0.5*(I2[0,:_l]+I2[:_l,0])
    # d3 = d3/d3[0]
    # d4 = d4/d4[0]
    # d3 = S2
    # d4 = I2
    return [vf, sa, d3, d4]

def descriptors_discrepancy(sd, sd_ref):
    discrepancy = 0.
    weights = [1.]*4
    weights = [0., 0., 1., 1.]
    for i in range(len(sd_ref)):
        delta_i  = (sd[i] - sd_ref[i]).square().mean()
        norm_i   = sd_ref[i].square().mean()
        weight_i = weights[i]
        discrepancy = discrepancy + (delta_i / norm_i) * weight_i
    return discrepancy.item()




def calibrate_material_cracks(RM, data_samples, **kwargs):
    """
    Arguments:
        RM                  :   Random material instance to be calibrated
        data_samples        :   List of samples of the target material to fit
    kwargs (optional):
        parameters_bounds   :   (optional) Dictionary (keys: parameter names; values: parameter bounds)
                                Only listed parameters are calibrated (within given bounds).
                                If None, all parameters are calibrated within default bounds.
        n_calls             :   Number of calls of the objective function (beside initial DoE).
                                Default: 100*dim
    """   
    print("NLopt package")

    model_parameters = {
        "ell"   : RM.Covariance.corrlen,    # correlation length of the gaussian noise
        "nu"    : RM.Covariance.nu,         # regularity of the gaussian noise
        "alpha" : RM.alpha,                 # noise level
        "tau"   : RM.Structure.thickness,   # crack thickness parameter
    }
    all_model_parameters_dict  = dict(RM.named_parameters())

    parameters_bounds = kwargs.get('parameters_bounds', None)
    if parameters_bounds is None:
        ### Default parameters bounds
        parameters_bounds = {
            "ell"   : [1.e-4, 0.1],
            "nu"    : [0.5, 2.5],
            "alpha" : [0., 1.],
            "tau"   : [0., 0.25],
        }

    parameter_names  = list(parameters_bounds.keys())
    parameter_bounds = list(parameters_bounds.values())
    dim = len(parameter_names)

    n_calls = kwargs.get('n_calls', 100*dim)
    maxtime = kwargs.get('maxtime', -1)
    discrepancy_tolerance = kwargs.get('discrepancy_tolerance', 1.e-3)
    n_samples = kwargs.get('n_samples', 5)
    seeds = [ np.random.randint(100) for i in range(n_samples) ]

    lowerbound, upperbound = list(zip(*parameter_bounds))
    shift = np.array(lowerbound)
    scale = np.array(upperbound) - np.array(lowerbound)
    assert all(scale > 0)
    lowerbound, upperbound = [0.]*dim, [1.]*dim


    sd_data_list = [ compute_descriptors(sample) for sample in data_samples ]
    sd_data = [ torch.stack(sd_i).mean(dim=0) for sd_i in zip(*sd_data_list)]

    def set_parameters(x):
        for i, pname in enumerate(parameter_names):
            # all_model_parameters_dict[pname].data[0] = x[i] * scale[i] + shift[i]
            val = x[i] * scale[i] + shift[i]
            if pname=="ell":
                RM.Covariance.corrlen = val
                RM.update() ### update correlation operator of the gaussian random field
            if pname=="nu":
                RM.Covariance.nu = val
                RM.update() ### update correlation operator of the gaussian random field
            if pname=="alpha":
                RM.alpha = val
            if pname=="tau":
                RM.Structure.thickness = val


    def print_parameters():        
        print(f"ell={RM.Covariance.corrlen[0].item():6.4f}, nu={RM.Covariance.nu.item():6.4f}, alpha={RM.alpha.item():6.4f}, tau={RM.Structure.thickness.item():6.4f}")

    RM.icount = 0
    RM.best_x = None
    RM.best_y = 1.e10

    def objectiveFunction(x, grad):

        # if RM.best_y < discrepancy_tolerance:
        #     raise nlopt.ForcedStop('Discrepancy tolerance is reached!')

        RM.icount += 1
        ### Set model parameters
        set_parameters(x)        

        print("---------------------------------") 
        print(f"Call {RM.icount}")
        print_parameters()

        ### Compute model descriptors and descrepancy with the data 
        with torch.no_grad():
            sd_list = []
            discrepancy_list = []
            max_n_samples = n_samples
            for i in range(max_n_samples):
                RM.seed(seeds[i])
                sample = RM.sample()
                sd_sample = compute_descriptors(sample)
                sd_list.append(sd_sample)
                # n_samples = len(sd_list)

                sd = [ torch.stack(sd_i).mean(dim=0) for sd_i in zip(*sd_list)]

                discrepancy = descriptors_discrepancy(sd, sd_data)
                discrepancy_list.append(discrepancy)

                # k_max = 10
                # tol = 0.05
                # if n_samples>k_max:
                #     stop_criterion = True
                #     for k in range(k_max):
                #         eps_k = np.abs((discrepancy - discrepancy_list[-2-k])/discrepancy)
                #         if eps_k>tol:
                #             stop_criterion = False
                #     if stop_criterion: break

        print(f"Discrepancy: {discrepancy}")
        print("---------------------------------") 

        if discrepancy < RM.best_y:
            RM.best_x = [ float(x[i] * scale[i] + shift[i]) for i in range(dim) ]
            RM.best_y = discrepancy

        print(f"Best x: {RM.best_x}")
        print(f"Best y: {RM.best_y}")
        print("---------------------------------") 

        # return [discrepancy]
        # return [ np.log(discrepancy) ]
        return discrepancy

    # objectiveFunction = ot.PythonFunction(dim, 1, objectiveFunction)

    # problem = ot.OptimizationProblem()
    # problem.setObjective(objectiveFunction)
    # problem.setMinimization(True)
    # bounds = ot.Interval(lowerbound, upperbound)
    # problem.setBounds(bounds)

    # # algo = ot.NLopt(problem, "GN_DIRECT")
    # # algo = ot.NLopt(problem, "GN_ISRES")
    # algo = ot.NLopt(problem, "GN_MLSL")
    # algo.setLocalSolver(ot.NLopt("LN_COBYLA"))
    # algo.setStartingPoint([0.] * dim)
    # algo.setMaximumCallsNumber(n_calls)

    # algo.run()
    # result = algo.getResult()
    # niter  = result.getIterationNumber()
    # xopt   = result.getOptimalPoint()
    # yopt   = result.getOptimalValue()
    
    opt = nlopt.opt(nlopt.GD_MLSL_LDS, dim)    
    # opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, dim)
    # opt = nlopt.opt(nlopt.GD_STOGO_RAND, dim)
    opt.set_local_optimizer(nlopt.opt(nlopt.LN_COBYLA, dim))
    # opt = nlopt.opt(nlopt.LN_COBYLA, dim)
    opt.set_lower_bounds(lowerbound)
    opt.set_upper_bounds(upperbound)
    opt.set_min_objective(objectiveFunction)
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(n_calls)
    opt.set_maxtime(maxtime)
    opt.set_stopval(discrepancy_tolerance)

    xopt = opt.optimize([0.] * dim)
    yopt = opt.last_optimum_value()
    niter= opt.get_numevals()

    print("optimum at ", xopt)
    print("minimum value = ", yopt)
    print("number of iterations = ", niter)
    # print("result code = ", opt.last_optimize_result())

    set_parameters(xopt)
    xopt_ = [ xopt[i] * scale[i] + shift[i] for i in range(dim) ]
    return xopt_



###########################################################################################
