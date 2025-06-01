import numpy as np
import openturns as ot
import openturns.experimental as otexp
import torch
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
    _l = 20
    _l = 20
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

    n_samples = 5
    seeds = [ np.random.randint(100) for i in range(n_samples) ]

    def objectiveFunction(x):
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
        # return [ np.log(discrepancy) ]
        return [ discrepancy ]

    objectiveFunction = ot.PythonFunction(dim, 1, objectiveFunction)    
    


    ot.Log.Show(ot.Log.NONE)
    listUniformDistributions = [
        ot.Uniform(lowerbound[i], upperbound[i]) for i in range(dim)
    ]
    distribution = ot.JointDistribution(listUniformDistributions)
    sampleSize = 10*dim
    experiment = ot.LHSExperiment(distribution, sampleSize)
    inputSample = experiment.generate()
    outputSample = objectiveFunction(inputSample)
    # outputSample = (outputSample - outputSample.computeMean()) / outputSample.computeStandardDeviation()
    
    covarianceModel = ot.MaternModel([1.0] * dim, [1.0], 1.5)
    basis = ot.ConstantBasisFactory(dim).build()
    kriging = ot.KrigingAlgorithm(inputSample, outputSample, covarianceModel, basis)


    s2 = 1.e-6
    # s2 = (0.05*outputSample.computeStandardDeviation()[0])**2
    print('Noise variance =', s2)
    def noise_variance(x):
        return [s2]
    noise_variance = ot.PythonFunction(dim, 1, noise_variance)

    noiseSample = noise_variance(inputSample)
    kriging.setNoise([x[0] for x in noiseSample])

    kriging.run()

    if dim==1:
        xMin, xMax = 0, 1
        X, Y = kriging.getInputSample(), kriging.getOutputSample()
        graph = plotMyBasicKriging(kriging.getResult(), xMin, xMax, X, Y)
        view = viewer.View(graph)
        plt.show()


    problem = ot.OptimizationProblem()
    problem.setObjective(objectiveFunction)
    bounds = ot.Interval(lowerbound, upperbound)
    problem.setBounds(bounds)

    algo = ot.EfficientGlobalOptimization(problem, kriging.getResult(), noise_variance)
    algo.setMaximumCallsNumber(n_calls)
    algo.setOptimizationAlgorithm(ot.NLopt("GN_DIRECT"))
    # algo.setOptimizationAlgorithm(ot.NLopt("GN_ISRES")) 

    # uniform = ot.JointDistribution([ot.Uniform(0., 1.)] * dim)
    # ot.RandomGenerator.SetSeed(0)
    # init_pop = uniform.getSample(30)
    # solver = ot.Pagmo('gaco')
    # solver.setStartingSample(init_pop)
    # algo.setOptimizationAlgorithm(solver)
    
    

    algo.run()
    result = algo.getResult()
    niter  = result.getIterationNumber()
    xopt   = result.getOptimalPoint()
    yopt   = result.getOptimalValue()



    graph = result.drawOptimalValueHistory()
    # optimum_curve = ot.Curve(ot.Sample([[0, fexact[0][0]], [29, fexact[0][0]]]))
    # graph.add(optimum_curve)
    view = viewer.View(graph, axes_kw={"xticks": range(0, result.getIterationNumber(), 5)})
    plt.show()

    if dim==1:
        kriging_result = algo.getKrigingResult()
        xMin, xMax = 0, 1
        X, Y = kriging_result.getInputSample(), kriging_result.getOutputSample()
        graph = plotMyBasicKriging(kriging_result, xMin, xMax, X, Y, objectiveFunction)
        view = viewer.View(graph)
        plt.show()


    set_parameters(xopt)
    xopt_ = [ xopt[i] * scale[i] + shift[i] for i in range(dim) ]
    return xopt_

sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])



###########################################################################################

def plotMyBasicKriging(gprResult, xMin, xMax, X, Y, objectiveFunction=None, level=0.95):
    """
    Given a kriging result, plot the data, the kriging metamodel
    and a confidence interval.
    """
    samplesize = X.getSize()
    meta = gprResult.getMetaModel()
    graphKriging = meta.draw(xMin, xMax)
    graphKriging.setLegends(["Kriging"])
    # Create a grid of points and evaluate the function and the kriging
    nbpoints = 30
    xGrid = linearSample(xMin, xMax, nbpoints)
    if objectiveFunction is not None:
        yFunction = objectiveFunction(xGrid)
    yKrig = meta(xGrid)
    # Compute the conditional covariance
    # gpcc = otexp.GaussianProcessConditionalCovariance(gprResult)
    # epsilon = ot.Sample(nbpoints, [1.0e-8])
    # conditionalVariance = gpcc.getConditionalMarginalVariance(xGrid) + epsilon
    conditionalVariance = gprResult.getConditionalMarginalVariance(xGrid)
    conditionalSigma = sqrt(conditionalVariance)
    # Compute the quantile of the Normal distribution
    alpha = 1 - (1 - level) / 2
    quantileAlpha = ot.DistFunc.qNormal(alpha)
    # Graphics of the bounds
    epsilon = 1.0e-8
    dataLower = [
        yKrig[i, 0] - quantileAlpha * conditionalSigma[i, 0] for i in range(nbpoints)
    ]
    dataUpper = [
        yKrig[i, 0] + quantileAlpha * conditionalSigma[i, 0] for i in range(nbpoints)
    ]
    # Compute the Polygon graphics
    boundsPoly = ot.Polygon.FillBetween(xGrid.asPoint(), dataLower, dataUpper)
    boundsPoly.setLegend("95% bounds")
    # Validate the kriging metamodel
    metamodelPredictions = meta(xGrid)
    if objectiveFunction is not None:
        mmv = ot.MetaModelValidation(yFunction, metamodelPredictions)
        r2Score = mmv.computeR2Score()[0]
    # Plot the function
    if objectiveFunction is not None:
        graphFonction = ot.Curve(xGrid, yFunction)
        graphFonction.setLineStyle("dashed")
        graphFonction.setColor("magenta")
        graphFonction.setLineWidth(2)
        graphFonction.setLegend("Function")
    # Draw the X and Y observed
    cloudDOE = ot.Cloud(X, Y)
    cloudDOE.setPointStyle("circle")
    cloudDOE.setColor("red")
    cloudDOE.setLegend("Data")
    # Assemble the graphics
    graph = ot.Graph()
    graph.add(boundsPoly)
    if objectiveFunction is not None:
        graph.add(graphFonction)
    graph.add(cloudDOE)
    graph.add(graphKriging)
    graph.setLegendPosition("lower right")
    graph.setAxes(True)
    graph.setGrid(True)
    if objectiveFunction is not None:
        graph.setTitle("Size = %d, R2=%.2f%%" % (samplesize, 100 * r2Score))
    graph.setXTitle("X")
    graph.setYTitle("Y")
    # graph.setLogScale(ot.GraphImplementation.LOGY)
    return graph


def linearSample(xmin, xmax, npoints):
    """Returns a sample created from a regular grid
    from xmin to xmax with npoints points."""
    step = (xmax - xmin) / (npoints - 1)
    rg = ot.RegularGrid(xmin, step, npoints)
    vertices = rg.getVertices()
    return vertices