from gpytorch.kernels import RBFKernel, PeriodicKernel, MaternKernel, RQKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood

import torch

from src.models.gp_models.gp_model import GPModel
from src.models.quantum_kernels.iqp_kernel import IQPKernel


def evaluate_gp(parameterization: dict, train_x: torch.Tensor, train_y: torch.Tensor,
                gp_model: GPModel, likelihood: Likelihood, mll: MarginalLogLikelihood) -> dict:
    """
    Expensive-to-evaluate function. This computes the Marginal Log Likelihood p(train_y | train_x).

    :param parameterization: a dictionary of the structure {"param1": v1, "param2": v2, "param3": v3, ...}
    :param train_x: training points dataset of shape (num_points, num_features)
    :param train_y: training labels dataset of shape (num_features,)
    :param gp_model: gaussian process model used
    :param likelihood: likelihood model used
    :param mll: marginal log likelihood instance

    :return: a dictionary containing the value of the mll + 0.0 => setting a noiseless observations approach
    """
    # Set up the gp with the given parameterization
    parameterize_gp(parameterization, gp_model, likelihood)

    # Get the device in which the gp model is hosted. This can be different that train_x.device !
    gp_device = next(gp_model.parameters()).device

    gp_model.train()  # To speed up the computations: this does not lead to gradient computation
    with torch.no_grad():  # So that we do not compute any gradients of our expensive-to-compute function
        output = gp_model(train_x.to(gp_device))
        marginal_ll = mll(output, train_y.to(gp_device))

    return {"mll": (marginal_ll.to(train_x.device), 0.0)}  # 0.0 => Noiseless observations


def parameterize_gp(parameterization: dict, gp_model: GPModel, likelihood: Likelihood):
    """
    Sets up the given parameterization given the kernel used

    :param parameterization: the given parameterization
    :param gp_model: the used gp model
    :param likelihood: the used gp model
    """
    likelihood.noise = parameterization["noise_level"]
    gp_model.mean_module.constant = parameterization["mean"]

    if isinstance(gp_model.covar_module, IQPKernel):
        gp_model.covar_module.alpha = parameterization["alpha"]
    elif isinstance(gp_model.covar_module, RBFKernel):
        gp_model.covar_module.lengthscale = parameterization["lengthscale"]
    elif isinstance(gp_model.covar_module, PeriodicKernel):
        gp_model.covar_module.period_length = parameterization["period_length"]
        gp_model.covar_module.lengthscale = parameterization["lengthscale"]
    elif isinstance(gp_model.covar_module, MaternKernel):
        gp_model.covar_module.nu = parameterization["nu"]
        gp_model.covar_module.lengthscale = parameterization["lengthscale"]
    elif isinstance(gp_model.covar_module, RQKernel):
        gp_model.covar_module.nu = parameterization["alpha"]
        gp_model.covar_module.lengthscale = parameterization["lengthscale"]
    else:
        raise ValueError(f"GP model of kernel type `{type(gp_model.covar_module)}`, is not supported")
