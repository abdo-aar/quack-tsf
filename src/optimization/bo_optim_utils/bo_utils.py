from ax.service.utils.instantiation import TParameterRepresentation
from typing import List

from typing import Tuple

import gpytorch
from gpytorch.kernels import RBFKernel, PeriodicKernel, MaternKernel, RQKernel
from gpytorch.likelihoods.likelihood import Likelihood
from omegaconf import DictConfig
from qiskit_ibm_provider import IBMProvider

from src.models.gp_models.gp_model import GPModel
from src.models.quantum_kernels.iqp_kernel import IQPKernel
from src.data.data_utils import create_train_test_series, pre_process_standardize_time_series
from src.data.synthetic.synthetic_data_generators import trend_periodic_time_series
import torch
from gpytorch.constraints import Interval
from omegaconf import OmegaConf


def get_search_space(cfg: DictConfig) -> List[TParameterRepresentation]:
    """
    Prepares the search space to be used in the Bayesian Optimization process

    :param cfg: a config dictionary
    :return search_space_list: list of dictionaries representing parameters in the experiment search space.
    """
    cfg1 = cfg.gp_specs.global_params
    cfg2 = cfg.model.kernel_params

    OmegaConf.set_struct(cfg1, False)
    OmegaConf.set_struct(cfg2, False)

    search_space_cfg = OmegaConf.merge(cfg1, cfg2)

    search_space_list = []
    for key, item in search_space_cfg.items():
        if item.type == "range":
            search_space = {"name": key, "type": item.type, "bounds": [item.bounds[0], item.bounds[1]],
                            "log_scale": item.log_scale, "value_type": item.value_type}

        elif item.type == "fixed":
            search_space = {"name": key, "type": item.type, "value": item.value, "value_type": item.value_type}
        elif item.type == "choice":
            search_space = {"name": key, "type": item.type,
                            "values": [item['values'][i] for i in range(len(item['values']))],
                            "value_type": item.value_type, "is_ordered": item.is_ordered}
        else:
            raise ValueError("invalid search space parameter type")
        search_space_list.append(search_space)
    return search_space_list


def load_gp_likelihood(cfg: DictConfig, train_x: torch.Tensor, train_y: torch.Tensor) -> Tuple[GPModel, Likelihood]:
    """
    Prepares the gp model to use with its likelihood

    :param cfg: a DictConfig object holding necessary parameters for model instantiation
    :param train_x: training data points used, of shape (num_points, num_features)
    :param train_y: training data labels used, of shape (num_points,)
    :return: gp_model, likelihood
    """
    noise_constraint = Interval(lower_bound=cfg.gp_specs.global_params.noise_level.bounds[0],
                                upper_bound=cfg.gp_specs.global_params.noise_level.bounds[1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint)

    if cfg.model.name in ["iqp_kernel", "iqp_kernel_qq"]:
        gp_model = load_iqp_gp(cfg, likelihood, train_x, train_y)
    elif cfg.model.name in ["rbf_kernel", "periodic_kernel", "matern_kernel", "rqk_kernel", "spmix_kernel"]:
        gp_model = load_classical_gp(cfg, likelihood, train_x, train_y)
    else:
        raise ValueError(f"Model of type {cfg.model.name} is not supported")

    return gp_model, likelihood


def load_iqp_gp(cfg: DictConfig, likelihood: Likelihood, train_x: torch.Tensor, train_y: torch.Tensor) -> GPModel:
    """
    Loads an IQP-based GP model

    :param cfg: config used
    :param likelihood: likelihood model used
    :param train_x: training data points used, of shape (num_points, num_features)
    :param train_y: training data labels used, of shape (num_points,)
    :return: iqp-based gp model
    """
    # Set the IBMQ provider if we are running on the quantum computer
    provider = IBMProvider(instance=cfg.model.instance) if cfg.model.instance else None

    likelihood = likelihood.to(cfg.model.iqp_torch_device)
    quantum_kernel = IQPKernel(num_features=train_x.shape[-1], torch_device=cfg.model.iqp_torch_device,
                               mod=cfg.model.mod, provider=provider, backend=cfg.model.backend,
                               qq_inter=cfg.model.qq_inter).to(cfg.model.iqp_torch_device)
    gp_model = GPModel(train_x, train_y, likelihood, kernel=quantum_kernel).to(cfg.model.iqp_torch_device)
    return gp_model


def load_classical_gp(cfg: DictConfig, likelihood: Likelihood, train_x: torch.Tensor, train_y: torch.Tensor):
    """
    Loads a classical kernel-based GP model

    :param cfg: config used
    :param likelihood: likelihood model used
    :param train_x: training data points used, of shape (num_points, num_features)
    :param train_y: training data labels used, of shape (num_points,)
    :return: classical-based gp model
    """

    if cfg.model.name == "rbf_kernel":
        kernel = RBFKernel()
    elif cfg.model.name == "periodic_kernel":
        kernel = PeriodicKernel()
    elif cfg.model.name == "matern_kernel":
        kernel = MaternKernel()
    elif cfg.model.name == "rqk_kernel":
        kernel = RQKernel()
    else:
        raise ValueError(f"Kernel {cfg.model.name} is not supported")

    kernel = kernel.to(cfg.bo_optim.specs.bo_torch_device)  # Use the device used in the BO process
    gp_model = GPModel(train_x, train_y, likelihood, kernel=kernel).to(cfg.bo_optim.specs.bo_torch_device)

    return gp_model


def load_data(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads data given the config input

    :param cfg: configuration used
    :return series, train_x, train_y, test_x, test_y: tuple of the series generated + train and test tensors
    """
    # Load the specified time series
    if cfg.data.name == "synthetic":
        if cfg.data.type == "trend_periodic":
            series = trend_periodic_time_series(num_time_steps=cfg.data.specs.num_time_steps,
                                                noise_level=cfg.data.specs.noise_level,
                                                trend_changes=cfg.data.specs.trend_changes,
                                                seed=cfg.data.specs.gen_seed,
                                                base_scale1=cfg.data.specs.base_scale1,
                                                base_scale2=cfg.data.specs.base_scale2,
                                                period1=cfg.data.specs.period1, )
        else:
            raise ValueError(f"Synthetic data of type {cfg.data.type} is not supported")
    else:
        raise ValueError(f"Synthetic data of name {cfg.data.name} is not supported")

    if cfg.model.standardize:
        # Standardize and train test split the data
        standardized_time_series, _, _ = pre_process_standardize_time_series(series)
    else:
        standardized_time_series = series

    train_x, train_y, test_x, test_y = create_train_test_series(series=standardized_time_series,
                                                                window_length=cfg.bo_optim.train_hparams.window_length,
                                                                train_overlap=cfg.bo_optim.train_hparams.train_overlap,
                                                                test_steps_gap=cfg.data.train_test_specs.test_steps_gap,
                                                                train_ratio=cfg.data.train_test_specs.train_ratio,
                                                                torch_device=cfg.bo_optim.specs.bo_torch_device)

    return series, train_x, train_y, test_x, test_y
