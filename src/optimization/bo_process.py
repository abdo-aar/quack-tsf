import gpytorch
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import hydra
import logging

from src.optimization.bo_optim_utils.bo_utils import get_search_space, load_data, load_gp_likelihood
from src.optimization.bo_optim_utils.bo_gp_model import BoGPModel
from src.utils.logging_setup import logging_setup

import torch

from src.utils.settings import PROJECT_ROOT_PATH

# Set the PROJECT_ROOT environment variable before initializing Hydra.
os.environ['PROJECT_ROOT'] = PROJECT_ROOT_PATH

# Suppress logs coming from hydra.experimental.callbacks.PickleJobInfoCallback.
logging.getLogger(f"hydra.experimental.callbacks.PickleJobInfoCallback").setLevel(logging.CRITICAL)
logging.getLogger(f"qiskit").setLevel(logging.CRITICAL)  # Suppress unnecessary logs coming from the Quantum Computer
logging.getLogger(f"json_decoder").setLevel(logging.CRITICAL)  # Suppress unnecessary logs coming from json_decoder

logging.captureWarnings(True)  # Catch warnings by the default logger.


def main_wrapper(config_name: str, instance: str = None, backend: str = None):
    """
    Wraps the main function containing the logic of model optimization.

    :param config_name: the name of the config to use.
    :param instance: The used instance
    :param backend: The used backend
    """

    # configs = directory holding the config file used. It is of a path relative to the script being run.
    # We still need in command line to specify the exact config used, done using GROUP=OPTION.
    @hydra.main(version_base=None, config_path="configs", config_name=config_name)
    def main(cfg: DictConfig) -> None:

        # Set the default torch dtype to be double.
        torch.set_default_dtype(torch.float64)

        hydra_cfg = HydraConfig.get()  # Get the Hydra config.
        logging_setup(cfg, hydra_cfg, handle_logs=True)  # Set up the logger object after hydra's initialization.

        # Load the data depending on the given data config
        series, train_x, train_y, test_x, test_y = load_data(cfg)

        logging.info(f"-------------------------------- Beginning of the BO process --------------------------------")

        logging.info(f"Generated series shape: {series.shape}")
        logging.info(f"Training data shape: {train_x.shape}, {train_y.shape}")
        logging.info(f"Testing data shape: {test_x.shape}, {test_y.shape}")

        # Set the backend in the main config file
        if instance and backend:
            cfg.model.instance = instance
            cfg.model.backend = backend

        # Load the gp_model and likelihood.
        gp_model, likelihood = load_gp_likelihood(cfg, train_x, train_y)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Get the ax_client path.
        ax_client_filename = f"{hydra_cfg.job.name}_ax_client.json"
        ax_client_path = os.path.join(os.getcwd(), ax_client_filename)

        # Instantiate the bo_gp model.
        bo_gp_model = BoGPModel(gp_model=gp_model, likelihood=likelihood, mll=mll,
                                train_x=train_x, train_y=train_y, ax_client_path=ax_client_path)

        # Get the parameters' search space.
        search_space_list = get_search_space(cfg)

        # Fit the gp model.
        bo_gp_model.fit(num_init_trials=cfg.bo_optim.optim_hparams.num_init_trials,
                        num_optimization_trials=cfg.bo_optim.optim_hparams.num_optimization_trials,
                        search_space_list=search_space_list, objective_name=cfg.gp_specs.metric.name,
                        experiment_name=cfg.model.name, random_seed=cfg.bo_optim.optim_hparams.random_seed)

        # Produce predictions.
        pred_filename = f"test_predictions.pickle"  # Get the path to predictions file.
        pred_path = os.path.join(os.getcwd(), pred_filename)

        _ = bo_gp_model.predict(test_x=test_x, pred_path=pred_path)  # Produce and save the predictions.
        logging.info(f"-------------------------------- End of the BO process --------------------------------")

    main()
