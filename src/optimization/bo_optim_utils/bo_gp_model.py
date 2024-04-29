import logging
import os.path

import torch
from ax import Models
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import TParameterRepresentation

from src.models.gp_models.gp_model import GPModel
from typing import List, Optional

from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood

from src.models.gp_models.gpr_utils import predict, save_predictions
from src.optimization.bo_optim_utils.gp_evaluator import evaluate_gp, parameterize_gp
from gpytorch.distributions.multivariate_normal import MultivariateNormal
import time
import warnings

# Ignore FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


class BoGPModel:
    """
    class that implements utility functions to train a GP's hyperparameters using the function self.train()

    :param gp_model: GP model used
    :param likelihood: Likelihood used
    :param mll: Marginal Log Likelihood used
    :param train_x: training data points of shape (num_points, num_features)
    :param train_y: training data labels of shape (num_points,)
    :param ax_client_path: path to the ax_client location
    """
    gp_model = GPModel
    likelihood = Likelihood
    mll = MarginalLogLikelihood
    train_x = torch.Tensor
    train_y = torch.Tensor
    ax_client = AxClient

    def __init__(self, gp_model: GPModel, likelihood: Likelihood, mll: MarginalLogLikelihood,
                 train_x: torch.Tensor, train_y: torch.Tensor, ax_client_path, verbose_logging=True):
        self.gp_model = gp_model
        self.likelihood = likelihood
        self.mll = mll
        self.train_x = train_x
        self.train_y = train_y
        self.ax_client_path = ax_client_path
        self.verbose_logging = verbose_logging

        if os.path.exists(self.ax_client_path):
            self.ax_client = AxClient.load_from_json_file(filepath=self.ax_client_path,
                                                          verbose_logging=self.verbose_logging)
        else:
            self.ax_client = None

        # Define a wrapped eval function to directly call by optimize()
        def evaluate_function(parameterization: dict):
            return evaluate_gp(parameterization=parameterization, train_x=self.train_x, train_y=self.train_y,
                               gp_model=self.gp_model, likelihood=self.likelihood, mll=self.mll)

        self.evaluate_function = evaluate_function

    def fit(self, num_init_trials: int,
            num_optimization_trials: int,
            search_space_list: List[TParameterRepresentation],
            objective_name: str,
            experiment_name: Optional[str] = None,
            random_seed: int = 1997):
        """
        Fits the GP model using Bayesian optimization

        :param num_init_trials: num training hyperparameters points to generate using the SOBOL sequence
        :param num_optimization_trials: num of hyperparameters to generate during the BO process
        :param search_space_list: search space to cover
        :param objective_name: name of the objective used, typically "mll"
        :param experiment_name: name of the experiment, Optional
        :param random_seed: random seed to use
        """

        logger = logging.getLogger()

        start_time = time.perf_counter()  # Start time

        # Initialize client if it wasn't loaded
        if not self.ax_client:
            # Set up generation strategy
            gs = GenerationStrategy(
                steps=[
                    # 1. Initialization step (does not require pre-existing data and is well-suited for
                    # initial sampling of the search space)
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=num_init_trials,  # How many init trials should be produced
                        min_trials_observed=num_init_trials,  # Enforce that all init trials be met
                    ),
                    # 2. Bayesian optimization step (requires data obtained from previous phase and learns
                    # from all data available at the time of each new candidate generation call)
                    GenerationStep(
                        model=Models.GPEI,  # Specify GP as surrogate model + EI as acquisition function
                        num_trials=num_optimization_trials,  # Num trials to be produced from this step
                    ),
                ]
            )

            self.ax_client = AxClient(generation_strategy=gs, random_seed=random_seed,
                                      verbose_logging=self.verbose_logging)

            self.ax_client.create_experiment(parameters=search_space_list,
                                             name=experiment_name,
                                             minimize=False,
                                             objective_name=objective_name)

        # Sobol sequence generation
        completed_trials = len(self.ax_client.experiment.completed_trials)
        logger.info(msg=f"Num completed trials = {completed_trials}")

        remaining_num_init_trials = num_init_trials - completed_trials

        if remaining_num_init_trials > 0:  # Skip init step if init trials have already been done
            logger.info(msg="Sobol sequence generation")
            for i in range(remaining_num_init_trials):
                parameterization, trial_index = self.ax_client.get_next_trial()
                self.ax_client.complete_trial(trial_index=trial_index,
                                              raw_data=self.evaluate_function(parameterization))
                self.ax_client.save_to_json_file(self.ax_client_path)

        # BO process
        completed_trials = len(self.ax_client.experiment.completed_trials)
        remaining_num_bo_trials = (num_init_trials + num_optimization_trials) - completed_trials

        if remaining_num_bo_trials > 0:  # Skip bo step if bo trials have already been done
            logger.info(msg="Bayesian Optimization process")
            for i in range(remaining_num_bo_trials):
                parameterization, trial_index = self.ax_client.get_next_trial()
                self.ax_client.complete_trial(trial_index=trial_index,
                                              raw_data=self.evaluate_function(parameterization))
                self.ax_client.save_to_json_file(self.ax_client_path)

        best_parameters, _ = self.ax_client.get_best_parameters()

        end_time = time.perf_counter()  # End time

        logger.info(msg=f"Best parameters are: {best_parameters}")

        # Log the execution time
        execution_time_seconds = end_time - start_time  # Calculate execution time in seconds
        hours, remainder = divmod(execution_time_seconds, 3600)  # Convert seconds to hh:mm:ss format
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"  # Format the time string
        logger.info(msg=f"Execution time is: {time_str}")

        logger.info(msg="############### Model fitting finished ###############")

    def predict(self, test_x: torch.Tensor, pred_path: str = None) -> MultivariateNormal:
        """
        Produces the predictions with the GP model
        :param test_x: tensor containing the test points, of shape (num_test_pts, num_features)
        :param pred_path: contains the path where to store the predictions

        :return observed_pred: a MultivariateNormal object containing the test posterior
        """
        logger = logging.getLogger()
        logger.info(msg="Preparing the test predictions ...")
        self.set_best_model()
        observed_pred = predict(self.gp_model, self.likelihood, test_x)
        if pred_path:  # Save predictions if there's a path given
            save_predictions(observed_pred, pred_path)
        return observed_pred

    def set_best_model(self, ):
        """
        Loads best GP and likelihood models using the best_parameters coming from the ax_client object
        """
        # If ax_client doesn't exist yet, we'll work with the already existing self.gp_model and self.likelihood models
        if not self.ax_client:
            warnings.warn(message="Working with sub-optimal parameters: ax_client is not created yet !")

        else:
            # Check if experiment is complete or not
            is_init_step = self.ax_client.generation_strategy.current_step.index == 0  # Are we still in the first step
            _, remaining_trials = self.ax_client.generation_strategy.current_step.num_trials_to_gen_and_complete()
            are_trials_remaining = remaining_trials > 0  # Are there any trials remaining. This is important for 2d step
            non_complete_experiment = is_init_step or are_trials_remaining

            if non_complete_experiment:
                warnings.warn(message="Working with sub-optimal parameters: model fitting is not yet complete !")

            best_parameters, _ = self.ax_client.get_best_parameters()

            # Set model parameter
            parameterize_gp(parameterization=best_parameters, gp_model=self.gp_model, likelihood=self.likelihood)
