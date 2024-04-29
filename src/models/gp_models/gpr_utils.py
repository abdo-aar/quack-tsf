import pickle
from typing import Union

import gpytorch
import torch
from torch import Tensor
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.models import ExactGP
from torch.distributions import Distribution
from gpytorch.distributions.multivariate_normal import MultivariateNormal


def train(model: ExactGP, likelihood: Likelihood, training_iter: int, train_x: Tensor, train_y: Tensor, lr: float = 0.1,
          verbose: bool = False):
    """
    Utility that trains a Gaussian Process ExactGP model using a gradient descent algorithm.

    :param model: a GP model
    :param likelihood: a likelihood instance
    :param training_iter:  number of training iterations
    :param train_x: training points of shape (batch_size, dim)
    :param train_y: training labels of shape (batch_size,)
    :param lr: the learning rate
    :param verbose: boolean when set to true displays the training progress
    :return: None.
    """
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Initial parameters display
    if verbose:
        output_str = (f'Initial parameters - noise: {model.likelihood.noise.item():.3f}'
                      f'   mean_constant: {model.mean_module.raw_constant.item():.3f}')

        # Append parameter information to the output string
        for name, param in model.covar_module.named_parameters():
            # Replace 'raw_' with '' in the parameter name for display
            if name.startswith('raw_'):
                # Derive the property name by removing 'raw_' prefix
                property_name = name[4:]  # Remove 'raw_' prefix to get the property name
                transformed_param = getattr(model.covar_module, property_name)
                output_str += f'   {property_name}: {transformed_param.item()}'
            else:
                output_str += f'   {name}: {param.item()}'

        # Print the consolidated output string
        print(output_str)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        # After training iteration parameters display
        if verbose:
            output_str = (
                f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}   noise: {model.likelihood.noise.item():.3f}'
                f'   mean_constant: {model.mean_module.raw_constant.item():.3f}')

            # Append parameter information to the output string
            for name, param in model.covar_module.named_parameters():
                # Replace 'raw_' with '' in the parameter name for display
                if name.startswith('raw_'):
                    # Derive the property name by removing 'raw_' prefix
                    property_name = name[4:]  # Remove 'raw_' prefix to get the property name
                    transformed_param = getattr(model.covar_module, property_name)
                    output_str += f'   {property_name}: {transformed_param.item()}'
                else:
                    output_str += f'   {name}: {param.item()}'

            # Print the consolidated output string
            print(output_str)


def predict(gp_model: ExactGP, likelihood: Likelihood, test_x: Tensor) -> Union[MultivariateNormal, Distribution]:
    """
    Outputs the GP posterior distribution

    :param gp_model: GP model
    :param likelihood: used likelihood
    :param test_x: test points of shape (batch_size, dim)
    :return: the posterior distribution
    """

    gp_model.eval()
    likelihood.eval()

    # Get the device in which the gp model is hosted. This can be different that train_x.device !
    gp_device = next(gp_model.parameters()).device

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(gp_model(test_x.to(gp_device)))


def save_predictions(observed_pred: MultivariateNormal, pred_path: str):
    """
    Saves the posterior test predictions

    :param observed_pred: a MultivariateNormal object containing the test posterior
    :param pred_path: contains the path where to store the predictions
    """
    # Save the object to disk
    with open(pred_path, 'wb') as pred_file:
        pickle.dump(observed_pred, pred_file)


def load_predictions(pred_path: str) -> MultivariateNormal:
    """
    Loads the posterior test predictions
    :param pred_path: contains the path where predictions are stored

    :return:
    """
    with open(pred_path, 'rb') as pred_file:
        observed_pred = pickle.load(pred_file)

    return observed_pred
