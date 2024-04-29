import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from src.data.data_utils import post_process_time_series
from matplotlib.figure import Figure


def generate_plot_with_predictions(series: Tensor, observed_pred: MultivariateNormal, train_x: Tensor, test_x: Tensor,
                                   window_length: int, train_overlap: int, test_steps_gap: int = 1,
                                   train_ratio: float = 0.8, figsize: tuple[float, float] = (15, 8),
                                   post_process: bool = True) -> Figure:
    """
    Plots the train test series against the post-processed predictions

    :param series: original real time series
    :param observed_pred: model predictions
    :param train_x: train windows
    :param test_x: test windows
    :param window_length: w
    :param train_overlap: overlap between successive training windows
    :param test_steps_gap: the gap between two consecutive test label points.
    :param train_ratio: The ratio of the dataset to be used for training (default is 0.8).
    :param figsize: figure size to plot with
    :param post_process: whether post_process the predictions or not
    :return: fig: the generated figure
    """

    # Calculate the indices for the training and testing series
    total_train_windows, total_test_windows = train_x.shape[0], test_x.shape[0]
    total_length = series.shape[0]
    train_test_div_index = int(total_length * train_ratio)

    train_end_index = total_train_windows * (window_length - train_overlap) + window_length

    test_start_index = train_test_div_index
    test_end_index = test_start_index + total_test_windows * test_steps_gap

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(series[:train_end_index + 1].numpy(), label='Training Series', alpha=0.75)
    ax.plot(range(test_start_index, test_start_index + len(series[test_start_index:])),
            series[test_start_index:].numpy(), label='Testing Series', alpha=0.75)

    # Plot predictions and confidence intervals
    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        mean = observed_pred.mean

        if post_process:
            org_mean = series.mean()
            org_std = series.std()

            (lower, upper) = (post_process_time_series(lower, org_mean, org_std),
                              post_process_time_series(upper, org_mean, org_std))
            mean = post_process_time_series(mean, org_mean, org_std)

        # Adjust the indices for plotting predictions to align with the test series
        # test_predictions_indices = test labels indices
        test_predictions_indices = np.arange(test_start_index, test_end_index, test_steps_gap)

        # Plot predictive means as a blue line
        ax.plot(test_predictions_indices, mean.cpu().numpy(), 'b-', label='Predictions')

        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_predictions_indices, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, color='blue',
                        label='Confidence Interval')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('GPR Model Predictions with Training and Testing Series')
    ax.legend()

    # Return the figure object
    return fig


def get_train_test_series(series: Tensor, train_x: Tensor, test_x: Tensor,
                          window_length: int, train_overlap: int,
                          train_ratio: float = 0.8, figsize: tuple[float, float] = (15, 8),) -> Figure:
    """
    Plots the train test series

    :param series: original real time series
    :param train_x: train windows
    :param test_x: test windows
    :param window_length: w
    :param train_overlap: overlap between successive training windows
    :param train_ratio: The ratio of the dataset to be used for training (default is 0.8).
    :param figsize: figure size to plot with
    :return: fig: the generated figure
    """

    # Calculate the indices for the training and testing series
    total_train_windows, total_test_windows = train_x.shape[0], test_x.shape[0]
    total_length = series.shape[0]
    train_test_div_index = int(total_length * train_ratio)

    train_end_index = total_train_windows * (window_length - train_overlap) + window_length

    test_start_index = train_test_div_index

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(series[:train_end_index + 1].numpy(), label='Training Series', alpha=0.75)
    ax.plot(range(test_start_index, test_start_index + len(series[test_start_index:])),
            series[test_start_index:].numpy(), label='Testing Series', alpha=0.75)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Training and Testing Series')
    ax.legend()

    # Return the figure object
    return fig
