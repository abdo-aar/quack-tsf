from typing import Callable

import torch as t

from gpytorch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import norm
import numpy as np

MEAN_METRICS = ['SMAPE', 'WAPE', 'MAPE', 'RMSE', 'MAE', 'MSE']
PROBABILISTIC_METRICS = ['LogLikelihood', 'MCRPS']


def mse_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean Squared Error

    :param prediction: predictions of shape (num_points,)
    :param target: targets of shape (num_points,)
    :return: mse loss
    """
    assert prediction.shape == target.shape
    target = target.to(prediction.device)

    nan_mask = ~target.isnan()

    return t.nn.MSELoss()(prediction[nan_mask], target[nan_mask])


def mae_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean Absolute Error

    :param prediction: predictions of shape (num_points,)
    :param target: targets of shape (num_points,)
    :return: mae loss
    """
    assert prediction.shape == target.shape
    target = target.to(prediction.device)

    nan_mask = ~target.isnan()

    return t.nn.L1Loss()(prediction[nan_mask], target[nan_mask])


def rmse_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Root Mean Squared Error

    :param prediction: predictions of shape (num_points,)
    :param target: targets of shape (num_points,)
    :return: rmse loss
    """
    assert prediction.shape == target.shape
    target = target.to(prediction.device)

    return t.sqrt(mse_loss(prediction, target))


def mape_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean absolute percentage error

    :param prediction: predictions of shape (num_points,)
    :param target: targets of shape (num_points,)
    :return: smape loss
    """
    assert prediction.shape == target.shape
    target = target.to(prediction.device)

    mask_indices = target > 0
    nan_mask = ~target.isnan()
    mask_indices &= nan_mask

    return ((prediction[mask_indices] - target[mask_indices]).abs() / target[mask_indices].abs()).mean()


def smape_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Symmetric mean absolute percentage error

    :param prediction: predictions of shape (num_points,)
    :param target: targets of shape (num_points,)
    :return: smape loss
    """
    assert prediction.shape == target.shape
    target = target.to(prediction.device)

    mask_indices = target > 0
    nan_mask = ~target.isnan()
    mask_indices &= nan_mask

    pred = prediction[mask_indices]
    targ = target[mask_indices]

    return ((pred - targ).abs() / ((pred.abs() + targ.abs()) / 2)).mean()


def wape_loss(prediction: t.Tensor, target: t.Tensor):
    """
    Weighted average percentage error

    :param prediction: predictions of shape (num_points,)
    :param target: targets of shape (num_points,)
    :return: wape loss
    """
    assert prediction.shape == target.shape
    target = target.to(prediction.device)

    nan_mask = ~target.isnan()

    return ((target[nan_mask] - prediction[nan_mask]).abs()).mean() / target[nan_mask].abs().mean()


def mcrps_loss(prediction: MultivariateNormal, target: t.Tensor) -> t.Tensor:
    r"""
    Compute the Mean Continuous Ranked Probability Score (MCRPS) loss given by the following equation:

    .. math::
    \begin{aligned}
        CRPS( \mathcal{N}(\hat{y}_t, \sigma_t^2), y_t) &= \sigma_t \left( \omega_t (2 . \Phi(\omega_t) - 1) +
                                                                 2 . \phi(\omega_t) -
                                                                 \frac{1}{\sqrt{\phi}}
                                                                 \right) \\
                                                \omega &= (y_t - \hat{y}_t) / \sigma_t,
    \end{aligned}

    where y_t is a test point (y = (y_1, ..., y_{T_{test}})^\intercal), \hat{y}_t, \sigma_t are the mean and standard
    deviation predictions at time t, \Phi(\omega_t) is the cumulative normal distribution at omega_t and \phi(\omega) is
    the probability density function at \omega_y of the normal distribution.

    The total Mean CRPS is given as :math: `mcrps = \frac{1}{T_{test}} CRPS(\mathcal{N}(\hat{y}_t, \sigma_t^2), y_t)`.

    :param prediction: MultivariateNormal object representing the predictive distribution.
    :param target: Actual target values as a torch.Tensor.
    :return: Average CRPS loss.
    """
    target = target.to(prediction.mean.device)

    # Extract the mean and standard deviation from the prediction
    predicted_mean = prediction.mean
    predicted_std = prediction.stddev

    assert predicted_mean.shape == target.shape

    # Normalize target
    omega = (target - predicted_mean) / predicted_std
    device = omega.device
    omega = omega.cpu().detach().numpy()
    predicted_std = predicted_std.cpu().detach().numpy()

    # Compute CRPS using the formula for Gaussian distributions
    crps_values = predicted_std * (omega * (2 * norm.cdf(omega) - 1) +
                                   2 * norm.pdf(omega) -
                                   1 / np.sqrt(np.pi))

    # Return the mean CRPS value
    return t.tensor(np.mean(crps_values)).to(device)


def hellinger_distance(prediction_1: MultivariateNormal, prediction_2: MultivariateNormal, ) -> t.Tensor:
    r"""
    Computes the Hellinger distance between the multivariate posterior predictions produced from two models.
    The squared Hellinger distance between two multivariate Gaussian distributions
    P \sim \mathcal{N}(\mathbf{\mu}_1, \Sigma_1) and Q \sim \mathcal{N}(\mathbf{\mu}_2, \Sigma_2) is given by:

    .. math::
    H^2(P, Q) = 1 - \frac{\det(\Sigma_1)^{\frac{1}{4}} \det(\Sigma_2)^{\frac{1}{4}}}{\det\left(\frac{\Sigma_1 +
    \Sigma_2}{2}\right)^{\frac{1}{2}}} \exp\left\{
                                                  -\frac{1}{8} (\mathbf{mu}_1 - \mathbf{mu}_2)^\intercal
                                                  \left(\frac{\Sigma_1 + \Sigma_2}{2}\right)
                                                  (\mathbf{mu}_1 - \mathbf{mu}_2)
                                                \right\}.

    :param prediction_1: MultivariateNormal object representing the predictive distribution of model 1.
    :param prediction_2: MultivariateNormal object representing the predictive distribution of model 2.
    :return distance: The Hellinger distance.
    """
    device = prediction_1.mean.device
    mu1, mu2 = prediction_1.mean, prediction_2.mean.to(device)
    cov1, cov2 = prediction_1.covariance_matrix, prediction_2.covariance_matrix.to(device)

    covariance_matrix_avg = (cov1 + cov2) / 2

    det_cov1 = t.det(cov1).pow(1 / 4)
    det_cov2 = t.det(cov2).pow(1 / 4)
    det_cov_avg = t.det(covariance_matrix_avg).pow(1 / 2)

    exponent_term = (-1 / 8 * (mu1 - mu2).unsqueeze(-1).mT @ t.inverse(covariance_matrix_avg)
                     @ (mu1 - mu2).unsqueeze(-1))

    squared_distance = (1 - (det_cov1 * det_cov2 / det_cov_avg) *
                        t.exp(exponent_term))
    distance = t.sqrt(squared_distance)  # The square root gives the Hellinger distance

    return distance


def mean_hellinger_distance(prediction_1: MultivariateNormal, prediction_2: MultivariateNormal) -> t.Tensor:
    """
    Computes the mean Hellinger distance between predictions produced between two models.
    Each prediction is considered to be a univariate Gaussian distribution at each point.

    :param prediction_1: MultivariateNormal object representing the predictive distribution of model 1.
    :param prediction_2: MultivariateNormal object representing the predictive distribution of model 2.
    :return: the mean Hellinger distance.
    """
    device = prediction_1.mean.device
    # Extract mean and standard deviation for each prediction
    mean1, stddev1 = prediction_1.mean, prediction_1.stddev
    mean2, stddev2 = prediction_2.mean.to(device), prediction_2.stddev.to(device)

    # Ensure standard deviations are not zero (adding a small epsilon if necessary)
    stddev1 = t.clamp(stddev1, min=1e-8)
    stddev2 = t.clamp(stddev2, min=1e-8)

    # Compute Hellinger distance for each pair of univariate Gaussian distributions
    distances = []
    for i in range(len(mean1)):
        term1 = t.sqrt(2 * stddev1[i] * stddev2[i] / (stddev1[i] ** 2 + stddev2[i] ** 2))
        term2 = t.exp(-1 / 4 * ((mean1[i] - mean2[i]) ** 2) / (stddev1[i] ** 2 + stddev2[i] ** 2))
        h_distance = t.sqrt(1 - term1 * term2)
        distances.append(h_distance)

    # Take the mean of all Hellinger distances
    mean_distance = t.mean(t.stack(distances))

    return mean_distance


def log_likelihood(prediction: MultivariateNormal, target: t.Tensor) -> t.Tensor:
    r"""
    Computes the Log likelihood: :math: `p(\hat{y} | f) \sim \mathcal{N})(f, \sigma_n^2)`.

    :param prediction: predictions of shape
    :param target: targets of shape (num_points,)
    :return: log likelihood loss
    """
    target = target.to(prediction.mean.device)

    return prediction.log_prob(target)


def get_metric_fn(metric_name: str) -> Callable:
    """
    Gets a Callable metric function based on its name

    :param metric_name: name of the metric function
    :return: metric_fn
    """

    def metric_fn(**kwargs):
        if metric_name == 'MSE':
            return mse_loss(**kwargs)
        elif metric_name == 'MAE':
            return mae_loss(**kwargs)
        elif metric_name == 'RMSE':
            return rmse_loss(**kwargs)
        elif metric_name == 'MAPE':
            return mape_loss(**kwargs)
        elif metric_name == 'WAPE':
            return wape_loss(**kwargs)
        elif metric_name == 'SMAPE':
            return smape_loss(**kwargs)
        elif metric_name == 'MCRPS':
            return mcrps_loss(**kwargs)
        elif metric_name == 'HELLINGER':
            return hellinger_distance(**kwargs)
        elif metric_name == 'MEAN_HELLINGER':
            return mean_hellinger_distance(**kwargs)
        elif metric_name == 'LogLikelihood':
            return log_likelihood(**kwargs)
        else:
            raise Exception(f'Unknown objective function: {metric_name}')

    return metric_fn
