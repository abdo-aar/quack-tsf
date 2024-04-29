import unittest

import torch as t
from gpytorch.distributions import MultivariateNormal
import numpy as np

from src.utils.metrics import get_metric_fn

t.set_default_dtype(t.float64)


class TestLossFunctions(unittest.TestCase):

    def test_crps(self):
        metric_fn = get_metric_fn('MCRPS')

        num_points = 10000
        predicted_mean = t.rand(num_points)
        covariance_matrix = t.eye(num_points) * t.rand(num_points)
        predicted_std = t.sqrt(t.diag(covariance_matrix))

        prediction = MultivariateNormal(predicted_mean, covariance_matrix)
        target = predicted_mean  # Target is exactly the predicted mean

        expected_mcrps = predicted_std * ((np.sqrt(2) - 1) / np.sqrt(np.pi))  # Expected MCRPS
        computed_mcrps = metric_fn(prediction=prediction, target=target)  # Compute MCRPS using our function

        # Check if the computed MCRPS is close to the expected value
        self.assertTrue(t.allclose(t.mean(expected_mcrps), computed_mcrps, atol=1e-4))

    def test_log_likelihood(self):
        metric_fn = get_metric_fn('LogLikelihood')

        num_points = 10000
        mean = 0
        std_dev = 1

        predicted_mean = t.full((num_points,), mean)
        covariance_matrix = t.eye(num_points) * (std_dev ** 2)
        prediction = MultivariateNormal(predicted_mean, covariance_matrix)

        target = t.normal(mean=mean, std=std_dev, size=(num_points,))

        computed_log_likelihood = metric_fn(prediction=prediction, target=target)  # Computed LL

        # Expected LL
        expected_log_likelihood = (-num_points / 2.0) * np.log(2 * np.pi * std_dev ** 2) - \
                                  (1 / (2 * std_dev ** 2)) * t.sum((target - predicted_mean) ** 2)

        # Check if the computed log likelihood is close to the expected value
        self.assertAlmostEqual(computed_log_likelihood.item(), expected_log_likelihood.item(), places=4)


if __name__ == "__main__":
    unittest.main()
