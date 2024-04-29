import unittest

import torch as t
from gpytorch.distributions import MultivariateNormal
from math import sqrt, exp

from src.utils.metrics import get_metric_fn

t.set_default_dtype(t.float64)


class TestHellingerDistance(unittest.TestCase):
    def test_hellinger_distance(self):
        # This sets up two MultivariateNormal distributions with known parameters.
        # The test checks if the hellinger_distance function computes the distance
        # close to the expected value based on the parameters.
        metric_fn = get_metric_fn('HELLINGER')

        num_points = 5  # Using a small number for simplicity of the test
        mean1 = t.zeros(num_points)
        mean2 = t.zeros(num_points)
        mean2[0] = 0.1  # Small difference to calculate the distance

        covariance = t.eye(num_points)

        prediction_1 = MultivariateNormal(mean1, covariance_matrix=covariance)
        prediction_2 = MultivariateNormal(mean2, covariance_matrix=covariance)

        computed_hellinger_distance = metric_fn(prediction_1=prediction_1, prediction_2=prediction_2)

        # With same covariance matrices, the expected distance formula simplifies
        expected_distance = sqrt(1 - exp(-0.1 ** 2 / 8))

        # Assert that the computed distance is within a small tolerance of the expected distance
        self.assertAlmostEqual(computed_hellinger_distance.item(), expected_distance, places=4)


if __name__ == "__main__":
    unittest.main()
