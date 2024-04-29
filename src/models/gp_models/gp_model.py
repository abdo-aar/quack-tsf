import gpytorch
from gpytorch.kernels import Kernel
from torch import Tensor
from gpytorch.likelihoods.likelihood import Likelihood


# Use the simplest form of GP model, exact inference
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood, kernel: Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
