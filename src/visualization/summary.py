import os.path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import torch
from sklearn.decomposition import PCA

from src.data.data_utils import pre_process_standardize_time_series
from src.models.gp_models.gpr_utils import load_predictions
from src.optimization.bo_optim_utils.bo_gp_model import BoGPModel
from src.optimization.bo_optim_utils.bo_utils import load_data, load_gp_likelihood
from omegaconf import OmegaConf, DictConfig
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.utils.metrics import MEAN_METRICS, PROBABILISTIC_METRICS, get_metric_fn
from src.visualization.gp_visualization_utils import generate_plot_with_predictions, get_train_test_series
from matplotlib.figure import Figure
from gpytorch.distributions.multivariate_normal import MultivariateNormal
import umap

from scipy.interpolate import griddata

torch.set_default_dtype(torch.float64)


class Summary:
    """
    Class that prepares experiments' summaries queried by different notebooks.

    :param path_to_exp: the path to the experiment to query
    """
    cfg: DictConfig
    hydra_cfg: DictConfig

    series: torch.Tensor
    train_x: torch.Tensor
    train_x: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    test_steps_gap: int
    train_ratio: float
    window_length: int
    train_overlap: int

    bo_gp_model: BoGPModel

    def __init__(self, path_to_exp: str):
        self.path_to_exp = path_to_exp

        # Load the used experiment configuration
        path_to_cfg_file = os.path.join(self.path_to_exp, '.hydra', 'config.yaml')
        path_to_hydra_cfg = os.path.join(self.path_to_exp, '.hydra', 'hydra.yaml')
        self.cfg = OmegaConf.load(path_to_cfg_file)
        self.hydra_cfg = OmegaConf.load(path_to_hydra_cfg).hydra

        self.series, self.train_x, self.train_y, self.test_x, self.test_y = load_data(self.cfg)

        # get data settings
        self.window_length = self.cfg.bo_optim.train_hparams.window_length
        self.train_overlap = self.cfg.bo_optim.train_hparams.train_overlap
        self.test_steps_gap = self.cfg.data.train_test_specs.test_steps_gap
        self.train_ratio = self.cfg.data.train_test_specs.train_ratio

        # Load the gp_model and likelihood
        gp_model, likelihood = load_gp_likelihood(self.cfg, self.train_x, self.train_y)
        mll = ExactMarginalLogLikelihood(likelihood, gp_model)

        # Get the ax_client path
        ax_client_filename = f"{self.hydra_cfg.job.name}_ax_client.json"
        ax_client_path = os.path.join(self.path_to_exp, ax_client_filename)

        # Initialize the BoGPModel wrapper
        self.bo_gp_model = BoGPModel(gp_model=gp_model, likelihood=likelihood, mll=mll, train_x=self.train_x,
                                     train_y=self.train_y, ax_client_path=ax_client_path, verbose_logging=False)

        # Refit the model to be able to generate plots
        self.bo_gp_model.ax_client.fit_model()

        # Set best hyperparameters
        self.bo_gp_model.set_best_model()

    def generate_predictions_plot(self, figsize: tuple[float, float] = (15, 8)) -> Figure:
        """
        Generates the plot of mean and confidence interval predictions along the original series

        :param figsize: size of the figure to use
        :return:
        """
        observed_pred = self.get_test_predictions()

        post_process = self.cfg.model.standardize  # Post process if the data was standardized
        fig = generate_plot_with_predictions(series=self.series, observed_pred=observed_pred, train_x=self.train_x,
                                             test_x=self.test_x, window_length=self.window_length,
                                             train_overlap=self.train_overlap, test_steps_gap=self.test_steps_gap,
                                             train_ratio=self.train_ratio, figsize=figsize, post_process=post_process)
        return fig

    def get_test_predictions(self) -> MultivariateNormal:
        """
        Generates the posterior test predictions

        :return: observed_pred
        """
        test_predictions_filename = "test_predictions.pickle"
        pred_path = os.path.join(self.path_to_exp, test_predictions_filename)

        # Load predictions from disk if already saved
        if os.path.exists(pred_path):
            observed_pred = load_predictions(pred_path)
        else:
            observed_pred = self.bo_gp_model.predict(test_x=self.test_x, pred_path=pred_path)

        return observed_pred

    def compute_metrics(self) -> DataFrame:
        """
        Generates a dataframe containing computed metrics in columns

        :return: the metrics' data frame
        """
        predictions = self.get_test_predictions()
        mean_predictions = predictions.mean
        computed_metrics = {'model_name': [self.cfg.model.name]}

        # Compute the mean metrics
        for metric in MEAN_METRICS:
            metric_fn = get_metric_fn(metric)
            metric_value = metric_fn(prediction=mean_predictions, target=self.test_y)
            computed_metrics[metric] = [metric_value.item()]

        # Compute the probabilistic metrics
        for metric in PROBABILISTIC_METRICS:
            metric_fn = get_metric_fn(metric)
            metric_value = metric_fn(prediction=predictions, target=self.test_y)
            computed_metrics[metric] = [metric_value.item()]

        return DataFrame(data=computed_metrics)

    def get_train_test_series(self, figsize: tuple[float, float] = (10, 6)) -> Figure:
        return get_train_test_series(self.series, self.train_x, self.test_x, self.window_length,
                                     self.train_overlap, self.train_ratio, figsize)

    def fidelity_state_overlap(self, reference_point: torch.Tensor, sample_points: torch.Tensor) -> torch.Tensor:
        """
        Computes the fidelity state overlap between an array of points and a reference point
        :param reference_point: a reference point, of shape (num_features,) to compare to points from sample_points
        :param sample_points: sample points to compare to the reference_point, of shape (num_points, num_features)
        :return fidelities: tensor of shape (num_points,) that holds the fidelity state overlaps
        """
        kernel = self.bo_gp_model.gp_model.covar_module
        device = kernel.device

        # fidelities of shape (num_points, 1) bellow
        fidelities = kernel(sample_points.to(device), reference_point.unsqueeze(0).to(device))

        return fidelities.squeeze(-1)

    def get_reduced_fidelity_plot_umap(self, reference_point: torch.Tensor, num_sample_points: int,
                                       n_neighbors: int = 15, random_seed: int = None,
                                       figsize: tuple[float, float] = (15, 8), ) -> Tuple[Figure, torch.Tensor,]:
        """
        Generates a summarized fidelity state overlap plot. We reduce the n-dimensional input space into a 2-dimensional
        one using the UMAP dimensionality reduction technique fitted on data of the series in hand. Then we plot the
        fidelity against these sample points.

        :param reference_point: The main reference point to compare each sample point to.
        :param num_sample_points: The number of sample points to work with.
        :param random_seed: The random seed for reproducibility
        :param n_neighbors: The size of local neighborhood (in terms of number of neighboring
                            sample points) used for manifold approximation. Larger values
                            result in more global views of the manifold, while smaller
                            values result in more local data being preserved. In general
                            values should be in the range 2 to 100.
        :param figsize: The size of the figure to use
        :return fig: figure containing the plot.
        """
        # Get the data to fit UMAP on
        if self.cfg.model.standardize:
            standardized_series, _, _ = pre_process_standardize_time_series(self.series)
        else:
            standardized_series = self.series

        num_features = self.train_x.shape[-1]
        # train_data of shape (num_points, num_features)
        train_data = np.array([standardized_series[i:i + num_features]
                               for i in range(len(standardized_series) - num_features)])

        # Fit UMAP on such a data set
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, n_jobs=1,
                                 random_state=random_seed, )
        umap_reducer.fit(train_data)

        # Generate the space of data
        lower_bound = float(standardized_series.min())  # Set the min bound to the min value of the series
        upper_bound = float(standardized_series.max())  # Set the max bound to the max value of the series

        # Generate samples by sampling uniformly across each dimension of num_features
        torch.manual_seed(random_seed)
        sample_points = torch.rand(num_sample_points, num_features) * (upper_bound - lower_bound) + lower_bound

        # Compute the fidelities between sample points and the reference point
        fidelities = self.fidelity_state_overlap(reference_point=reference_point, sample_points=sample_points)
        fidelities = fidelities.detach().cpu().numpy()

        # UMAP transformation
        transformed_sample_points = umap_reducer.transform(sample_points.cpu().numpy())  # Use your UMAP reducer here

        # Grid points for interpolation
        grid_x, grid_y = np.mgrid[min(transformed_sample_points[:, 0]):max(transformed_sample_points[:, 0]):100j,
                         min(transformed_sample_points[:, 1]):max(transformed_sample_points[:, 1]):100j]

        # Interpolating the fidelity values
        grid_z = griddata(transformed_sample_points, fidelities, (grid_x, grid_y), method='cubic')

        # Plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', vmax=1.0)
        fig.colorbar(surf, ax=ax, label='Fidelity')
        ax.set_title('Fidelity Overlap with UMAP Reduced Dimensions')
        ax.set_xlabel('UMAP Dim 1')
        ax.set_ylabel('UMAP Dim 2')
        ax.set_zlabel('Fidelity')

        return fig, fidelities

    def get_fidelity_state_overlap_plot(self, reference_point: torch.Tensor, num_sample_points: int = 20,
                                        lower_bound: int = -1, upper_bound: int = 1,
                                        figsize: tuple[float, float] = (15, 8)) -> Tuple[Figure, torch.Tensor]:
        """
        Generates state fidelity overlap plot, where we fix the last 3 dimensions to math the reference point's values
        and set only the first 2 to vary. The figure is a 3D plot with the z axis is reserved for the fidelity overlap
        and the x and y axes describe the variation of the first 2 dimensions.

        :param reference_point: the main reference point, of shape (num_features,), to compare each sample point to.
        :param num_sample_points: num_sample points to cover per each combination.
        :param lower_bound: lower bound of the range of the variables to cover.
        :param upper_bound: upper bound of the range of the variables to cover.
        :param figsize: size of the figure to use
        :return fig: figure containing the plots
        """
        device = reference_point.device

        num_features = self.train_x.shape[-1]

        x1_range = np.linspace(lower_bound, upper_bound, num_sample_points)
        x2_range = np.linspace(lower_bound, upper_bound, num_sample_points)
        x1, x2 = np.meshgrid(x1_range, x2_range)

        # Creating a meshgrid from the combinations of each pair of elements from x1_range and x2_range
        sample_points_dims = np.stack([x1.ravel(), x2.ravel()], axis=1)

        # Setting up the sample points
        all_dims = list(range(num_features))  # Generating the list of indices [0, 1, ..., n-1]

        dims = [0, 1]
        remaining_dims = [dim for dim in all_dims if dim not in dims]  # Get the remaining dims

        num_points = num_sample_points ** 2
        sample_points = torch.zeros(num_points, num_features).to(device)

        # Set all the other dims to the value of the reference point
        sample_points[:, remaining_dims] = reference_point[remaining_dims].unsqueeze(0)

        # Set the sample points
        sample_points[:, dims] = torch.tensor(sample_points_dims).to(device)

        # Compute the fidelity overlaps. `fidelities` of shape (num_points,)
        fidelities = self.fidelity_state_overlap(reference_point, sample_points).cpu().detach().numpy()

        # Reshape fidelities to match the shape of x1 and x2
        fidelities = fidelities.reshape(x1.shape)

        # Plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x1, x2, fidelities, cmap='viridis')
        fig.colorbar(surf, ax=ax, label='Fidelity')
        ax.set_title('Fidelity Overlap')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Fidelity')

        fidelities = torch.tensor(fidelities).unsqueeze(0)

        return fig, fidelities

    def get_reduced_fidelity_plot_pca(self, reference_point: torch.Tensor, num_sample_points: int,
                                      random_seed: int = None,
                                      figsize: tuple[float, float] = (15, 8), ) -> Tuple[Figure, torch.Tensor,]:
        """
        Generates a summarized fidelity state overlap plot. We reduce the n-dimensional input space into a 2-dimensional
        one using the PCA dimensionality reduction technique fitted on data of the series in hand. Then we plot the
        fidelity against a set of sample points drawn uniformly from the n-dimensional original space.

        :param reference_point: The main reference point to compare each sample point to.
        :param num_sample_points: The number of sample points to work with.
        :param random_seed: The random seed for reproducibility
        :param figsize: The size of the figure to use
        :return fig: figure containing the plot.
        """
        # Get the data to fit PCA on
        if self.cfg.model.standardize:
            standardized_series, _, _ = pre_process_standardize_time_series(self.series)
        else:
            standardized_series = self.series

        num_features = self.train_x.shape[-1]
        # train_data of shape (num_points, num_features)
        train_data = np.array([standardized_series[i:i + num_features]
                               for i in range(len(standardized_series) - num_features)])

        # Fit PCA on such data set
        pca = PCA(n_components=2, random_state=random_seed)
        pca.fit(train_data)

        # Generate the space of data
        lower_bound = float(standardized_series.min())  # Set the min bound to the min value of the series
        upper_bound = float(standardized_series.max())  # Set the max bound to the max value of the series

        # Generate samples by sampling uniformly across each dimension of num_features
        torch.manual_seed(random_seed)
        sample_points = torch.rand(num_sample_points, num_features) * (upper_bound - lower_bound) + lower_bound

        # Compute the fidelities between sample points and the reference point
        fidelities = self.fidelity_state_overlap(reference_point=reference_point, sample_points=sample_points)
        fidelities = fidelities.detach().cpu().numpy()

        # PCA transformation
        transformed_sample_points = pca.transform(sample_points.cpu().numpy())

        # Grid points for interpolation
        grid_x, grid_y = np.mgrid[min(transformed_sample_points[:, 0]):max(transformed_sample_points[:, 0]):100j,
                         min(transformed_sample_points[:, 1]):max(transformed_sample_points[:, 1]):100j]

        # Interpolating the fidelity values
        grid_z = griddata(transformed_sample_points, fidelities, (grid_x, grid_y), method='cubic')

        # Plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', vmax=1.0)
        fig.colorbar(surf, ax=ax, label='Fidelity')
        ax.set_title('Fidelity Overlap with PCA Reduced Dimensions')
        ax.set_xlabel('PCA Dim 1')
        ax.set_ylabel('PCA Dim 2')
        ax.set_zlabel('Fidelity')

        return fig, fidelities

    def get_inv_pca_transformed_fidelity_plot(self, reference_point: torch.Tensor,
                                              num_sample_points: int, random_seed: int = None,
                                              figsize: tuple[float, float] = (15, 8)) -> Tuple[Figure, torch.Tensor]:
        """
       Generates a summarized fidelity state overlap plot. We reduce the n-dimensional input space into a 2-dimensional
       one using the PCA dimensionality reduction technique fitted on data of the series in hand. Then we plot the
       fidelity against a well-structured sample points spanning the hole reduced space.

       :param reference_point: The main reference point to compare each sample point to.
       :param num_sample_points: The number of sample points to work with.
       :param random_seed: The random seed for reproducibility
       :param figsize: The size of the figure to use
       :return fig: figure containing the plot.
       """

        # Get the data to fit PCA on
        if self.cfg.model.standardize:
            standardized_series, _, _ = pre_process_standardize_time_series(self.series)
        else:
            standardized_series = self.series

        num_features = self.train_x.shape[-1]
        # train_data of shape (num_points, num_features)
        train_data = np.array([standardized_series[i:i + num_features]
                               for i in range(len(standardized_series) - num_features)])

        # Fit PCA on such data set
        pca = PCA(n_components=2, random_state=random_seed)
        transformed_train_data = pca.fit_transform(train_data)

        # Bounds based on the transformed training data
        dim1_min, dim1_max = transformed_train_data[:, 0].min(), transformed_train_data[:, 0].max()
        dim2_min, dim2_max = transformed_train_data[:, 1].min(), transformed_train_data[:, 1].max()

        # Create the range for the two principal components
        dim1_range = np.linspace(dim1_min, dim1_max, num_sample_points)
        dim2_range = np.linspace(dim2_min, dim2_max, num_sample_points)

        # Create a meshgrid for the PCA range
        grid_x, grid_y = np.meshgrid(dim1_range, dim2_range)

        # Inverse transform the PCA grid to the original space
        inverse_transformed_points = pca.inverse_transform(np.c_[grid_x.ravel(), grid_y.ravel()])

        # Compute fidelities using the inverse transformed points
        sample_points = torch.tensor(inverse_transformed_points, dtype=torch.float).to(reference_point.device)
        fidelities = self.fidelity_state_overlap(reference_point=reference_point, sample_points=sample_points)
        fidelities = fidelities.cpu().detach().numpy()

        # Reshape fidelities to match the shape of x1 and x2
        fidelities = fidelities.reshape(grid_x.shape)

        # Plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, fidelities, cmap='viridis', edgecolor='none', vmax=1.0)
        fig.colorbar(surf, ax=ax, label='Fidelity')
        ax.set_title('Fidelity Overlap with PCA Reduced Dimensions')
        ax.set_xlabel('PCA Dim 1')
        ax.set_ylabel('PCA Dim 2')
        ax.set_zlabel('Fidelity')

        fidelities = torch.tensor(fidelities).unsqueeze(0)

        return fig, fidelities

    @staticmethod
    def get_results_per_qubits(path_multi_run: str) -> DataFrame:
        """
        Prepares the figure plotting multiple metrics against the varrying of the number of qubits

        :param path_multi_run: path to the multirun experiments
        :return results_df: results_df containing the results.
        """
        results_df = DataFrame()
        runs = [d for d in os.listdir(path_multi_run) if os.path.isdir(os.path.join(path_multi_run, d))]

        for run in runs:
            if 'test_predictions.pickle' not in os.listdir(os.path.join(path_multi_run, runs[0])):
                continue  # Continue if a not fully finished experiment

            summary = Summary(os.path.join(path_multi_run, run))

            # Get the list of qubits in `used_qubits`
            qubits = summary.cfg.bo_optim.train_hparams.window_length

            predictions = summary.get_test_predictions()
            mean_predictions = predictions.mean

            # Compute the list of ll per qubit and store them in `lls`
            ll_fn = get_metric_fn('LogLikelihood')
            ll = ll_fn(prediction=predictions, target=summary.test_y).item()

            # Compute the list of mae per qubit and store them in `maes`
            mae_fn = get_metric_fn('MAE')
            mae = mae_fn(prediction=mean_predictions, target=summary.test_y).item()

            results_df = pd.concat(
                [results_df, DataFrame({'qubits': [qubits], 'LL': ll, 'MAE': mae})],
                ignore_index=True)

        results_df.sort_values(by='qubits', ascending=True, inplace=True, ignore_index=True)

        return results_df

    @staticmethod
    def get_hellinger_distances(summary_list: list, hole_multivariate_distance: bool = True) -> DataFrame:
        """
        Computes the piecewise mean Hellinger distances between each model.

        :param summary_list: list of model summaries.
        :param hole_multivariate_distance: whether to compute the hole Hellinger distance out of the hole posterior
                                           distribution or as a mean of Hellinger distances computed out of univariate
                                           posterior distributions. Default is the hole multivariate distance.
        :return results_df: results_df containing the results.
        """
        assert len(summary_list) > 1

        predictions = [summary.get_test_predictions() for summary in summary_list]
        distances = np.zeros((len(predictions), len(predictions)))
        distance_name = 'HELLINGER' if hole_multivariate_distance else 'MEAN_HELLINGER'
        helligner_distance_fn = get_metric_fn(distance_name)

        # Compute the piece-wise Hellinger distances between the different model predictions
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                distances[i, j] = helligner_distance_fn(prediction_1=predictions[i], prediction_2=predictions[j])
                distances[j, i] = distances[i, j]

        # Prepare them into one DataFrame
        results_df = DataFrame(columns=[summary.cfg.model.name for summary in summary_list],
                               index=[[summary.cfg.model.name for summary in summary_list]])

        for i in range(len(predictions)):
            results_df.loc[summary_list[i].cfg.model.name, summary_list[i].cfg.model.name] = 0
            for j in range(i + 1, len(predictions)):
                results_df.loc[summary_list[i].cfg.model.name, summary_list[j].cfg.model.name] = distances[i, j]
                results_df.loc[summary_list[j].cfg.model.name, summary_list[i].cfg.model.name] = distances[i, j]

        return results_df
