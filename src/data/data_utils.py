import torch
from torch import Tensor


def pre_process_standardize_time_series(series: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Standardize an input series: set its values to have a mean of 0 and a std of 1

    :param series: Tensor object of shape (T,)
    :return: standardized_time_series, org_mean, org_std: standardized time series + mean & std
    """

    org_mean = series.mean()
    org_std = series.std()

    # Standardize the time series
    standardized_time_series = (series - org_mean) / org_std

    return standardized_time_series, org_mean, org_std


def post_process_time_series(series: Tensor, org_mean: Tensor, org_std: Tensor) -> Tensor:
    # Post-process the time series
    post_processed_series = series * org_std + org_mean

    return post_processed_series


# Define function to subdivide the series into training and testing sets with specified overlap and no overlap
# between sets
def create_train_test_series(series: Tensor, window_length: int, train_overlap: int, test_steps_gap: int = 1,
                             train_ratio: float = 0.8, torch_device='cpu') -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Subdivides a time series into training and testing datasets based on the specified parameters.

    :param series: The complete time series data of shape (T,).
    :param window_length: The length of each window (w).
    :param train_overlap: The overlap between consecutive train windows.
    :param test_steps_gap: The gap between two consecutive test label points.
    :param train_ratio: The ratio of the dataset to be used for training (default is 0.8).
    :param torch_device: which device to store the tensors at:'cpu', or a 'cuda:index' device

    :return: Four tensors - train_x, train_y, test_x, test_y.
    """

    assert train_overlap < window_length, "Parameter `train_overlap` should be less than `window_length` !"
    assert train_overlap >= 0, "Parameter `train_overlap` should be positive !"

    torch_device = torch.device(torch_device)
    total_length = series.shape[0]
    train_test_div_index = int(total_length * train_ratio)

    # Produce train time series
    train_length = train_test_div_index - 1  # To avoid future leakage problem
    total_train_windows = (train_length - window_length) // (window_length - train_overlap)
    train_end_index = total_train_windows * (window_length - train_overlap) + window_length  # This <= train_length

    train_x, train_y = [], []
    for i in range(0, train_end_index - window_length, window_length - train_overlap):
        train_x.append(series[i:i + window_length])
        train_y.append(series[i + window_length])

    # Produce test time series
    test_x, test_y = [], []

    # test_start_index > train_end_index => no future leakage problem
    test_start_index, test_length = train_test_div_index, (total_length - train_test_div_index)
    total_test_windows = test_length // test_steps_gap
    test_end_index = test_start_index + total_test_windows * test_steps_gap

    for i in range(test_start_index, test_end_index, test_steps_gap):
        test_y.append(series[i])
        test_x.append(series[i - window_length:i])

    return (torch.stack(train_x).to(torch_device), torch.tensor(train_y).to(torch_device),
            torch.stack(test_x).to(torch_device), torch.tensor(test_y).to(torch_device))
