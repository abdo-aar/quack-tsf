import torch


def financial_like_series(num_time_steps: int, noise_level: int = 1.5, volatility_scale: int = 2, base_scale: int = 1.5,
                          seed: int = 1997):
    """
    Generates a synthetic time series with complex patterns resembling financial data,
    including smooth zigzags and time-dependent volatility.

    Parameters:
    - num_time_steps: Number of time steps the time series should have.
    - noise_level: The level of noise to add for zigzags. Default is 1.5.
    - volatility_scale: Scale factor for the volatility modulator. Default is 2.
    - base_scale: Scale factor for the base sine waves. Default is 1.5.
    - seed: Manual seed for reproducibility

    Returns:
    - s_financial_like_adjusted: The generated synthetic time series.
    """

    torch.manual_seed(seed)

    T = num_time_steps + 1  # Adjust total number of time steps accordingly
    time_steps = torch.arange(0, T)

    # Recreate complex patterns with adjusted noise and volatility
    sine_wave1 = torch.sin(time_steps / 2) * base_scale
    sine_wave2 = torch.sin(time_steps / 3) * base_scale
    sine_wave3 = torch.sin(time_steps / 5) * base_scale

    # Time-dependent volatility: use a sine wave to modulate the volatility over time
    volatility_modulator = torch.sin(time_steps / 50) * volatility_scale + 3
    random_walk_volatility = torch.cumsum(torch.randn(T) * volatility_modulator, dim=0)

    # Combine components with moderated noise for less pronounced zigzags
    combined_financial_series_adjusted = sine_wave1 + sine_wave2 + sine_wave3 + random_walk_volatility
    noise_adjusted = torch.randn(T) * noise_level
    s_financial_like_adjusted = combined_financial_series_adjusted + noise_adjusted

    return s_financial_like_adjusted


def trend_periodic_time_series(num_time_steps: int, noise_level: float = 0.5, trend_changes: int = 4, seed: int = 1997,
                               base_scale1=1, base_scale2=0.5, period1=10):
    """
    Generates a synthetic time series with multiple up and downtrends and non-linear dynamics.

    Parameters:
    - num_time_steps: Number of time steps the time series should have.
    - noise_level: The level of Gaussian noise to add. Default is 0.5.
    - trend_changes: Number of points where the trend direction changes. Default is 4.
    - seed: Manual seed for reproducibility.

    Returns:
    - time_series: The generated synthetic time series.
    """

    T = num_time_steps + 1  # Adjust total number of time steps accordingly

    torch.manual_seed(seed)

    # if not trend_changes:
    #     trend_changes = torch.randint(low=1, high=10, size=(1,))

    time_steps = torch.arange(0, T).float()
    period2 = T / (trend_changes * 2)  # Period for sine waves to align with trend changes

    # Non-linear dynamics: a combination of sine waves
    sine_wave1 = torch.sin(time_steps / period1) * base_scale1  # Longer period, larger amplitude
    sine_wave2 = torch.sin(time_steps / period2) * base_scale2  # Synced with trend changes

    # Piecewise linear trends
    trend = torch.zeros(T)
    for i in range(trend_changes):
        start_idx = int(i * T / trend_changes)
        end_idx = int((i + 1) * T / trend_changes)
        if i % 2 == 0:  # Up trend
            trend[start_idx:end_idx] = torch.linspace(start_idx, end_idx, end_idx - start_idx) * 0.002
        else:  # Down trend
            trend[start_idx:end_idx] = torch.linspace(end_idx, start_idx, end_idx - start_idx) * 0.002

    # Adjusting the trend to have zero mean to overlay with sine components seamlessly
    trend -= trend.mean()

    # Combine components and add noise
    combined_series = sine_wave1 + sine_wave2 + trend
    noise = torch.randn(T) * noise_level
    series = combined_series + noise

    return series
