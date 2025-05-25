import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """
    Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    Args:
        y_true (numpy.ndarray or torch.Tensor): True values.
        y_pred (numpy.ndarray or torch.Tensor): Predicted values.

    Returns:
        dict: A dictionary containing MAE and RMSE.
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Flatten arrays to ensure they are 1D for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Handle potential NaNs by removing them if present (or raising an error)
    # For this example, we'll remove corresponding pairs if NaNs are in either.
    valid_indices = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_clean = y_true_flat[valid_indices]
    y_pred_clean = y_pred_flat[valid_indices]

    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        print("Warning: No valid (non-NaN) data points found for metric calculation.")
        return {'mae': np.nan, 'rmse': np.nan}
        
    if y_true_clean.shape != y_pred_clean.shape:
        # This case should ideally be handled before calling this function,
        # for example, by ensuring predictions match the horizon.
        # However, if there's a mismatch not caught earlier:
        min_len = min(len(y_true_clean), len(y_pred_clean))
        y_true_clean = y_true_clean[:min_len]
        y_pred_clean = y_pred_clean[:min_len]
        print(f"Warning: y_true and y_pred have different shapes after NaN removal/flattening. Metrics calculated on common length {min_len}.")


    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    return {'mae': mae, 'rmse': rmse}

def plot_forecast(y_true, y_pred, grid_id, forecast_horizon, save_dir="plots"):
    """
    Generates and saves a line plot comparing actual vs. predicted rainfall.

    Args:
        y_true (numpy.ndarray or torch.Tensor): True values (expected 1D or squeezable to 1D).
        y_pred (numpy.ndarray or torch.Tensor): Predicted values (expected 1D or squeezable to 1D).
        grid_id (str): The ID of the grid for labeling the plot.
        forecast_horizon (int): The number of time steps in the forecast.
        save_dir (str): Directory to save the plot.
    """
    # Convert to numpy arrays if they are torch tensors and ensure they are 1D
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.squeeze().flatten()
    y_pred = y_pred.squeeze().flatten()

    # Ensure lengths match the forecast horizon for plotting
    # This might truncate or error if lengths are inconsistent
    if len(y_true) != forecast_horizon or len(y_pred) != forecast_horizon:
        print(f"Warning: Length of y_true ({len(y_true)}) or y_pred ({len(y_pred)}) "
              f"does not match forecast_horizon ({forecast_horizon}). Plot might be misleading.")
        # Adjust to the minimum of the lengths or horizon for plotting
        min_len = min(len(y_true), len(y_pred), forecast_horizon)
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        time_steps = np.arange(1, min_len + 1)
    else:
        time_steps = np.arange(1, forecast_horizon + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_true, label='Actual Rainfall', marker='.', linestyle='-')
    plt.plot(time_steps, y_pred, label='Predicted Rainfall', marker='x', linestyle='--')
    
    plt.title(f"Rainfall Forecast vs Actual for Grid {grid_id}")
    plt.xlabel(f"Time Step into Horizon (Days 1-{len(time_steps)})")
    plt.ylabel("Rainfall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    plot_filename = os.path.join(save_dir, f"forecast_plot_grid_{grid_id}.png")
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close() # Close the figure to free memory

if __name__ == "__main__":
    # Dummy data for demonstration
    forecast_horizon_sample = 365
    y_true_sample = np.random.rand(forecast_horizon_sample) * 10 # Simulate rainfall values
    y_pred_sample = y_true_sample + (np.random.rand(forecast_horizon_sample) * 2 - 1) # Simulate predictions with some noise
    grid_id_sample = "dummy_grid_01"

    # Demonstrate calculate_metrics
    print("--- Calculating Metrics ---")
    metrics = calculate_metrics(y_true_sample, y_pred_sample)
    print(f"Metrics for {grid_id_sample}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    # Demonstrate calculate_metrics with NaNs
    y_true_with_nan = y_true_sample.copy()
    y_true_with_nan[10:20] = np.nan # Introduce some NaNs
    y_pred_with_nan = y_pred_sample.copy()
    y_pred_with_nan[15:25] = np.nan
    print("\n--- Calculating Metrics with NaNs ---")
    metrics_nan = calculate_metrics(y_true_with_nan, y_pred_with_nan)
    print(f"Metrics for {grid_id_sample} (with NaNs): MAE={metrics_nan['mae']:.4f}, RMSE={metrics_nan['rmse']:.4f}")
    
    # Demonstrate plot_forecast
    print("\n--- Plotting Forecast ---")
    plot_forecast(y_true_sample, y_pred_sample, grid_id_sample, forecast_horizon_sample)
    
    # Demonstrate with slightly mismatched lengths for plotting (should warn and adjust)
    print("\n--- Plotting Forecast with Mismatched Length (for warning demo) ---")
    plot_forecast(y_true_sample[:300], y_pred_sample, "mismatch_grid", forecast_horizon_sample)

    print("\nEvaluation script demonstration finished.")
