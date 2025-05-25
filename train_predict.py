import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

# Attempt to import from local modules
try:
    from preprocessing import read_and_combine_data, parse_system_index, convert_date_column, pivot_data, split_data
    from patchtst_model import PatchTST
except ImportError:
    print("Make sure preprocessing.py and patchtst_model.py are in the same directory.")
    # You might want to exit or raise an error if these are critical
    # For now, we'll define dummy functions if they are not found, so the script can be partially runnable
    def read_and_combine_data(*args, **kwargs): print("Dummy read_and_combine_data called"); return None
    def parse_system_index(df, *args, **kwargs): print("Dummy parse_system_index called"); return df
    def convert_date_column(df, *args, **kwargs): print("Dummy convert_date_column called"); return df
    def pivot_data(df, *args, **kwargs): print("Dummy pivot_data called"); return None
    def split_data(df, *args, **kwargs): print("Dummy split_data called"); return None, None
    class PatchTST(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Linear(1,1); print("Dummy PatchTST initialized")
        def forward(self, x): print(f"Dummy PatchTST forward called with x shape: {x.shape if isinstance(x, torch.Tensor) else 'Non-tensor input'}"); return torch.randn_like(x) if isinstance(x, torch.Tensor) else torch.randn(1,1,1)


# 2. Define training parameters
HISTORY_LEN = 104 # Length of the input sequence (lookback window)
FORECAST_HORIZON = 365 # Number of steps to forecast (e.g., 365 for one year) - For PatchTST, this is also pred_len
PATCH_LEN = 16
STRIDE = 8
D_MODEL = 128
N_HEADS = 8
N_ENCODER_LAYERS = 3
D_FF = 256
DROPOUT = 0.1
ATTN_DROPOUT = 0.0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10 # Keep low for initial testing
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File path for data - adjust as needed
FILE_PATH_TEMPLATE = "data/{year}ProcessedMeanRainfallGriddedData5000m.csv" 
# Example: "data/rainfall_data/{year}ProcessedMeanRainfallGriddedData5000m.csv"
# or if the files are in the same directory as the script: 
# "{year}ProcessedMeanRainfallGriddedData5000m.csv"

def create_sequences(data, history_len, forecast_horizon):
    """
    Creates sequences from time series data.
    Args:
        data (np.array): Time series data (single feature).
        history_len (int): Length of the input sequence.
        forecast_horizon (int): Length of the output sequence.
    Returns:
        np.array: Input sequences (X).
        np.array: Target sequences (y).
    """
    X, y = [], []
    for i in range(len(data) - history_len - forecast_horizon + 1):
        X.append(data[i:(i + history_len)])
        y.append(data[(i + history_len):(i + history_len + forecast_horizon)])
    return np.array(X), np.array(y)

def main():
    print(f"Using device: {DEVICE}")

    # 3. Load and Preprocess Data
    print("Loading and preprocessing data...")
    try:
        # Check if data directory exists
        if not os.path.exists("data"):
            print("Data directory 'data/' not found. Please create it and place your CSV files there.")
            print("Continuing with dummy data if modules were not imported properly.")
            # If we are using dummy functions because real ones failed to import,
            # we need to ensure pivot_df is not None for the script to proceed somewhat.
            if 'pivot_data' in globals() and globals()['pivot_data'].__doc__ is None : # Check if it's a dummy
                 # Create a dummy pivot_df for single grid processing if real data loading fails
                dates_train = pd.date_range(start='2014-01-01', end='2023-12-31', freq='D')
                data_train = np.random.rand(len(dates_train))
                train_df_dummy = pd.DataFrame(data={'dummy_grid': data_train}, index=dates_train)

                dates_test = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
                data_test = np.random.rand(len(dates_test))
                test_df_dummy = pd.DataFrame(data={'dummy_grid': data_test}, index=dates_test)
                
                # Make sure the dummy PatchTST can handle the forecast horizon
                global FORECAST_HORIZON 
                if isinstance(PatchTST(1,1,1,1,1,1,1,1,1,1), nn.Module) and PatchTST(1,1,1,1,1,1,1,1,1,1).fc.out_features == 1 : # if it is a dummy PatchTST
                    print(f"Adjusting FORECAST_HORIZON for dummy PatchTST from {FORECAST_HORIZON} to 1.")
                    FORECAST_HORIZON = 1 # Dummy PatchTST might not handle arbitrary forecast_horizon

            else: # Real functions are available, but data dir is missing
                return # Stop execution

        combined_data = read_and_combine_data(FILE_PATH_TEMPLATE, start_year=2014, end_year=2024)
        if combined_data is None:
            print("Failed to load data. Exiting.")
            if 'pivot_data' not in globals() or globals()['pivot_data'].__doc__ is not None: # Not using dummy
                return
            # else: proceed with dummy data structure created above if modules are dummy

        if combined_data is not None: # if real data was loaded
            combined_data = parse_system_index(combined_data)
            combined_data = convert_date_column(combined_data)
            pivot_df = pivot_data(combined_data)
            if pivot_df is None:
                print("Pivoting failed. Exiting.")
                return
            train_df_all_grids, test_df_all_grids = split_data(pivot_df)
            if train_df_all_grids is None or test_df_all_grids is None:
                print("Data splitting failed. Exiting.")
                return
        else: # if combined_data is None and we are using dummy modules
            train_df_all_grids, test_df_all_grids = train_df_dummy, test_df_dummy


    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data files are in the 'data/' subdirectory.")
        return
    except Exception as e:
        print(f"An error occurred during data loading/preprocessing: {e}")
        return

    # Focus on a single grid for initial development
    # TODO: Loop through all grids for full grid-by-grid prediction
    if not train_df_all_grids.empty:
        grid_id = train_df_all_grids.columns[0] 
        print(f"Focusing on Grid_ID: {grid_id}")
        train_data_single_grid = train_df_all_grids[grid_id].values
        test_data_single_grid = test_df_all_grids[grid_id].values
    else:
        print("No data available to select a grid. Exiting.")
        return

    # Handle potential NaN values from real data (simple imputation for now)
    train_data_single_grid = np.nan_to_num(train_data_single_grid, nan=0.0)
    test_data_single_grid = np.nan_to_num(test_data_single_grid, nan=0.0)


    # 4. Prepare Data for Model
    print("Preparing data for the model...")
    X_train, y_train = create_sequences(train_data_single_grid, HISTORY_LEN, FORECAST_HORIZON)
    X_test, y_test = create_sequences(test_data_single_grid, HISTORY_LEN, FORECAST_HORIZON)

    # Reshape for PatchTST: [batch_size, n_vars, history_len]
    # n_vars = 1 for single grid
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    if X_train_tensor.nelement() == 0 or X_test_tensor.nelement() == 0 :
        print(f"Not enough data for Grid_ID {grid_id} to create sequences with history_len={HISTORY_LEN} and forecast_horizon={FORECAST_HORIZON}.")
        print(f"Train data length: {len(train_data_single_grid)}, Test data length: {len(test_data_single_grid)}")
        # TODO: Add logic here to iterate to the next grid or handle this scenario appropriately
        return # For now, we just exit if a single grid doesn't have enough data.

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Instantiate Model, Loss, and Optimizer
    print("Instantiating model, loss, and optimizer...")
    # For single grid, c_in (n_vars) = 1. seq_len is HISTORY_LEN.
    model = PatchTST(
        c_in=1,  # Number of input variables (features)
        seq_len=HISTORY_LEN, 
        forecast_horizon=FORECAST_HORIZON,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_epoch_loss:.4f}")

    # 7. Prediction
    print("Generating predictions...")
    model.eval()
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(batch_y.cpu().numpy())

    # Concatenate predictions and actuals from all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    print(f"Predictions shape: {all_predictions.shape}") # Should be [num_test_samples, 1, forecast_horizon]
    print(f"Actuals shape: {all_actuals.shape}")   # Should be [num_test_samples, 1, forecast_horizon]

    # TODO: Implement evaluation metrics (MAE, RMSE) for this grid
    # Example:
    # mse = np.mean((all_predictions - all_actuals)**2)
    # mae = np.mean(np.abs(all_predictions - all_actuals))
    # print(f"Grid {grid_id} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

    # TODO: Implement plotting for this grid (actual vs. predicted)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(all_actuals[0, 0, :], label='Actual') # Plot first sample, first variable
    # plt.plot(all_predictions[0, 0, :], label='Predicted')
    # plt.title(f"Forecast for Grid {grid_id} - First Test Sample")
    # plt.xlabel("Time Step into Horizon")
    # plt.ylabel("Rainfall")
    # plt.legend()
    # plt.show()

    print("\nTraining and prediction script finished for single grid.")

if __name__ == "__main__":
    main()
