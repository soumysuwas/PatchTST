import pandas as pd
import os

def read_and_combine_data(file_path_template, start_year=2014, end_year=2024):
    """
    Reads and combines data from multiple CSV files.

    Args:
        file_path_template (str): A string template for the file paths, 
                                  e.g., "data/{year}ProcessedMeanRainfallGriddedData5000m.csv"
        start_year (int): The starting year.
        end_year (int): The ending year.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data, or None if an error occurs.
    """
    all_data = []
    for year in range(start_year, end_year + 1):
        file_path = file_path_template.format(year=year)
        try:
            df = pd.read_csv(file_path)
            # Keep only relevant columns
            df = df[['system:index', 'Date', 'Rainfall']]
            all_data.append(df)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Please check the file path and try again.")
            return None
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
            return None
    
    if not all_data:
        return None
        
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def parse_system_index(df):
    """
    Parses the 'system:index' column to create 'Grid_ID' and 'Day_of_Year' columns.

    Args:
        df (pandas.DataFrame): The input DataFrame with a 'system:index' column.

    Returns:
        pandas.DataFrame: The DataFrame with added 'Grid_ID' and 'Day_of_Year' columns.
    """
    if 'system:index' not in df.columns:
        print("Error: 'system:index' column not found in DataFrame.")
        return df

    # Split 'system:index' into 'Grid_ID' and 'Day_of_Year'
    # Assumes format "GridID_DayOfYear"
    split_data = df['system:index'].str.split('_', expand=True)
    if split_data.shape[1] == 2:
        df['Grid_ID'] = split_data[0].astype(int)
        df['Day_of_Year'] = split_data[1].astype(int)
    else:
        print("Error: 'system:index' column is not in the expected 'GridID_DayOfYear' format.")
    return df

def convert_date_column(df):
    """
    Converts the 'Date' column to datetime objects.

    Args:
        df (pandas.DataFrame): The input DataFrame with a 'Date' column.

    Returns:
        pandas.DataFrame: The DataFrame with the 'Date' column converted to datetime objects.
    """
    if 'Date' not in df.columns:
        print("Error: 'Date' column not found in DataFrame.")
        return df
        
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    except Exception as e:
        print(f"Error converting 'Date' column to datetime: {e}")
    return df

def pivot_data(df):
    """
    Pivots the data to have 'Date' as index, 'Grid_ID' as columns, and 'Rainfall' as values.

    Args:
        df (pandas.DataFrame): The input DataFrame with 'Date', 'Grid_ID', and 'Rainfall' columns.

    Returns:
        pandas.DataFrame: The pivoted DataFrame, or None if an error occurs.
    """
    required_columns = ['Date', 'Grid_ID', 'Rainfall']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: DataFrame must contain the following columns for pivoting: {', '.join(required_columns)}")
        return None

    try:
        pivot_df = df.pivot(index='Date', columns='Grid_ID', values='Rainfall')
    except Exception as e:
        print(f"Error pivoting data: {e}")
        return None
    return pivot_df

def split_data(df):
    """
    Splits the data into training (2014-2023) and testing (2024) sets.

    Args:
        df (pandas.DataFrame): The pivoted DataFrame with a datetime index.

    Returns:
        tuple: A tuple containing the training DataFrame and the testing DataFrame.
               Returns (None, None) if an error occurs or if the DataFrame is empty.
    """
    if df is None or df.empty:
        print("Error: Input DataFrame is empty or None. Cannot split data.")
        return None, None

    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index must be a DatetimeIndex for splitting.")
        return None, None

    try:
        # Define the date ranges for training and testing sets
        train_start_date = '2014-01-01'
        train_end_date = '2023-12-31'
        test_start_date = '2024-01-01'
        test_end_date = '2024-12-31'

        # Create boolean masks for selecting data within the date ranges
        train_mask = (df.index >= train_start_date) & (df.index <= train_end_date)
        test_mask = (df.index >= test_start_date) & (df.index <= test_end_date)
        
        train_df = df[train_mask]
        test_df = df[test_mask]

        if train_df.empty:
            print("Warning: Training data is empty after splitting. Check date ranges and input data.")
        if test_df.empty:
            print("Warning: Testing data is empty after splitting. Check date ranges and input data.")
            
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None
        
    return train_df, test_df

if __name__ == '__main__':
    # This is an example of how to use the functions.
    # The user should replace 'path/to/your/data/{year}ProcessedMeanRainfallGriddedData5000m.csv'
    # with the actual path to their data.
    
    # --- Configuration ---
    # TODO: **User Action Required** 
    # Replace this with the actual path template to your CSV files.
    # For example: "data/rainfall_data/{year}ProcessedMeanRainfallGriddedData5000m.csv"
    # or if the files are in the same directory as the script: 
    # "{year}ProcessedMeanRainfallGriddedData5000m.csv"
    file_path_template = "path/to/your/data/{year}ProcessedMeanRainfallGriddedData5000m.csv" 

    # --- Data Loading and Preprocessing ---
    print("Starting data loading and preprocessing...")
    combined_data = read_and_combine_data(file_path_template)

    if combined_data is not None:
        print(f"Successfully combined data. Shape: {combined_data.shape}")
        
        # Parse 'system:index'
        combined_data = parse_system_index(combined_data)
        if 'Grid_ID' in combined_data.columns and 'Day_of_Year' in combined_data.columns:
            print("Successfully parsed 'system:index'.")
        else:
            print("Skipping further processing due to error in parsing 'system:index'.")
            exit() # Exit if parsing failed as other steps depend on it.

        # Convert 'Date' column
        combined_data = convert_date_column(combined_data)
        if combined_data['Date'].dtype == 'datetime64[ns]':
             print("Successfully converted 'Date' column to datetime.")
        else:
            print("Skipping further processing due to error in converting 'Date' column.")
            exit() # Exit if date conversion failed

        # Pivot data
        pivoted_df = pivot_data(combined_data)
        if pivoted_df is not None:
            print(f"Successfully pivoted data. Shape: {pivoted_df.shape}")

            # Split data
            train_df, test_df = split_data(pivoted_df)
            if train_df is not None and test_df is not None:
                print(f"Successfully split data. Training shape: {train_df.shape}, Testing shape: {test_df.shape}")
                
                # --- Further Analysis (Placeholder) ---
                # At this point, train_df and test_df are ready for use in a time series model.
                # For example, you could print some basic info:
                print("\n--- Sample of Training Data ---")
                print(train_df.head())
                
                print("\n--- Sample of Testing Data ---")
                print(test_df.head())

                print("\n--- Training Data Info ---")
                train_df.info()

                print("\n--- Testing Data Info ---")
                test_df.info()

            else:
                print("Data splitting failed.")
        else:
            print("Data pivoting failed.")
    else:
        print("Data loading and combination failed. Please check file paths and file contents.")

    print("\nPreprocessing script finished.")
