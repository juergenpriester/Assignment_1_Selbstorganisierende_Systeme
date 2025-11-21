"""
Data loader module for optimization algorithms.

This module handles loading of data for the 
Travelling Salesman Problem and Rastrigin Problem.
"""

import os
import pandas as pd
import kagglehub
import glob

def load_data():
    """
    Load all CSV datasets from the Kaggle dataset cache.
    If data is not present, it will be downloaded automatically.
    
    Returns:
        dict: A dictionary of pandas DataFrames, with dataset names as keys.
    """
    print("Locating dataset... (will download if not cached)")
    # This will download the dataset to a local cache if not present,
    # and return the path to the dataset directory.
    dataset_path = kagglehub.dataset_download("mexwell/traveling-salesman-problem")
    print(f"Dataset path: {dataset_path}")

    # Find all CSV files in the dataset directory
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))

    if not csv_files:
        print("Error: No CSV files found in the dataset directory.")
        return {}

    dataframes = {}
    print("Loading all TSP datasets...")
    for file_path in csv_files:
        # Use the filename without extension as the key
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f" - Loading '{dataset_name}'...")
        dataframes[dataset_name] = pd.read_csv(file_path)
    
    return dataframes
