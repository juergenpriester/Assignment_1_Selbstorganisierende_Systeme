"""
Data loader module for optimization algorithms.

This module handles loading and preprocessing of data for the 
Travelling Salesman Problem and Rastrigin Problem.
"""

import os
import pandas as pd
import kagglehub
import glob

def load_data():
    """
    Load all CSV datasets from the Kaggle dataset cache.
    If data is not present, it will be downloaded.
    
    Returns:
        dataframes: A dictionary of pandas DataFrames, or an empty dictionary if load is False.
    """

    print("Downloading dataset (if not already cached)...")
    # Download dataset to a local cache and return the path
    dataset_path = kagglehub.dataset_download("mexwell/traveling-salesman-problem")
    print(f"Dataset is available at: {dataset_path}")

    print("Loading datasets from the dataset directory...") 
    # Find all CSV files in the dataset directory
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))

    if not csv_files:
        print("No CSV files found in the dataset directory.")
        return {}

    dataframes = {}
    print("Loading all TSP datasets...")
    for file_path in csv_files:
        # Use the filename without extension as the key
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f" - Loading {dataset_name}...")
        dataframes[dataset_name] = pd.read_csv(file_path)
    
    return dataframes


def preprocess_data(data):
    """
    Preprocess the loaded data.
    
    Args:
        data (dict): Raw data dictionary
        
    Returns:
        dict: Preprocessed data dictionary
    """
    # TODO: Implement preprocessing logic
    return data
