#TODO: Standardization of data given a numpy array


import csv
import numpy as np

def load_csv(file_path):
    """
    Load a CSV file located at the specified file path and return a dictionary containing the column names
    and values in the file.

    Args:
        file_path (str): The path to the CSV file to load.

    Returns:
        dict: A dictionary with the following keys:
            - "columns": a list of strings representing the column names in the CSV file.
            - "values": a numpy array representing the values in the CSV file.
    
    Raises:
        Exception: If the number of values in a row does not match the number of columns in the CSV file.
    """
    vals = []
    cols = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        cols = list(reader.fieldnames)
        n_features = len(cols)
        
        for row in reader:
            observation = list(row.values())
            if len(observation) != n_features: 
                raise Exception("Number of values do not match the number of columns")
            vals.append(list(row.values()))
    vals = np.array(vals, dtype=float)
    return {"columns": cols, "values":vals}

def one_hot_encode(arr, prefix):
    """
    Encode an array of strings using one-hot encoding and return a dictionary
    containing the encoded data with column names as keys.

    Args:
    arr (np.ndarray): Array of strings to encode
    prefix (str): Prefix to use for column names

    Returns:
    dict: A dictionary containing the encoded data with column names as keys
    a small change
    """
    unique_vals = np.unique(arr)
    encoded = np.zeros((len(arr), len(unique_vals)))
    
    for i, val in enumerate(unique_vals):
        encoded[:, i] = (arr == val).astype(int)
    
    columns = [f"{prefix}_{val}" for val in unique_vals]
    result_dict = dict(zip(columns, encoded.T))
    
    return result_dict