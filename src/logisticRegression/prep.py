#TODO: OneHotEncoding given an array (without sklearn)
#TODO: Standardization of data given a numpy array

import csv
import numpy

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
    vals = numpy.array(vals, dtype=float)
    return {"columns": cols, "values":vals}