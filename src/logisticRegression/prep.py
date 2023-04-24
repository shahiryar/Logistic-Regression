#TODO: OneHotEncoding given an array (without sklearn)
#TODO: Standardization of data given a numpy array

import csv
import numpy

def load_csv(file_path):
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