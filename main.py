"""
Title: main.py
Author: Damian Grunert
Date: 17.07.2024
License: MIT License
Contact: damian.grunert@gmx.de

Description:
This Python script implements a handy way to apply the optimal model obtained in the paper "A Machine Learning Based Algorithm for the Prediction of Eigenfrequencies of Railway Bridges", International Journal of Structural Stability and Dynamics (IJSSD) to a given dataset.
As an input, it takes a csv file of feature values for a number of datapoints with a given ID and then returns a csv file with the corresponding predictions.
[TODO]: Preprocesses the data as to remove outliers unsuitable for model application.  

Notes:
- To run this script, the python programming language as well as certain libraries have to be installed according to the requirements.txt - file
    - To install these automatically, please run "pip install -r requirements.txt" in this directory using the command line  
- As stated in the readme, the csv files must have the following entries for each row, in THIS particular order:
     BRIDGE_ID,eigenfrequency_n03_Hz,span_L_m,stiffness_EI_MNm2,mass_m_tpm,yoc_yyyy,maxlocalspeed_v_kmph,bearingcapacity_beta_,heigtogconstruction_hk_m,numberofcaps_,skewness_degree,bearing_

Usage:
Use the data_path argument to specify the csv file predictions should be applied for.

E.g.:
python main.py 'SampleData.csv'

"""

import XGBoostPrediction
import argparse 
import pandas as pd
import numpy as np

def main(data_path):
    print(f'Loading data...')
    # Load the dataset from the provided file path
    data_unknown = pd.read_csv(data_path)
    data_unknown = data_unknown.to_numpy()

    # Extract relevant columns
    data_id = data_unknown[:, 0]
    data_unknown = np.delete(data_unknown, 0, 1)

    print(f'Obtaining prediction...')
    predictions = XGBoostPrediction.get_prediction(data_unknown)

    results = pd.DataFrame({'data_id': data_id.reshape(-1), 'prediction': predictions.reshape(-1)})
    results.to_csv('predictions.csv', index=False)

    print(f'Predictions saved under predictions.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost model training and saving script")
    parser.add_argument('data_path', type=str, help='Path to the CSV data file')
    args = parser.parse_args()
    
    main(args.data_path)
