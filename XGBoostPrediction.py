"""
Title: XGBoostPrediction.py
Author: Damian Grunert
Date: 17.07.2024
License: MIT License
Contact: damian.grunert@gmx.de

Description:
This Python script implements the eigenfrequency prediction for railway bridges as well as evaluation methods using the final XGBoost model 
obtained in the paper: "A Machine Learning Based Algorithm for the Prediction of Eigenfrequencies of Railway Bridges", International Journal of Structural Stability and Dynamics (IJSSD). 

Notes:
- To run this script, the python programming language as well as certain libraries have to be installed according to the requirements.txt - file
    - To install these automatically, please run "pip install -r requirements.txt" in this directory using the command line  
- Data is expected in the following formats:
    - Features: 2D numpy array of datapoints in first dimension and second dimensions representing, in order, the following metrics:
    
    n0_3,STW_NUM,BIEGESTEIFIGKEIT,MASSE,BAUJAHR,V_SOLL_IST,BETA_HT_M_DEB,HOEHE_KONST_UEB_MITTE_DEB,Ssp_Randkonstruktion,WANDSCHIEFE_RAHMEN_P1_NUM2,DEB_I_LAGERUNG_UEBERBAU2

    - Labels: 2D numpy array of the measured eigenfrequency in the second dimension for each datapoint represented in the first dimension
    - Both must, row-wise, correspond to the same datapoint, and must thus in particular match in the first dimension
    
Usage:
In order to use this file, one should import it using 
    
    import XGBoostPrediction.py 

at the beginning of another python script. 

One can call the corresponding functions, keeping in mind the requirements posed above  
This includes:
- get_k_fold_error(feaures, labels)
    - Returns the k-fold-error for the given dataset with a supplied feature- and label matrix, see Notes

- get_test_error(features, labels)
    - Returns the testing error for a supplied testing set with a given feature- and label matrix, see Notes

- get_prediction(features)
    - Returns the eigenfrequency predictions for a dataset with supplied features, see Notes

Example usage in a script:

import XGBoostPrediction.py
features = pd.read_csv('SampleData.csv')
data_id = features[:, 0]
features = np.delete(features, 0, 1)
labels = np.array([[13.7]])
print(XGBoostPrediction.get_test_error(features, labels))
print(XGBoostPrediction.get_k_fold_error(features, labels))
print(XGBoostPrediction.get_prediction(features))
"""

import xgboost
import numpy as np

# Optimal hyperparameters identified:
xgboost_opt = [0.0, 0.07896507820168863, 6, 969, 0.7184912525930043, 2, 0.0, 3.211285854810482]

# Initializing and loading the resulting model
xgboost_reg = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
xgboost_reg.load_model('final_xgboost_model.json')

# Accessory methods used in the following
def mape(y_hat, y):
        return np.sum( np.abs((np.array(y_hat) - np.array(y).reshape(np.array(y_hat).shape)) / np.array(y_hat)) ) / (np.array(y_hat).size)

def train(features, labels):
    xgboost_reg.fit(features, labels)

def cost(features, labels):
    pred = xgboost_reg.predict(features).reshape(labels.shape[0], 1)
    return mape(labels, pred)

# Method for obtaining the k-fold-crossvalidation loss on a given training set, e.g. for own optimization experiments
def get_k_fold_error(features, labels):
    def k_fold_cross_val(k, features, labels, train_func, cost_func, seed):
        #Shuffling
        np.random.seed(seed)
        p = np.random.permutation(features.shape[0])
        shuffled_features = features.copy()[p]
        shuffled_labels = labels.copy()[p]
        error = 0
        for l in range(k - 1):
            #The test data of the current fold
            test_features = shuffled_features[features.shape[0] // k * l:features.shape[0] // k * (l+1), :]
            test_labels = shuffled_labels[features.shape[0] // k * l:features.shape[0] // k * (l+1), :]
            #The remaining training data of the current fold
            train_features = np.vstack((shuffled_features[:features.shape[0] // k * l, :], shuffled_features[features.shape[0] // k * (l+1):, :]))
            train_labels = np.vstack((shuffled_labels[:features.shape[0] // k * l, :], shuffled_labels[features.shape[0] // k * (l+1):, :])) 
            #Now, train the model on the current fold
            train_func(train_features, train_labels)
            error += cost_func(test_features, test_labels) / k

        #For the last fold, we dont really know the size of the holdout-set (we dont know about the divisibility of the amount of datapoints by k) so we do this seperately
        #The test data of the last fold 
        test_features = shuffled_features[features.shape[0] // k * (l + 1):, :]
        test_labels = shuffled_labels[features.shape[0] // k * (l + 1):, :]
        #The remaining training data of the last fold
        train_features = shuffled_features[:features.shape[0] // k * (l + 1), :]
        train_labels = shuffled_labels[:features.shape[0] // k * (l + 1), :]
        #Now, train the model on the current fold
        train_func(train_features, train_labels)
        error += cost_func(test_features, test_labels) / k
        return error
    
    avg = 0
    for i in range(3):
        val = k_fold_cross_val(10, features, labels, train, cost, seed = i)
        print(val)
        avg += val / 3

    xgboost_reg.load_model('final_xgboost_model.json') # Reloading the model after training occured during k-fold-cv evaluation
    return avg

# Method for obtaining a test error based on a given testing set
def get_test_error(features, labels):
    return cost(features, labels)

# Method for obtaining a prediction on the given dataset:
def get_prediction(features):
    return xgboost_reg.predict(features).reshape(features.shape[0], 1)
