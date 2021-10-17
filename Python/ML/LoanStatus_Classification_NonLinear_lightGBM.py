# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
##################### Classification - Nonlinear ##############################
#####################     light GBM Methods      ##############################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK
import lightgbm as lgb
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin, tpe, Trials
import csv
from timeit import default_timer as timer
import ast
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

path = r'D:\Loan-Status\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_NonLinear_lightGBM'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Read data
df = pd.read_csv('LendingTree_LoanStatus_final.csv', low_memory=False)
df = df.drop_duplicates()
print(df.shape)

###############################################################################
########################## Resampling Techniques ##############################
###############################################################################
########################   1. Oversample minority class #######################
######################## 2. Split over upsampling with SMOTE  #################
###############################################################################
###############################################################################
# Separate input features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

###############################################################################
########################   1. Oversample minority class #######################
###############################################################################
# Setting up testing and training sets for oversampling minor class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed_value)

# Concatenate training data back together
df1 = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes
current = df1[df1.loan_status==0]
default = df1[df1.loan_status==1]

del df1

# Upsample minority
default_upsampled = resample(default,
                          replace=True, # sample with replacement
                          n_samples=len(current), # match number in majority 
                          random_state=seed_value) 

# Combine majority and upsampled minority
upsampled = pd.concat([current, default_upsampled])

del default_upsampled, current, default

# Examine counts of new class
print('\nExamine Loan Status after oversampling minority class') 
print(upsampled.loan_status.value_counts())
print('======================================================================')

# Separate input features and target of upsampled train data
X_train = upsampled.drop('loan_status', axis=1)
y_train = upsampled.loan_status

del upsampled

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###############################################################################
######################## 2. Split over upsampling with SMOTE  #################
###############################################################################
# Setting up testing and training sets for upsampling with SMOTE
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed_value)

smote = SMOTE(sampling_strategy='minority', random_state=seed_value)
X1_train, y1_train = smote.fit_sample(X1_train, y1_train)

print('\nExamine Loan Status after upsampling with SMOTE') 
print(y1_train.value_counts())
print('======================================================================')

# Feature Scaling
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)

# Change path to Results from Machine Learning
path = r'D:\Loan-Status\Python\ML_Results'
os.chdir(path)

###############################################################################
######################## light GBM HPO for Upsampling Set #####################
###############################################################################
# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label = y_train, params=params)

# Define an objective function
NUM_EVAL = 100
N_FOLDS = 3

#UserWarning: Found `n_estimators` in params. Will use it instead of argument
#UserWarning: Early stopping is not available in dart mode
#    force_col_wise=true
def objective(params, n_folds = N_FOLDS):
    """Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for param_name in ['max_depth', 'num_leaves']:
        params[param_name] = int(params[param_name])
        
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 100, nfold = N_FOLDS, 
                        early_stopping_rounds = 10, metrics = 'auc',
                        seed = seed_value)
    
    run_time = timer() - start
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}
    
# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', "+"),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(5, 6, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 100, dtype=int)),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# Select the optimization algorithm
tpe_algorithm = tpe.suggest

# File to save results
out_file = 'lightGBM_HPO_Upsampling.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate=seed_value)

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_Upsampling_trials_results = sorted(bayesOpt_Upsampling_trials.results,
                                            key = lambda x: x['loss'])
print('Upsampling HPO: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_Upsampling.csv')

# Sort best scores
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Use the HPO from best model fit a model
best_bayes_Upsampling_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'binary',
                                      random_state = seed_value, **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'LoanStatus_lightGBM_Upsampling_BayesHyperopt.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from lightGBM HPO on Upsampling Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_Upsampling_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_Upsampling_HPO))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y_test, y_pred_Upsampling_HPO))
print('Precision score : %.3f'%precision_score(y_test, y_pred_Upsampling_HPO))
print('Recall score : %.3f'%recall_score(y_test, y_pred_Upsampling_HPO))
print('F1 score : %.3f'%f1_score(y_test, y_pred_Upsampling_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_Upsampling_model.predict_proba(X_test)[:, 1]

print('The best model from Upsampling Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_Upsampling.csv', index = False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize = (10, 5),
                                                      color = 'blue',
                                                      title = 'Bayes Optimization Boosting Type')

print('Upsampling Bayes Optimization boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

###############################################################################
######################### light GBM HPO for SMOTE Set #########################
###############################################################################
# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X1_train, label = y1_train, params=params)

# File to save results
out_file = 'lightGBM_HPO_SMOTE.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate=seed_value)

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results,
                                       key = lambda x: x['loss'])
print('SMOTE HPO: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_SMOTE.csv')

# Sort best scores
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Use the HPO from best model fit a model
best_bayes_SMOTE_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'binary',
                                      random_state = seed_value, **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'LoanStatus_lightGBM_SMOTE_BayesHyperopt.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from lightGBM HPO on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE_HPO))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y1_test, y_pred_SMOTE_HPO))
print('Precision score : %.3f'%precision_score(y1_test, y_pred_SMOTE_HPO))
print('Recall score : %.3f'%recall_score(y1_test,y_pred_SMOTE_HPO))
print('F1 score : %.3f'%f1_score(y1_test, y_pred_SMOTE_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_SMOTE_model.predict_proba(X1_test)[:, 1]

print('The best model from SMOTE Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_SMOTE.csv', index = False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize = (10, 5),
                                                      color = 'green',
                                                      title = 'Bayes Optimization Boosting Type')

print('SMOTE Bayes Optimization boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

###############################################################################