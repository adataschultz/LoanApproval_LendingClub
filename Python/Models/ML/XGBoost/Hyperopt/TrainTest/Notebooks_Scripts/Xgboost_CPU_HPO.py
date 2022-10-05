# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
##################### Classification - Nonlinear ##############################
#####################       XGBoost Methods      ##############################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import STATUS_OK
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import fmin, hp, tpe, Trials
import csv
from datetime import datetime, timedelta
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
import eli5
from eli5.sklearn import PermutationImportance 
from eli5 import show_weights
import webbrowser
from eli5.sklearn import explain_weights_sklearn
from eli5.formatters import format_as_dataframe, format_as_dataframes
from eli5 import show_prediction
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

path = r'D:\Loan-Status\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_XGBoost'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Read data
train_US = pd.read_csv('trainDF_US.csv', low_memory=False)
test_US = pd.read_csv('testDF_US.csv', low_memory=False)
train_SMOTE = pd.read_csv('trainDF_SMOTE.csv', low_memory=False)
test_SMOTE = pd.read_csv('testDF_SMOTE.csv', low_memory=False)

# Upsampling - Separate input features and target
X_train = train_US.drop('loan_status', axis=1)
y_train = train_US[['loan_status']]

X_test = test_US.drop('loan_status', axis=1)
y_test = test_US[['loan_status']]

# SMOTE - Separate input features and target
X1_train = train_SMOTE.drop('loan_status', axis=1)
y1_train = train_SMOTE[['loan_status']]

X1_test = test_SMOTE.drop('loan_status', axis=1)
y1_test = test_SMOTE[['loan_status']]

del train_US, test_US, train_SMOTE, test_SMOTE

###############################################################################
##############################  Baseline  #####################################
###############################################################################
# Set the baseline model for Upsampling
# Define model type
clf = XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    random_state=seed_value,
    booster='gbtree',   
    scale_pos_weight=1,
    use_label_encoder=False,
    verbosity=0)

# Fit the model to the data
clf.fit(X_train, y_train)

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Hyperopt\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'xgb_Upsampling_Baseline.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

# =============================================================================
# # To load saved model
# model = joblib.load('xgb_Upsampling_Baseline.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_US = clf.predict(X_test)

print('Results from XGBoost using Baseline on Upsampled data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_US)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_US))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y_test, y_pred_US))
print('Precision score : %.3f'%precision_score(y_test, y_pred_US))
print('Recall score : %.3f'%recall_score(y_test, y_pred_US))
print('F1 score : %.3f'%f1_score(y_test, y_pred_US))

###############################################################################
# Set baseline model for SMOTE
# Fit the model to the data
clf.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'xgb_SMOTE_Baseline.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

# =============================================================================
# # To load saved model
# model = joblib.load('xgb_SMOTE_Baseline.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_SMOTE = clf.predict(X_test)

print('Results from XGBoost using Baseline on SMOTE data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_SMOTE))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y_test, y_pred_SMOTE))
print('Precision score : %.3f'%precision_score(y_test, y_pred_SMOTE))
print('Recall score : %.3f'%recall_score(y_test, y_pred_SMOTE))
print('F1 score : %.3f'%f1_score(y_test, y_pred_SMOTE))

###############################################################################
######################### XGBoost HPO for Upsampling Set ######################
################################## 100 Trials #################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\trialOptions'
os.chdir(path)

# Define the number of trials
NUM_EVAL = 100

# Set same k-folds for reproducibility
kfolds = KFold(n_splits=3, shuffle=True, random_state=seed_value)

# Define parameter grid
xgb_tune_kwargs= {
    'n_estimators': hp.choice('n_estimators', np.arange(100, 500, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'subsample': hp.uniform('subsample', 0.25, 0.75),
    'gamma': hp.uniform('gamma', 0, 9),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.3),
    'reg_alpha' : hp.choice('reg_alpha', np.arange(0, 30, dtype=int)),
    'reg_lambda' : hp.uniform('reg_lambda', 0, 3),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel' : hp.uniform('colsample_bylevel', 0.05, 0.5),  
    'min_child_weight' : hp.choice('min_child_weight', np.arange(0, 10, dtype=int)),
    }

# Define a function for optimization of hyperparameters
def xgb_upsampling(config):
    """XGBoost HPO"""
    
    # Keep track of evaluations
    global ITERATION
    
    ITERATION += 1  
    
    # Parameters that are integers to remain integers
    config['n_estimators'] = int(config['n_estimators'])   
    
    # Start hyperopt at 3 for max_depth   
    config['max_depth'] = int(config['max_depth']) + 3
    
    # Define model type
    xgb = XGBClassifier( 
        objective='binary:logistic',
        n_jobs=-1,
        random_state=seed_value,
        booster='gbtree',   
        scale_pos_weight=1,
        use_label_encoder=False,
        verbosity=0,
        **config)
    
    # Start timer for each trial
    start = timer()
    
    # Perform k_folds cross validation to find lower error
    scores = -cross_val_score(xgb, 
                              X_train, 
                              y_train, 
                              scoring='roc_auc', 
                              cv=kfolds)
    
    run_time = timer() - start
    # Extract the best score
    best_score = np.max(scores)
    
    # Loss must be minimized
    loss = 1 - best_score

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, config, ITERATION, run_time])
    
    return {'loss': loss, 'params': config, 'iteration': ITERATION, 
            'train_time': run_time, 'status': STATUS_OK}    

# Optimization algorithm
tpe_algorithm = tpe.suggest

# File to save first results
out_file = 'XGBoost_HPO_Upsampling_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))

best_param = fmin(xgb_upsampling, xgb_tune_kwargs, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate= np.random.RandomState(42))

# End timer for experiment
end_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
print('%-20s %s' % ('End Time', end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_Upsampling_trials_results = sorted(bayesOpt_Upsampling_trials.results,
                                            key=lambda x: x['loss'])
print('Upsampling HPO 100 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Access results
results = pd.read_csv('XGBoost_HPO_Upsampling_100.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_PKL'
os.chdir(path)
                                      
# Re-create the best model and train on the training data
best_bayes_Upsampling_model = xgb.XGBClassifier(objective='binary:logistic', 
                                                booster='gbtree', 
                                                scale_pos_weight=1, 
                                                n_jobs=-1, 
                                                use_label_encoder=False, 
                                                verbosity=0, 
                                                random_state=seed_value, 
                                                **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'XGBoost_HPO_Upsampling_100.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('XGBoost_HPO_Upsampling_100.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for XGBoost HPO Upsampling 100trials')
y_train_pred = best_bayes_Upsampling_model.predict(X_train)
y_test_pred = best_bayes_Upsampling_model.predict(X_test)

# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from XGBoost HPO 100 on Upsampling Data:')
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

print('The best model from Upsampling Bayes 100 trials optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_XGBoost_HPO_Upsampling_100.csv', index=False)

# Convert data types for graphing
bayes_params['colsample_bylevel'] = bayes_params['colsample_bylevel'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['gamma'] = bayes_params['gamma'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', 
            linewidth=2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_child_weight', 'n_estimators']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'gamma', 'colsample_bylevel',
                         'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize=(14,6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_Upsampling_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)
X_train1 = pd.DataFrame(X_train, columns=X_train.columns)
X_test1 = pd.DataFrame(X_test, columns=X_test.columns)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_Upsampling_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_Upsampling_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_Upsampling_100_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1],
    predict_fn=best_bayes_Upsampling_model.predict_proba)

exp.save_to_file('best_bayes_Upsampling_100_LIME.html')

###############################################################################
######################### light GBM HPO for SMOTE Set #########################
################################## 100 Trials #################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\trialOptions'
os.chdir(path)

# Define a function for optimization of hyperparameters
def xgb_smote(config):
    """XGBoost HPO"""
    
    # Keep track of evaluations
    global ITERATION
    
    ITERATION += 1  
    
    # Parameters that are integers to remain integers
    config['n_estimators'] = int(config['n_estimators'])   
    
    # Start hyperopt at 3 for max_depth   
    config['max_depth'] = int(config['max_depth']) + 3
    
    # Define model type
    xgb = XGBClassifier(
        objective='binary:logistic',
        n_jobs=-1,
        random_state=seed_value,
        booster='gbtree',   
        scale_pos_weight=1,
        use_label_encoder=False,
        verbosity=0,
        **config)
    
    # Start timer for each trial
    start = timer()
    
    # Perform k_folds cross validation to find lower error
    scores = -cross_val_score(xgb, 
                              X1_train, 
                              y1_train,
                              scoring='roc_auc',
                              cv=kfolds) 
    
    run_time = timer() - start
    # Extract the best score
    best_score = np.max(scores)
    
    # Loss must be minimized
    loss = 1 - best_score

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, config, ITERATION, run_time])
    
    return {'loss': loss, 'params': config, 'iteration': ITERATION, 
            'train_time': run_time, 'status': STATUS_OK}    

# Optimization algorithm
tpe_algorithm = tpe.suggest

# File to save first results
out_file = 'XGBoost_HPO_SMOTE_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))

best_param = fmin(xgb_smote, xgb_tune_kwargs, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate= np.random.RandomState(42))

# End timer for experiment
end_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
print('%-20s %s' % ('End Time', end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results, 
                                       key=lambda x: x['loss'])
print('SMOTE HPO 100 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('XGBoost_HPO_SMOTE_100.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
    
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_PKL'
os.chdir(path)
                                  
# Re-create the best model and train on the training data
best_bayes_SMOTE_model = xgb.XGBClassifier(objective='binary:logistic', 
                                           booster='gbtree', 
                                           scale_pos_weight=1, 
                                           use_label_encoder=False, 
                                           verbosity=0, 
                                           random_state=seed_value, 
                                           n_jobs=-1, 
                                           **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'XGBoost_HPO_SMOTE_100.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('XGBoost_HPO_SMOTE_100.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for XGBoost HPO SMOTE 100trials')
y_train_pred = best_bayes_SMOTE_model.predict(X1_train)
y_test_pred = best_bayes_SMOTE_model.predict(X1_test)

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from XGBoost HPO 100 trials on Upsampling Data:')
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

print('The best model from SMOTE Bayes optimization 100 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_XGBoost_HPO_SMOTE_100.csv', index=False)

# Convert data types for graphing
bayes_params['colsample_bylevel'] = bayes_params['colsample_bylevel'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['gamma'] = bayes_params['gamma'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', 
            linewidth=2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_child_weight', 'n_estimators']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'gamma', 'colsample_bylevel',
                         'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize=(14,6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_SMOTE_model,
                                        random_state=seed_value).fit(X1_test,
                                                                     y1_test)

X1_train1 = pd.DataFrame(X1_train, columns=X1_train.columns)
X1_test1 = pd.DataFrame(X1_test, columns=X1_test.columns)
                                                                     
# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X1_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_SMOTE_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_SMOTE_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_SMOTE_100_WeightsExplain.csv', index=False)
           
###############################################################################
# LIME for model explanation                                                                 
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test1.iloc[1],
    predict_fn=best_bayes_Upsampling_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_100_LIME.html')

###############################################################################
######################### XGBoost HPO for Upsampling Set ######################
################################## 300 Trials #################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\trialOptions'
os.chdir(path)

# Define the number of trials
NUM_EVAL = 300

# File to save first results
out_file = 'XGBoost_HPO_Upsampling_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))

best_param = fmin(xgb_upsampling, xgb_tune_kwargs, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate= np.random.RandomState(42))

# End timer for experiment
end_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
print('%-20s %s' % ('End Time', end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_Upsampling_trials_results = sorted(bayesOpt_Upsampling_trials.results,
                                            key=lambda x: x['loss'])
print('Upsampling HPO 100 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Access results
results = pd.read_csv('XGBoost_HPO_Upsampling_300.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
                                    
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_PKL'
os.chdir(path)
  
# Re-create the best model and train on the training data
best_bayes_Upsampling_model = xgb.XGBClassifier(objective='binary:logistic', 
                                                booster='gbtree', 
                                                scale_pos_weight=1, 
                                                n_jobs=-1, 
                                                use_label_encoder=False, 
                                                verbosity=0, 
                                                random_state=seed_value, 
                                                **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'XGBoost_HPO_Upsamplingt_300.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('XGBoost_HPO_Upsamplingt_300.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for XGBoost HPO Upsampling 300trials')
y_train_pred = best_bayes_Upsampling_model.predict(X_train)
y_test_pred = best_bayes_Upsampling_model.predict(X_test)

# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from XGBoost HPO 300 on Upsampling Data:')
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

print('The best model from Upsampling Bayes 300 trials optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_XGBoost_HPO_Upsampling_300.csv', 
                    index=False)

# Convert data types for graphing
bayes_params['colsample_bylevel'] = bayes_params['colsample_bylevel'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['gamma'] = bayes_params['gamma'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', 
            linewidth=2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_child_weight', 'n_estimators']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'gamma', 'colsample_bylevel',
                         'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize=(14,6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_Upsampling_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_Upsampling_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_Upsampling_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_Upsampling_300_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1],
    predict_fn=best_bayes_Upsampling_model.predict_proba)
exp.save_to_file('best_bayes_Upsampling_300_LIME.html')

###############################################################################
######################### light GBM HPO for SMOTE Set #########################
################################## 300 Trials #################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\trialOptions'
os.chdir(path)

# File to save first results
out_file = 'XGBoost_HPO_SMOTE_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))

best_param = fmin(xgb_smote, xgb_tune_kwargs, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate= np.random.RandomState(42))

# End timer for experiment
end_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
print('%-20s %s' % ('End Time', end_time))
print(str(timedelta(seconds=(end_time-start_time).seconds)))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results,
                                       key=lambda x: x['loss'])
print('SMOTE HPO 300 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('XGBoost_HPO_SMOTE_300.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
         
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_PKL'
os.chdir(path)
                            
# Re-create the best model and train on the training data
best_bayes_SMOTE_model = xgb.XGBClassifier(objective='binary:logistic', 
                                           booster='gbtree', 
                                           scale_pos_weight=1, 
                                           n_jobs=-1, 
                                           use_label_encoder=False, 
                                           verbosity=0, 
                                           random_state=seed_value, 
                                           **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'XGBoost_HPO_SMOTE_300.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('XGBoost_HPO_SMOTE_300.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for XGBoost HPO SMOTE 300trials')
y_train_pred = best_bayes_SMOTE_model.predict(X1_train)
y_test_pred = best_bayes_SMOTE_model.predict(X1_test)

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from XGBoost HPO 300 trials on SMOTE Data:')
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

print('The best model from SMOTE Bayes optimization 300 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_XGBoost_HPO_SMOTE_300.csv', 
                    index=False)

# Convert data types for graphing
bayes_params['colsample_bylevel'] = bayes_params['colsample_bylevel'].astype('float64')
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['gamma'] = bayes_params['gamma'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', 
            linewidth=2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric 
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_child_weight', 'n_estimators']: 
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'gamma', 'colsample_bylevel',
                         'colsample_bytree']):
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize=(14,6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_SMOTE_model,
                                        random_state=seed_value).fit(X1_test,
                                                                     y1_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X1_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_SMOTE_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\Models\ML\XGBoost\Model_Explanations\best_bayes_SMOTE_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_SMOTE_300_WeightsExplain.csv', index=False) 
           
###############################################################################
# LIME for model explanation                                                           
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test1.iloc[1],
    predict_fn=best_bayes_Upsampling_model.predict_proba)

exp.save_to_file('best_bayes_SMOTE_300_LIME.html')

###############################################################################