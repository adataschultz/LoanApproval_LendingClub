# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
########################## Lending Tree Loan Status ###########################
######################### Classification - Nonlinear ##########################
###########################  light GBM Methods  ###############################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials
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
os.environ['LoanStatus_lightGBM'] = str(seed_value)
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
clf = lgb.LGBMClassifier()

# Fit the model to the data
clf.fit(X_train, y_train)

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'lightGBM_Upsampling_Baseline.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_Upsampling_Baseline.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_US = clf.predict(X_test)

print('Results from lightGBM using Baseline on Upsampled data:')
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
Pkl_Filename = 'lightGBM_SMOTE_Baseline.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_SMOTE_Baseline.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_SMOTE = clf.predict(X_test)

print('Results from lightGBM using Baseline on SMOTE data:')
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
######################## light GBM HPO for Upsampling Set #####################
################################## 100 Trials #################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label=y_train, params=params)

# Define an lgb_hpo function
NUM_EVAL = 100
N_FOLDS = 3

def lgb_hpo(params, n_folds=N_FOLDS):
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
    cv_results = lgb.cv(params, train_set, num_boost_round=100, nfold=N_FOLDS, 
                        early_stopping_rounds=10, metrics='auc', seed=seed_value)
    
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
    'force_col_wise': hp.choice('force_col_wise', '+'),
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
out_file = 'lightGBM_HPO_Upsampling_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_Upsampling_trials_results = sorted(bayesOpt_Upsampling_trials.results,
                                            key=lambda x: x['loss'])
print('Upsampling HPO 100 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_Upsampling_100.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_Upsampling_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                                 n_jobs=-1, 
                                                 lgb_hpo='binary',
                                                 random_state=seed_value,
                                                 **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_Upsampling_100.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_Upsampling_100.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from lightGBM HPO 100 trials on Upsampling data:')
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

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_Upsampling_100.csv', index=False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize=(10,5),
                                                      color='blue',
                                                      title='Bayes Optimization Boosting Type')

print('Upsampling Bayes Optimization 100 trials boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

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
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 
                                                                      'dart': 1, 
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'max_depth', 
                         'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title ='{} over Trials'.format(hpo))
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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
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
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_Upsampling_model, X_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_100_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_100_Prediction.htm'
webbrowser.open(url2, new=2)

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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X1_train, label=y1_train, params=params)

# File to save results
out_file = 'lightGBM_HPO_SMOTE_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results,
                                       key=lambda x: x['loss'])
print('SMOTE HPO 100 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_SMOTE_100.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_SMOTE_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                            n_jobs=-1, 
                                            lgb_hpo='binary',
                                            random_state=seed_value,
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_SMOTE_100.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_SMOTE_100.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from lightGBM HPO 100 trials on SMOTE Data:')
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

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_SMOTE_100.csv', index=False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize=(10,5), 
                                                      color='green',
                                                      title='Bayes Optimization Boosting Type')

print('SMOTE Bayes Optimization 100 trials boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

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
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 
                                                                      'dart': 1, 
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'max_depth', 
                         'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title ='{} over Trials'.format(hpo))
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
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
os.chdir(path)

X1_train1 = pd.DataFrame(X1_train, columns=X1_train.columns)
X1_test1 = pd.DataFrame(X1_test, columns=X1_test.columns)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_SMOTE_model,
                                        random_state=seed_value).fit(X1_test,
                                                                     y1_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X1_test1.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_SMOTE_model, X1_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_100_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_100_Prediction.htm'
webbrowser.open(url2, new=2)

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
    predict_fn=best_bayes_SMOTE_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_100_LIME.html')

###############################################################################
######################## light GBM HPO for Upsampling Set #####################
################################# 500 Trials ##################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label=y_train, params=params)

# Define an lgb_hpo function
NUM_EVAL = 500
    
# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', '+'),
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
out_file = 'lightGBM_HPO_Upsampling_500.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_Upsampling_trials_results = sorted(bayesOpt_Upsampling_trials.results,
                                            key=lambda x: x['loss'])
print('Upsampling HPO 500 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_Upsampling_500.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_Upsampling_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                                 n_jobs=-1, 
                                                 lgb_hpo='binary',
                                                 random_state=seed_value, 
                                                 **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_Upsampling_500.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_Upsampling_500.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from lightGBM HPO 500 Trials on Upsampling Data:')
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

print('The best model from Upsampling Bayes 500 trials optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(colum0ns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_Upsampling_500.csv', index=False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize=(10,5),
                                                      color='blue',
                                                      title='Bayes Optimization Boosting Type')

print('Upsampling Bayes 500 Trials Optimization boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', linewidth=2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 
                                                                      'dart': 1, 
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'max_depth', 
                         'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title ='{} over Trials'.format(hpo))
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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
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
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_500_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_500_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_Upsampling_model, X_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_500_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_500_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_Upsampling_500_WeightsExplain.csv', index=False)

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
exp.save_to_file('best_bayes_Upsampling_500_LIME.html')

###############################################################################
################# light GBM using GBDT HPO for Upsampling Set #################
############################# 300 Trials ######################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# GBDT has lowest loss for Upsampling initial exploration
# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label=y_train, params=params)

NUM_EVAL = 300

param_grid = {
    'force_col_wise': hp.choice('force_col_wise', '+'),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 0.8),
    'bagging_frequency': hp.uniform('bagging_frequency', 5, 8),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.8),
    'min_sum_hessian_in_leaf': hp.choice('min_sum_hessian_in_leaf',  np.arange(0.1, 1, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(3, 15, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 200, dtype=int)),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.3, 1)}]),
    'colsample_bytree': hp.choice('colsample_by_tree', np.arange(1, 7, dtype=int)),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# File to save results
out_file = 'lightGBM_GBDT_HPO_Upsampling_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Select the optimization algorithm
tpe_algorithm = tpe.suggest

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_Upsampling_trials_results = sorted(bayesOpt_Upsampling_trials.results,
                                            key=lambda x: x['loss'])
print('Upsampling HPO GBDT 300 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_GBDT_HPO_Upsampling_300.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_Upsampling_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                                 n_jobs=-1, 
                                                 lgb_hpo='binary',
                                                 random_state=seed_value, 
                                                 **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_GBDT_Upsampling_300.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_GBDT_Upsampling_300.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from lightGBM GBDT HPO 300 trials on Upsampling Data:')
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

print('The best model from Upsampling GBDT Bayes optimization 300 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['feature_fraction'] = bayes_params['feature_fraction'].astype('float64')
bayes_params['bagging_fraction'] = bayes_params['bagging_fraction'].astype('float64')
bayes_params['bagging_frequency'] = bayes_params['bagging_frequency'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_GBDT_Upsampling_300.csv', index=False)

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', 
            linewidth=2)
plt.legend(loc=1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'gbdt': 0})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0], ['gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitive hyperparameters
fig, axs = plt.subplots(1, 6, figsize=(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'num_leaves',
                         'bagging_fraction', 'bagging_frequency',
                         'feature_fraction']):
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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
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
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_GBDT_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_GBDT_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_Upsampling_model, X_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_GBDT_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_Upsampling_GBDT_300_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_Upsampling_GBDT_300_WeightsExplain.csv', index=False)

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
exp.save_to_file('best_bayes_Upsampling_GBDT_300_LIME.html')

###############################################################################
######################### light GBM HPO for SMOTE Set #########################
################################## 300 Trials #################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X1_train, label=y1_train, params=params)

NUM_EVAL = 300

# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', '+'),
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
# File to save results
out_file = 'lightGBM_HPO_SMOTE_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results,
                                       key=lambda x: x['loss'])
print('SMOTE HPO 300 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_SMOTE_300.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_SMOTE_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                            n_jobs=-1, 
                                            lgb_hpo='binary',
                                            random_state=seed_value, 
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_SMOTE_300.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_SMOTE_300.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from lightGBM HPO 300 trials on SMOTE Data:')
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

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_SMOTE_300.csv', index=False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize=(10,5),
                                                      color='green',
                                                      title='Bayes Optimization Boosting Type')

print('SMOTE Bayes Optimization 300 trials boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

# Density plots of the learning rate distributions 
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', 
            linewidth=2)
plt.legend(loc=1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 
                                                                      'dart': 1, 
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'max_depth', 
                         'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title ='{} over Trials'.format(hpo))
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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
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
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_SMOTE_model, X1_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_300_Prediction.htm'
webbrowser.open(url2, new=2)

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
    predict_fn=best_bayes_SMOTE_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_300_LIME.html')

###############################################################################
######################### light GBM HPO for SMOTE Set #########################
################################# 500 Trials ##################################
###############################################################################
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X1_train, label=y1_train, params=params)

NUM_EVAL = 500

# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', '+'),
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
# File to save results
out_file = 'lightGBM_HPO_SMOTE_500.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results,
                                       key=lambda x: x['loss'])
print('SMOTE HPO 500 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_SMOTE_500.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_SMOTE_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                            n_jobs=-1, 
                                            lgb_hpo='binary',
                                            random_state=seed_value, 
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_SMOTE_500.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_SMOTE_500.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from lightGBM HPO 500 trials on SMOTE Data:')
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

print('The best model from SMOTE Bayes optimization 500 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_SMOTE_500.csv', index=False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize=(10,5),
                                                      color='green',
                                                      title='Bayes Optimization Boosting Type')

print('SMOTE Bayes Optimization 500 trials boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

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
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 
                                                                      'dart': 1, 
                                                                      'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize=(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'max_depth', 
                         'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
                   title ='{} over Trials'.format(hpo))
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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
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
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_SMOTE_model, X1_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_SMOTE_500_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test1.iloc[1],
    predict_fn=best_bayes_SMOTE_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_500_LIME.html')

###############################################################################
######################### light GBM HPO for SMOTE Set #########################
################################## GOSS & DART ################################
################################# 500 Trials ##################################
###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X1_train, label=y1_train, params=params)

# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', '+'),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 150, dtype=int)),
    'min_sum_hessian_in_leaf': hp.choice('min_sum_hessian_in_leaf',  np.arange(0.1, 1, dtype=int)),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.1, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# File to save results
out_file = 'lightGBM_HPO_SMOTE_500_2.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

best_param = fmin(lgb_hpo, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) 
bayesOpt_SMOTE_trials_results = sorted(bayesOpt_SMOTE_trials.results,
                                       key=lambda x: x['loss'])
print('SMOTE HPO 500 trials GOSS DART: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Read results from csv
results = pd.read_csv('lightGBM_HPO_SMOTE_500_2.csv')

# Sort best scores
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the number of estimators and hyperparameters from best AUC
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_PKL'
os.chdir(path)

# Use the HPO from best model fit a model
best_bayes_SMOTE_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 
                                            n_jobs=-1, 
                                            lgb_hpo='binary',
                                            random_state=seed_value, 
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_SMOTE_500_2.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_SMOTE_500_2.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from lightGBM HPO 500 trials GOSS DART on SMOTE Data:')
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

print('The best model from SMOTE Bayes optimization 500 trials GOSS DART scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns=list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index=list(range(len(results))))

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\bayesParams'
os.chdir(path)

# Save parameters to df
bayes_params.to_csv('bayes_params_HPO_SMOTE_500_2.csv', index=False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize=(10,5),
                                                      color='green',
                                                      title='Bayes Optimization Boosting Type')

print('SMOTE Bayes Optimization 500 trials GOSS DART boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

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
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise', 
                   'max_depth', 'min_sum_hessian_in_leaf']:
        plt.figure(figsize=(14,6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc=0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 
                                                                      'dart': 1})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1], ['goss', 'dart']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize =(20,5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'num_leaves']):
    
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
path = r'D:\Loan-Status\Python\Models\ML\lightGBM\Hyperopt\Model_Explanations'
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
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_2_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_2_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_SMOTE_model, X1_test1.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_2_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\lightGBM\Model_Explanations\best_bayes_SMOTE_500_2_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_SMOTE_500_2_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test1.iloc[1],
    predict_fn=best_bayes_SMOTE_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_500_2_LIME.html')

###############################################################################