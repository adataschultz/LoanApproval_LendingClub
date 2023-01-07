# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
########################## Lending Tree Loan Status ###########################
######################### Classification - Nonlinear ##########################
############################# Catboost Methods ################################
###############################################################################
import os
import random
import numpy as np
import warnings
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from catboost import CatBoostClassifier
import csv
from datetime import datetime, timedelta
from timeit import default_timer as timer
import ast
import pickle
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance 
import webbrowser
from eli5.formatters import format_as_dataframe
from lime import lime_tabular
warnings.filterwarnings('ignore')
my_dpi = 96

path = r'D:\LoanStatus\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_NonLinear_Catboost'] = str(seed_value)
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
# Set baseline model for Upsampling
cat = CatBoostClassifier(loss_function='Logloss', 
                         eval_metric='AUC', 
                         early_stopping_rounds=10, 
                         logging_level='Silent',
                         random_state=seed_value)

# Fit the model to the data
cat.fit(X_train, y_train)

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'Catboost_Upsampling_Baseline.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(cat, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_Upsampling_Baseline.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_US = cat.predict(X_test)

print('Results from Catboost using Baseline on Upsampled data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_US)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_US))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_US))
print('Precision score : %.3f' % precision_score(y_test, y_pred_US))
print('Recall score : %.3f' % recall_score(y_test, y_pred_US))
print('F1 score : %.3f'% f1_score(y_test, y_pred_US))

###############################################################################
# Set baseline model for SMOTE
cat.fit(X1_train, y1_train)
    
# Save model
Pkl_Filename = 'Catboost_SMOTE_Baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(cat, file)

# Predict based on training 
y_pred_SMOTE = cat.predict(X1_test)

print('Results from Catboost using Baseline on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y1_test, y_pred_SMOTE))
print('Precision score : %.3f' % precision_score(y1_test, y_pred_SMOTE))
print('Recall score : %.3f' % recall_score(y1_test, y_pred_SMOTE))
print('F1 score : %.3f' % f1_score(y1_test, y_pred_SMOTE))

###############################################################################
####################### Catboost HPO for Upsampling Set #######################
############################## 100 Trials #####################################
###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\trialOptions'
os.chdir(path)

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('Catboost_HPO_Upsampling_100trials_log.txt', 'w')

print('\nLoanStatus_CatboostHPO_Upsampling') 
print('======================================================================')

# Define the number of trials
NUM_EVAL = 100

# Set same k-folds for reproducibility
kfolds = KFold(n_splits=3, shuffle=True, random_state=seed_value)

# Define parameter grid
catboost_tune_kwargs= {
    'iterations': hp.choice('iterations', np.arange(100, 500, dtype=int)),
    'depth': hp.choice('depth', np.arange(3, 10, dtype=int)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1e-2, 1e0), 
    'learning_rate': hp.uniform('learning_rate', 1e-4, 0.3),                              
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(2, 20, 
                                                                dtype=int)),
    'one_hot_max_size': hp.choice('one_hot_max_size', np.arange(2, 20, 
                                                                dtype=int)),  
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1e-2, 1.0)
    }

# Define a function for optimization of hyperparameters
def catboost_hpo_us(config):
    """Catboost HPO"""
    
    # Keep track of evaluations
    global ITERATION
    
    ITERATION += 1  
    
    # Parameters that are integers to remain integers
    config['iterations'] = int(config['iterations'])   
    
    # Start hyperopt at 3 for max_depth   
    config['depth'] = int(config['depth']) + 3
 
    # Define model type
    cat = CatBoostClassifier(
        loss_function='Logloss', 
        eval_metric='AUC',
        early_stopping_rounds=10,
        random_state=seed_value,
        logging_level='Silent',
        **config)
    
    # Start timer for each trial
    start = timer()
    
    # Perform k_folds cross validation to find lower error
    scores = -cross_val_score(cat, X_train, y_train, scoring='roc_auc', 
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
out_file = 'Catboost_HPO_Upsampling_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_Upsampling_trials = Trials()

# Begin HPO trials for Upsampling data
# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
best_param = fmin(catboost_hpo_us, catboost_tune_kwargs, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_Upsampling_trials,
                  rstate=np.random.RandomState(42))

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
results = pd.read_csv('Catboost_HPO_Upsampling_100.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_Upsampling_model = CatBoostClassifier(loss_function='Logloss', 
                                                 eval_metric='AUC',
                                                 early_stopping_rounds=10,
                                                 logging_level='Silent', 
                                                 random_state=seed_value,
                                                 **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'Catboost_HPO_Upsampling_100.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_HPO_Upsampling_100.pkl')
# print(model)
# =============================================================================

print('\nModel Metrics for Catboost HPO Upsampling 100trials')
# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from Catboost HPO 100 on Upsampling Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_Upsampling_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_Upsampling_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_Upsampling_HPO))
print('Precision score : %.3f' % precision_score(y_test, y_pred_Upsampling_HPO))
print('Recall score : %.3f' % recall_score(y_test, y_pred_Upsampling_HPO))
print('F1 score : %.3f' % f1_score(y_test, y_pred_Upsampling_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_Upsampling_model.predict_proba(X_test)[:, 1]

print('The best model from Upsampling Bayes 100 trials optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, 
                                                                                                                                 preds)))
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
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_Catboost_HPO_Upsampling_100.csv', 
                    index=False)

# Convert data types for graphing
bayes_params['depth'] = bayes_params['depth'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['l2_leaf_reg'] = bayes_params['l2_leaf_reg'].astype('float64')
bayes_params['min_data_in_leaf'] = bayes_params['min_data_in_leaf'].astype('float64')
bayes_params['one_hot_max_size'] = bayes_params['one_hot_max_size'].astype('float64')

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
    if hpo not in ['iteration', 'scale_pos_weight', 'iterations']:
        plt.figure(figsize=(14,6))
        # Plot the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'min_data_in_leaf', 
                         'one_hot_max_size']): 
  # Scatterplot
  sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
  axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
             title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
ax = sns.regplot('iteration', 'l2_leaf_reg', data=bayes_params, 
                 label='Bayes Optimization') 
ax.set(xlabel='Iteration', ylabel='l2_leaf_reg')                 
plt.tight_layout()
plt.show()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_Upsampling_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_Upsampling_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_Upsampling_100_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance, 
                                           feature_names=X_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_Upsampling_100_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
X_train1 = pd.DataFrame(X_train, columns=X_train.columns)    
X_test1 = pd.DataFrame(X_test, columns=X_test.columns)                                                                    

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
# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin

###############################################################################
####################### Catboost HPO for SMOTE Set ############################
############################## 100 Trials #####################################
###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\trialOptions'
os.chdir(path)

# Define a function for optimization of hyperparameters
def catboost_hpo_smote(config):
    """Catboost HPO"""
    
    # Keep track of evaluations
    global ITERATION
    
    ITERATION += 1  
    
    # Parameters that are integers to remain integers
    config['iterations'] = int(config['iterations'])   
    
    # Start hyperopt at 3 for max_depth   
    config['depth'] = int(config['depth']) + 3
 
    # Define model type
    cat = CatBoostClassifier(
        random_state=seed_value,
        loss_function='Logloss', 
        eval_metric='AUC',
        early_stopping_rounds=10,
        logging_level='Silent',
        **config)
    
    # Start timer for each trial
    start = timer()
    
    # Perform k_folds cross validation to find lower error
    scores = -cross_val_score(cat, X1_train, y1_train, 
                              scoring='roc_auc', cv=kfolds)
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

# File to save first results
out_file = 'Catboost_HPO_SMOTE_100.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0
bayesOpt_SMOTE_trials = Trials()

# Begin HPO trials for Upsampling data
# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
best_param = fmin(catboost_hpo_smote, catboost_tune_kwargs, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_SMOTE_trials,
                  rstate=np.random.RandomState(42))

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

# Access results
results = pd.read_csv('Catboost_HPO_SMOTE_100.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_SMOTE_model = CatBoostClassifier(loss_function='Logloss', 
                                            eval_metric='AUC',
                                            early_stopping_rounds=10,
                                            logging_level='Silent', 
                                            random_state=seed_value,
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'Catboost_HPO_SMOTE_100.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_HPO_SMOTE_100.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for Catboost HPO SMOTE 100trials')
# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from Catboost HPO 100 trials on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y1_test, y_pred_SMOTE_HPO))
print('Precision score : %.3f' % precision_score(y1_test, y_pred_SMOTE_HPO))
print('Recall score : %.3f' % recall_score(y1_test,y_pred_SMOTE_HPO))
print('F1 score : %.3f' % f1_score(y1_test, y_pred_SMOTE_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_SMOTE_model.predict_proba(X1_test)[:, 1]

print('The best model from SMOTE Bayes optimization 100 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, 
                                                                                                                            preds)))
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
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_Catboost_HPO_SMOTE_100.csv', index=False)

# Convert data types for graphing
bayes_params['depth'] = bayes_params['depth'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['l2_leaf_reg'] = bayes_params['l2_leaf_reg'].astype('float64')
bayes_params['min_data_in_leaf'] = bayes_params['min_data_in_leaf'].astype('float64')
bayes_params['one_hot_max_size'] = bayes_params['one_hot_max_size'].astype('float64')

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
    if hpo not in ['iteration', 'scale_pos_weight', 'iterations']:
        plt.figure(figsize=(14,6))
        # Plot the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'min_data_in_leaf', 
                         'one_hot_max_size']): 
  # Scatterplot
  sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
  axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
             title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
ax = sns.regplot('iteration', 'l2_leaf_reg', data=bayes_params, 
                 label='Bayes Optimization') 
ax.set(xlabel='Iteration', ylabel='l2_leaf_reg')                 
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations'
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
with open(r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_SMOTE_100_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_SMOTE_100_WeightsFeatures.htm'
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
    predict_fn=best_bayes_SMOTE_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_100_LIME.html')

###############################################################################
###################### Catboost HPO for Upsampling Set ########################
############################## 300 Trials #####################################
###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\trialOptions'
os.chdir(path)

# Define the number of trials
NUM_EVAL = 300

# File to save first results
out_file = 'Catboost_HPO_Upsampling_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0

# Begin HPO trials for Upsampling data
# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
best_param = fmin(catboost_hpo_us, catboost_tune_kwargs, algo=tpe.suggest,
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
print('Upsampling HPO 300 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Access results
results = pd.read_csv('Catboost_HPO_Upsampling_300.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_Upsampling_model = CatBoostClassifier(loss_function='Logloss', 
                                                 eval_metric='AUC',
                                                 early_stopping_rounds=10,
                                                 logging_level='Silent', 
                                                 random_state=seed_value,
                                                 **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'Catboost_HPO_Upsampling_300.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_HPO_Upsampling_300.pkl')
# print(model)
# =============================================================================

print('\nModel Metrics for Catboost HPO Upsampling 300trials')
# Predict based on training
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from Catboost HPO 300 on Upsampling Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_Upsampling_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_Upsampling_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_Upsampling_HPO))
print('Precision score : %.3f' % precision_score(y_test, y_pred_Upsampling_HPO))
print('Recall score : %.3f' % recall_score(y_test, y_pred_Upsampling_HPO))
print('F1 score : %.3f' % f1_score(y_test, y_pred_Upsampling_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_Upsampling_model.predict_proba(X_test)[:, 1]

print('The best model from Upsampling Bayes 300 trials optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, 
                                                                                                                                 preds)))
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
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_Catboost_HPO_Upsampling_300.csv', 
                    index=False)

# Convert data types for graphing
bayes_params['depth'] = bayes_params['depth'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['l2_leaf_reg'] = bayes_params['l2_leaf_reg'].astype('float64')
bayes_params['min_data_in_leaf'] = bayes_params['min_data_in_leaf'].astype('float64')
bayes_params['one_hot_max_size'] = bayes_params['one_hot_max_size'].astype('float64')

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
    if hpo not in ['iteration', 'scale_pos_weight', 'iterations']:
        plt.figure(figsize=(14,6))
        # Plot the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'min_data_in_leaf', 
                         'one_hot_max_size']): 
  # Scatterplot
  sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
  axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
             title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
ax = sns.regplot('iteration', 'l2_leaf_reg', data=bayes_params, 
                 label='Bayes Optimization') 
ax.set(xlabel='Iteration', ylabel='l2_leaf_reg')                 
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations'
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
with open(r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_Upsampling_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_Upsampling_300_WeightsFeatures.htm'
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
######################### Catboost HPO for SMOTE Set ##########################
############################## 300 Trials #####################################
###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\trialOptions'
os.chdir(path)

# File to save first results
out_file = 'Catboost_HPO_SMOTE_300.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0

# Begin HPO trials for Upsampling data
# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
best_param = fmin(catboost_hpo_smote, catboost_tune_kwargs, algo=tpe.suggest,
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
print('SMOTE HPO 500 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Access results
results = pd.read_csv('Catboost_HPO_SMOTE_300.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_SMOTE_model = CatBoostClassifier(loss_function='Logloss', 
                                            eval_metric='AUC',
                                            early_stopping_rounds=10,
                                            logging_level='Silent', 
                                            random_state=seed_value,
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'Catboost_HPO_SMOTE_300.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_HPO_SMOTE_300.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for Catboost HPO SMOTE 300trials')
# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from Catboost HPO 300 trials on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y1_test, y_pred_SMOTE_HPO))
print('Precision score : %.3f' % precision_score(y1_test, y_pred_SMOTE_HPO))
print('Recall score : %.3f' % recall_score(y1_test,y_pred_SMOTE_HPO))
print('F1 score : %.3f' % f1_score(y1_test, y_pred_SMOTE_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_SMOTE_model.predict_proba(X1_test)[:, 1]

print('The best model from SMOTE Bayes optimization 300 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, 
                                                                                                                            preds)))
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
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_Catboost_HPO_SMOTE_300.csv', index=False)

# Convert data types for graphing
bayes_params['depth'] = bayes_params['depth'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['l2_leaf_reg'] = bayes_params['l2_leaf_reg'].astype('float64')
bayes_params['min_data_in_leaf'] = bayes_params['min_data_in_leaf'].astype('float64')
bayes_params['one_hot_max_size'] = bayes_params['one_hot_max_size'].astype('float64')

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
    if hpo not in ['iteration', 'scale_pos_weight', 'iterations']:
        plt.figure(figsize=(14,6))
        # Plot the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'min_data_in_leaf', 
                         'one_hot_max_size']): 
  # Scatterplot
  sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
  axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
             title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
ax = sns.regplot('iteration', 'l2_leaf_reg', data=bayes_params, 
                 label='Bayes Optimization') 
ax.set(xlabel='Iteration', ylabel='l2_leaf_reg')                 
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations'
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
with open(r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_SMOTE_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_SMOTE_300_WeightsFeatures.htm'
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
    predict_fn=best_bayes_SMOTE_model.predict_proba)
exp.save_to_file('best_bayes_SMOTE_300_LIME.html')

###############################################################################
###################### Catboost HPO for Upsampling Set ########################
############################## 500 Trials #####################################
###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\trialOptions'
os.chdir(path)

# Define the number of trials
NUM_EVAL = 500

# File to save first results
out_file = 'Catboost_HPO_Upsampling_500.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0

# Begin HPO trials for Upsampling data
# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
best_param = fmin(catboost_hpo_us, catboost_tune_kwargs, algo=tpe.suggest,
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
print('Upsampling HPO 500 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_Upsampling_trials_results[:2])

# Access results
results = pd.read_csv('Catboost_HPO_Upsampling_500.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_Upsampling_model = CatBoostClassifier(loss_function='Logloss', 
                                                 eval_metric='AUC',
                                                 early_stopping_rounds=10,
                                                 logging_level='Silent', 
                                                 random_state=seed_value,
                                                 **best_bayes_params)

# Fit the model
best_bayes_Upsampling_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'Catboost_HPO_Upsampling_500.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_Upsampling_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_HPO_Upsampling_500.pkl')
# print(model)
# =============================================================================

print('\nModel Metrics for Catboost HPO Upsampling 500trials')
# Predict based on training 
y_pred_Upsampling_HPO = best_bayes_Upsampling_model.predict(X_test)

print('Results from Catboost HPO 500 on Upsampling Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_Upsampling_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_Upsampling_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_Upsampling_HPO))
print('Precision score : %.3f' % precision_score(y_test, y_pred_Upsampling_HPO))
print('Recall score : %.3f' % recall_score(y_test, y_pred_Upsampling_HPO))
print('F1 score : %.3f' % f1_score(y_test, y_pred_Upsampling_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_Upsampling_model.predict_proba(X_test)[:, 1]

print('The best model from Upsampling Bayes 500 trials optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, 
                                                                                                                                 preds)))
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
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_Catboost_HPO_Upsampling_500.csv', 
                    index=False)

# Convert data types for graphing
bayes_params['depth'] = bayes_params['depth'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['l2_leaf_reg'] = bayes_params['l2_leaf_reg'].astype('float64')
bayes_params['min_data_in_leaf'] = bayes_params['min_data_in_leaf'].astype('float64')
bayes_params['one_hot_max_size'] = bayes_params['one_hot_max_size'].astype('float64')

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
    if hpo not in ['iteration', 'scale_pos_weight', 'iterations']:
        plt.figure(figsize=(14,6))
        # Plot the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'min_data_in_leaf', 
                         'one_hot_max_size']): 
  # Scatterplot
  sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
  axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
             title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
ax = sns.regplot('iteration', 'l2_leaf_reg', data=bayes_params, 
                 label='Bayes Optimization') 
ax.set(xlabel='Iteration', ylabel='l2_leaf_reg')                 
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations'
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
with open(r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_Upsampling_500_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_Upsampling_500_WeightsFeatures.htm'
webbrowser.open(url, new=2)

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
######################### Catboost HPO for SMOTE Set ##########################
############################## 500 Trials #####################################
###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\trialOptions'
os.chdir(path)

# File to save first results
out_file = 'Catboost_HPO_SMOTE_500.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global  ITERATION
ITERATION = 0

# Begin HPO trials for Upsampling data
# Start timer for experiment
start_time = datetime.now()
print('%-20s %s' % ('Start Time', start_time))
best_param = fmin(catboost_hpo_smote, catboost_tune_kwargs, algo=tpe.suggest,
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
print('SMOTE HPO 500 trials: Top two trials with the lowest loss (highest AUC)')
print(bayesOpt_SMOTE_trials_results[:2])

# Access results
results = pd.read_csv('Catboost_HPO_SMOTE_500.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number hyperparameters
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_SMOTE_model = CatBoostClassifier(loss_function='Logloss', 
                                            eval_metric='AUC',
                                            early_stopping_rounds=10,
                                            logging_level='Silent', 
                                            random_state=seed_value,
                                            **best_bayes_params)

# Fit the model
best_bayes_SMOTE_model.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'Catboost_HPO_SMOTE_500.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_SMOTE_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('Catboost_HPO_SMOTE_500.pkl')
# print(model)
# =============================================================================
    
print('\nModel Metrics for Catboost HPO SMOTE 500trials')
# Predict based on training 
y_pred_SMOTE_HPO = best_bayes_SMOTE_model.predict(X1_test)

print('Results from Catboost HPO 500 trials on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y1_test, y_pred_SMOTE_HPO))
print('Precision score : %.3f' % precision_score(y1_test, y_pred_SMOTE_HPO))
print('Recall score : %.3f' % recall_score(y1_test,y_pred_SMOTE_HPO))
print('F1 score : %.3f' % f1_score(y1_test, y_pred_SMOTE_HPO))

# Evaluate predictive probability on the testing data 
preds = best_bayes_SMOTE_model.predict_proba(X1_test)[:, 1]

print('The best model from SMOTE Bayes optimization 300 trials scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y1_test, 
                                                                                                                            preds)))
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
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_Catboost_HPO_SMOTE_500.csv', index=False)

# Convert data types for graphing
bayes_params['depth'] = bayes_params['depth'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['l2_leaf_reg'] = bayes_params['l2_leaf_reg'].astype('float64')
bayes_params['min_data_in_leaf'] = bayes_params['min_data_in_leaf'].astype('float64')
bayes_params['one_hot_max_size'] = bayes_params['one_hot_max_size'].astype('float64')

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
    if hpo not in ['iteration', 'scale_pos_weight', 'iterations']:
        plt.figure(figsize=(14,6))
        # Plot the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label='Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 3, figsize=(20,5))
i = 0
for i, hpo in enumerate(['learning_rate', 'min_data_in_leaf', 
                         'one_hot_max_size']): 
  # Scatterplot
  sns.regplot('iteration', hpo, data=bayes_params, ax=axs[i])
  axs[i].set(xlabel='Iteration', ylabel='{}'.format(hpo), 
             title='{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 18
ax = sns.regplot('iteration', 'l2_leaf_reg', data=bayes_params, 
                 label='Bayes Optimization') 
ax.set(xlabel='Iteration', ylabel='l2_leaf_reg')                 
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations'
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
with open(r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_SMOTE_500_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Catboost\Hyperopt\Model_Explanations\best_bayes_SMOTE_500_WeightsFeatures.htm'
webbrowser.open(url, new=2)

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