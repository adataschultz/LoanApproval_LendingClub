# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
########################## Lending Tree Loan Status ###########################
######################### Classification - Nonlinear ##########################
############################### Naive Bayes ###################################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
import joblib
from joblib import parallel_backend
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import time
import pickle
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
import eli5 
from eli5.sklearn import PermutationImportance 
import webbrowser
from eli5.formatters import format_as_dataframe
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

path = r'D:\LoanStatus\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_NB'] = str(seed_value)
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
################################ Baseline #####################################
###############################################################################
# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_PKL'
os.chdir(path)

# Set baseline model for Upsampling
nb = GaussianNB(random_state=seed_value)

# Fit the model
with parallel_backend('threading', n_jobs=-1):
    nb.fit(X_train, y_train)

# Save model
Pkl_Filename = 'NB_Upsampling_baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb, file)

# Predict based on training 
y_pred_US = nb.predict(X_test)

print('Results from Naives Bayes baseline model on Upsampled data:')
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
print('F1 score : %.3f' % f1_score(y_test, y_pred_US))

###############################################################################
# Set baseline model for SMOTE
# Fit the model
with parallel_backend('threading', n_jobs=-1):
    nb.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'NB_SMOTE_Baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb, file)

# Predict based on training 
y_pred_SMOTE = nb.predict(X_test)

print('Results from Naives Bayes baseline model on SMOTE data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_SMOTE))
print('Precision score : %.3f'% precision_score(y_test, y_pred_SMOTE))
print('Recall score : %.3f' % recall_score(y_test, y_pred_SMOTE))
print('F1 score : %.3f' % f1_score(y_test, y_pred_SMOTE))

###############################################################################
########################  Upsampling - Grid Search   ##########################
###############################################################################
# Define grid search parameters
param_grid = {
    'var_smoothing': np.logspace(0,-9, num=1000)
}

nb_grid_US = GridSearchCV(estimator=GaussianNB(), 
                          param_grid=param_grid, 
                          verbose=1, 
                          cv=3, 
                          n_jobs=-1)

print('Start Upsampling - Grid Search..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    nb_grid_US.fit(X_train, y_train)
print('Finished Upsampling - Grid Search :', time.time() - search_time_start)
print('======================================================================')
print('Naive Bayes: Upsampling')
print('- Best Score', nb_grid_US.best_score_)
print('- Best Estimator', nb_grid_US.best_estimator_)

# Fit best model from grid search on Upsampling data
nb_US_HPO = nb_grid_US.best_estimator_

print('Start fit the best hyperparameters from Upsampling grid search to the data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    nb_US_HPO.fit(X_train, y_train)
print('Finished fit the best hyperparameters from Upsampling grid search to the data :',
      time.time() - search_time_start)
print('======================================================================')
    
# Save model
Pkl_Filename = 'NB_HPO_Upsampling.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_US_HPO, file)

# =============================================================================
# # To load saved model
# nb_US_HPO = joblib.load('NB_HPO_Upsampling.pkl')
# print(nb_US_HPO)
# =============================================================================
    
# Predict based on training 
y_pred_US = nb_US_HPO.predict(X_test)

print('Results from Naives Bayes using Best HPO from GridSearchCV on Upsampled data:')
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
print('F1 score : %.3f' % f1_score(y_test, y_pred_US))

###############################################################################
# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(nb_US_HPO,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                                 feature_names=X_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_US_HPO_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights html file
url = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_US_HPO_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('NB_US_HPO_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X_test.iloc[1], 
    predict_fn=nb_US_HPO.predict_proba)

exp.save_to_file('NB_US_HPO_LIME.html')

###############################################################################
# Use best model from grid search to compare with SMOTE
print('Start Fit best model using gridsearch results on Upsamplimg to SMOTE data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    nb_US_HPO.fit(X1_train, y1_train)
print('Finished Fit best model using gridsearch results on Upsamplimg to SMOTE data :',
      time.time() - search_time_start)
print('======================================================================')

# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'NB_SMOTEusingUpsamplingHPO.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_US_HPO, file)
    
# Predict based on training 
y_pred_SMOTE_US = nb_US_HPO.predict(X1_test)

print('Results from Naives Bayes using Upsampling Best HPO from GridSearchCV on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE_US)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE_US))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_SMOTE))
print('Precision score : %.3f'% precision_score(y_test, y_pred_SMOTE))
print('Recall score : %.3f' % recall_score(y_test, y_pred_SMOTE))
print('F1 score : %.3f' % f1_score(y_test, y_pred_SMOTE))

###############################################################################
# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(nb_US_HPO,
                                        random_state=seed_value).fit(X1_test,
                                                                     y1_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                                 feature_names=X1_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_US_HPO_SMOTE_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights html file
url = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_US_HPO_SMOTE_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('NB_US_HPO_SMOTE_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test.iloc[1], 
    predict_fn=nb_US_HPO.predict_proba)
exp.save_to_file('NB_US_HPO_SMOTE_LIME.html')

###############################################################################
##########################  SMOTE - Grid Search  ##############################
###############################################################################
nb_grid_SMOTE = GridSearchCV(estimator=GaussianNB(), 
                             param_grid=param_grid,
                             verbose=1, 
                             cv=3, 
                             n_jobs=-1)

print('Start SMOTE - Grid Search..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    nb_grid_SMOTE.fit(X1_train, y1_train)
print('Finished SMOTE - Grid Search :', time.time() - search_time_start)
print('======================================================================')
print('Naive Bayes: SMOTE')    
print('- Best Score', nb_grid_SMOTE.best_score_)
print('- Best Estimator', nb_grid_SMOTE.best_estimator_)

# Fit best model from grid search on Upsampling data
nb_SMOTE_HPO = nb_grid_SMOTE.best_estimator_

print('Start fit the best hyperparameters from SMOTE grid search to the data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    nb_SMOTE_HPO.fit(X1_train, y1_train)
print('Finished fit the best hyperparameters from SMOTE grid search to the data:',
      time.time() - search_time_start)
print('======================================================================')

# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'NB_HPO_SMOTE.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_SMOTE_HPO, file)
    
# Predict based on training 
y_pred_SMOTE = nb_SMOTE_HPO.predict(X1_test)

print('Results from Naives Bayes using Best HPO from GridSearchCV on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_SMOTE))
print('Precision score : %.3f'% precision_score(y_test, y_pred_SMOTE))
print('Recall score : %.3f' % recall_score(y_test, y_pred_SMOTE))
print('F1 score : %.3f' % f1_score(y_test, y_pred_SMOTE))

###############################################################################
# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(nb_SMOTE_HPO,
                                        random_state=seed_value).fit(X1_test,
                                                                     y1_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                                 feature_names=X1_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_SMOTE_HPO_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights html file
url = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_SMOTE_HPO_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('NB_SMOTE_HPO_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test.iloc[1], 
    predict_fn=nb_SMOTE_HPO.predict_proba)

exp.save_to_file('NB_SMOTE_HPO_LIME.html')

###############################################################################
# Use best model from grid search to compare with Upsampling
print('Start fit best model using gridsearch results on SMOTE to Upsamplimg data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    nb_SMOTE_HPO.fit(X_train, y_train)
print('Finished fit best model using gridsearch results on SMOTE to Upsamplimg data :',
      time.time() - search_time_start)
print('======================================================================')

# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'NB_UpsamplingUsingSMOTEHPO.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_SMOTE_HPO, file)
    
# Predict based on training 
y_pred_US_SMOTE = nb_SMOTE_HPO.predict(X_test)

print('Results from Naives Bayes using SMOTE Best HPO from GridSearchCV on Upsampling Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_US_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_US_SMOTE))
print('\n')
print('Accuracy score : %.3f'% accuracy_score(y_test, y_pred_US_SMOTE))
print('Precision score : %.3f' % precision_score(y_test, y_pred_US_SMOTE))
print('Recall score : %.3f' % recall_score(y_test, y_pred_US_SMOTE))
print('F1 score : %.3f' % f1_score(y_test, y_pred_US_SMOTE))

###############################################################################
# Set path for Models results
path = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(nb_SMOTE_HPO,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                                 feature_names=X_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_SMOTE_HPO_US_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights html file
url = r'D:\LoanStatus\Python\Models\ML\NB\GridSearchCV\Model_Explanations\NB_SMOTE_HPO_US_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('NB_SMOTE_HPO_US_WeightsExplain.csv', index=False)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test.iloc[1], 
    predict_fn=nb_US_HPO.predict_proba)

exp.save_to_file('NB_SMOTE_HPO_US_LIME.html')

###############################################################################