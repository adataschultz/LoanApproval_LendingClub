# -*- coding: utf-8 -*-
"""
@author: aschu
"""

###############################################################################
##################### Classification - Nonlinear ##############################
############################ Naive Bayes ######################################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

path = r'D:\Loan-Status\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_NonLinear_NB'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Read data
df = pd.read_csv('LendingTree_LoanStatus_final.csv', low_memory=False)
df = df.drop_duplicates()
print(df.shape)

###############################################################################
############################# Resampling Techniques ###########################
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
                          random_state=42) 

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

# Change path to Results from Machine Learning
path = r'D:\Loan-Status\Python\ML_Results\NB'
os.chdir(path)

###############################################################################
################################ Naive Bayes ##################################
###############################################################################
# Set baseline model for Upsampling
nb_US = = GaussianNB(random_state=seed_value)

# Fit the model
with parallel_backend('threading', n_jobs=-1):
    nb_US.fit(X_train, y_train)

# Save model
Pkl_Filename = "LoanStatus_NB_Upsampling_gridSearch.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_US, file)

# Predict based on training 
y_pred_US = nb_US.predict(X_test)

print('Results from Naives Bayes baseline model on Upsampled Data:')
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
nb_SMOTE = = GaussianNB(random_state=seed_value)

# Fit the model
with parallel_backend('threading', n_jobs=-1):
    nb_SMOTE.fit(X1_train, y1_train)

# Save model
Pkl_Filename = "LoanStatus_NB_SMOTE_Baseline.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_SMOTE, file)

# Predict based on training 
y_pred_SMOTE = nb_SMOTE.predict(X_test)

print('Results from Naives Bayes baseline model on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y1_test, y_pred_SMOTE))
print('Precision score : %.3f'%precision_score(y1_test, y_pred_SMOTE))
print('Recall score : %.3f'%recall_score(y1_test, y_pred_SMOTE))
print('F1 score : %.3f'%f1_score(y1_test, y_pred_SMOTE))

###############################################################################
########################  Upsampling - Grid Search   ##########################
###############################################################################
# Define grid search parameters
param_grid = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

nb_grid_US = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid,
                            verbose=1, cv=10, n_jobs=-1)

with parallel_backend('threading', n_jobs=-1):
    nb_grid_US.fit(X_train, y_train)

# Save model
Pkl_Filename = 'LoanStatus_NB_Upsampling_gridSearch.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_grid_US, file)
    
print('Naive Bayes using Upsampling - Best Score')
print(nb_grid_US.best_score_)
print('\n')
print('Naive Bayes using Upsampling - Best Estimator')
print(nb_grid_US.best_estimator_)

# Fit best model from grid search on Upsampling data
nb_US_best = GaussianNB(**nb_grid_US.best_params)

with parallel_backend('threading', n_jobs=-1):
    nb_US_best.fit(X_train, y_train)
    
# Save model
Pkl_Filename = 'LoanStatus_NB_Upsampling_gridSearch_Best.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_US_best, file)
    
# Predict based on training 
y_pred_US = nb_US_best.predict(X_test)

print('Results from Naives Bayes using Best HPO from GridSearchCV on Upsampled Data:')
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
# Use best model from grid search to compare with SMOTE
nb_SMOTE = GaussianNB(**nb_grid_US.best_params)

with parallel_backend('threading', n_jobs=-1):
    nb_SMOTE.fit(X1_train, y1_train)
    
# Save model
Pkl_Filename = 'LoanStatus_NB_SMOTEusingUpsampling_Best.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_SMOTE, file)
    
# Predict based on training 
y_pred_SMOTE_US = nb_SMOTE.predict(X1_test)

print('Results from Naives Bayes using Upsampling Best HPO from GridSearchCV on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE_US)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE_US))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y1_test, y_pred_SMOTE_US))
print('Precision score : %.3f'%precision_score(y1_test, y_pred_SMOTE_US))
print('Recall score : %.3f'%recall_score(y1_test, y_pred_SMOTE_US))
print('F1 score : %.3f'%f1_score(y1_test, y_pred_SMOTE_US))

###############################################################################
##########################  SMOTE - Grid Search  ##############################
###############################################################################
nb_grid_SMOTE = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid,
                            verbose=1, cv=10, n_jobs=-1)

with parallel_backend('threading', n_jobs=-1):
    nb_grid_SMOTE.fit(X1_train, y1_train)

# Save model
Pkl_Filename = 'LoanStatus_NB_SMOTE_gridSearch.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_grid_SMOTE, file)
    
print('Naive Bayes using SMOTE - Best Score')
print(nb_grid_SMOTE.best_score_)
print('\n')
print('Naive Bayes using SMOTE - Best Estimator')
print(nb_grid_SMOTE.best_estimator_)

# Fit best model from grid search on Upsampling data
nb_SMOTE_best = GaussianNB(**nb_grid_SMOTE.best_params)

with parallel_backend('threading', n_jobs=-1):
    nb_SMOTE_best.fit(X1_train, y1_train)
    
# Save model
Pkl_Filename = 'LoanStatus_NB_SMOTE_gridSearch_Best.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_SMOTE_best, file)
    
# Predict based on training 
y_pred_SMOTE = nb_SMOTE_best.predict(X1_test)

print('Results from Naives Bayes using Best HPO from GridSearchCV on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_SMOTE))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y1_test, y_pred_SMOTE))
print('Precision score : %.3f'%precision_score(y1_test, y_pred_SMOTE))
print('Recall score : %.3f'%recall_score(y1_test, y_pred_SMOTE))
print('F1 score : %.3f'%f1_score(y_test, y1_pred_SMOTE))

###############################################################################
# Use best model from grid search to compare with Upsampling
nb_US = GaussianNB(**nb_grid_SMOTE.best_params)

with parallel_backend('threading', n_jobs=-1):
    nb_US.fit(X_train, y_train)
    
# Save model
Pkl_Filename = 'LoanStatus_nb_UpsamplingUsingSMOTE_Best.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb_US, file)
    
# Predict based on training 
y_pred_US_SMOTE = nb_US.predict(X_test)

print('Results from Naives Bayes using SMOTE Best HPO from GridSearchCV on Upsampling Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_US_SMOTE)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_US_SMOTE))
print('\n')
print('Accuracy score : %.3f'%accuracy_score(y_test, y_pred_US_SMOTE))
print('Precision score : %.3f'%precision_score(y_test, y_pred_US_SMOTE))
print('Recall score : %.3f'%recall_score(y_test, y_pred_US_SMOTE))
print('F1 score : %.3f'%f1_score(y_test, y_pred_US_SMOTE))

###############################################################################