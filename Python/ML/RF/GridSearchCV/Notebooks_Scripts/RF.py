# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
########################## Lending Tree Loan Status ###########################
######################### Classification - Nonlinear ##########################
############################ Random Forest ####################################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import eli5 as eli
from eli5.sklearn import PermutationImportance 
from eli5 import show_weights
import webbrowser
from eli5.sklearn import explain_weights_sklearn
from eli5.formatters import format_as_dataframe, format_as_dataframes
from eli5 import show_prediction
import lime
from lime import lime_tabular

path = r'D:\Loan-Status\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_NonLinear_RF'] = str(seed_value)
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

###############################################################################
############################## Random Forest ##################################
###############################################################################
# Set baseline model for Upsampling
rf_US = RandomForestClassifier(random_state=seed_value, n_jobs=-1)
print('Baseline parameters for RF', rf_US.get_params()) 

# Fit the grid search to the data
with parallel_backend('threading', n_jobs=-1):
    rf_US.fit(X_train, y_train)

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'RF_Upsampling_Baseline.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_US, file)

# =============================================================================
# # To load saved model
# model = joblib.load('RF_Upsampling_Baseline.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_US = rf_US.predict(X_test)

print('Results from Random Forest using Baseline on Upsampled Data:')
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
rf_SMOTE = RandomForestClassifier(random_state=seed_value, n_jobs=-1)

# Fit the grid search to the data
with parallel_backend('threading', n_jobs=-1):
    rf_SMOTE.fit(X1_train, y1_train)
    
# Save model
Pkl_Filename = 'RF_SMOTE_Baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_SMOTE, file)

# Predict based on training 
y_pred_SMOTE = rf_SMOTE.predict(X1_test)

print('Results from Random Forest using Baseline on SMOTE Data:')
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
# Define param grid 
param_grid = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [100, 200, 300, 500],
    'bootstrap': [True],
    'max_depth': [20, 40, 60, 80, 100, None],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

# Create a grid search based model
grid_search_US = GridSearchCV(estimator = rf_US, param_grid = param_grid, 
                           verbose = 1, cv = 3,  n_jobs = -1)

# Fit the grid search to the data
print('Start Upsampling - Grid Search..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    grid_search_US.fit(X_train, y_train)
print('Finished Upsampling - Grid Search :', time.time() - search_time_start)
print('======================================================================')

print('Upsampling RF HPO - best parameters')    
print(grid_search_US.best_params_)

# Use best model from grid search to see feature importance
rf_US_HPO = grid_search_US.best_estimator_

# Fit the results from grid search to the data
print('Start fit the best hyperparameters from Upsampling grid search to the data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-2):
    rf_US_HPO.fit(X_train, y_train)
print('Finished fit the best hyperparameters from Upsampling grid search to the data :',
      time.time() - search_time_start)
print('======================================================================')
    
# Save model
Pkl_Filename = 'RF_HPO_Upsampling.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_US_HPO, file)

# =============================================================================
# # To load saved model
#rf_US_HPO = joblib.load('RF_HPO_Upsampling.pkl')
#print(rf_US_HPO)
# =============================================================================
    
# Predict based on training 
y_pred_US = rf_US_HPO.predict(X_test)

print('Results from Random Forest using GridSearchCV on Upsampled Data:')
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

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_Explanations'
os.chdir(path)

# Print the name and entropy importance of each feature
df_rf = []
for feature in zip(X, rf_US_HPO.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)
df_rf.to_csv('Upsampling_rf_gridsearchBest_featureimportance.csv', index=False, 
             encoding='utf-8-sig')

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(rf_US_HPO,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(rf_US_HPO, X_test.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(rf_US_HPO, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X_test.iloc[1], 
    predict_fn=rf_US_HPO.predict_proba)

exp.save_to_file('RF_US_HPO_LIME.html')

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\WeightsExplain'
os.chdir(path)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('RF_US_HPO_WeightsExplain.csv', index=False, encoding='utf-8-sig')

###############################################################################
# Use best model from grid search to compare with SMOTE
print('Start Fit best model using gridsearch results on Upsamplimg to SMOTE data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    rf_US_HPO.fit(X1_train, y1_train)
print('Finished Fit best model using gridsearch results on Upsamplimg to SMOTE data :',
      time.time() - search_time_start)
print('======================================================================')

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'RF_SMOTEusingUpsamplingHPO.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_US_HPO, file)

# Predict based on training 
y_pred_SMOTE_US = rf_US_HPO.predict(X1_test)

print('Results from Random Forest using Upsampling Best from GridSearch on SMOTE Data:')
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

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_Explanations'
os.chdir(path)

# Print the name and entropy importance of each feature
df_rf = []
for feature in zip(X, rf_SMOTE.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)
df_rf.to_csv('SMOTEusingUpsampling_rf_gridsearchBest_featureimportance.csv',
             index=False, encoding='utf-8-sig')

###############################################################################
# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(rf_US_HPO,
                                        random_state=seed_value).fit(X1_test,
                                                                     y1_test)

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X1_test.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_SMOTE_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_SMOTE_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(rf_US_HPO, X1_test.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_SMOTE_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\Loan-Status\Python\ML\RF\Model_Explanations\RF_US_HPO_SMOTE_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(rf_US_HPO, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test.iloc[1], 
    predict_fn=rf_US_HPO.predict_proba)

exp.save_to_file('RF_US_HPO_SMOTE_LIME.html')

###############################################################################
# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\WeightsExplain'
os.chdir(path)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X1_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('RF_US_HPO_SMOTE_WeightsExplain.csv', index=False, 
           encoding='utf-8-sig')

###############################################################################
##########################  SMOTE - Grid Search  ##############################
###############################################################################
# Create a grid search based model
grid_search_SMOTE = GridSearchCV(estimator = rf_SMOTE, param_grid = param_grid, 
                           verbose = 1, cv = 3,  n_jobs = -1)

# Fit the grid search to the data
print('Start SMOTE - Grid Search..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    grid_search_SMOTE.fit(X1_train, y1_train)
print('Finished SMOTE - Grid Search :', time.time() - search_time_start)
print('======================================================================')

print('SMOTE RF HPO - best parameters')    
print(grid_search_SMOTE.best_params_)

# SMOTEe best model from grid search to see feature importance
rf_SMOTE_HPO = grid_search_SMOTE.best_estimator_

# Fit the results from grid search to the data
print('Start fit the best hyperparameters from SMOTE grid search to the data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    rf_SMOTE_HPO.fit(X1_train, y1_train)
print('Finished fit the best hyperparameters from SMOTE grid search to the data:',
      time.time() - search_time_start)
print('======================================================================')
    
# Save model
Pkl_Filename = 'RF_HPO_SMOTE.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_SMOTE_HPO, file)
    
# Predict based on training 
y_pred_SMOTE = rf_SMOTE_HPO.predict(X1_test)

print('Results from Random Forest using GridSearchCV on SMOTE Data:')
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

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_Explanations'
os.chdir(path)

# Print the name and entropy importance of each feature
df_rf = []
for feature in zip(X, rf_SMOTE_HPO.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)
df_rf.to_csv('SMOTE_rf_gridsearchBest_featureimportance.csv', index=False, 
             encoding='utf-8-sig')

###############################################################################
# Use best model from grid search to compare with US
print('Start fit best model using gridsearch results on SMOTE to Upsamplimg data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    rf_SMOTE_HPO.fit(X_train, y_train)
print('Finished fit best model using gridsearch results on SMOTE to Upsamplimg data :',
      time.time() - search_time_start)
print('======================================================================')

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_PKL'
os.chdir(path)
    
# Save model
Pkl_Filename = 'RF_UpsamplingUsingSMOTEHPO.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_SMOTE_HPO, file)

# Predict based on training 
y_pred_US_SMOTE = rf_SMOTE_HPO.predict(X_test)

print('Results from Random Forest using SMOTE Best from GridSearch on Upsampled Data:')
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

# Set path for ML results
path = r'D:\Loan-Status\Python\ML\RF\Model_Explanations'
os.chdir(path)

# Print the name and entropy importance of each feature
df_rf = []
for feature in zip(X, rf_US.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)
df_rf.to_csv('UpsamplingUsingSMOTE_rf_gridsearchBest_featureimportance.csv',
             index=False, encoding='utf-8-sig')

###############################################################################