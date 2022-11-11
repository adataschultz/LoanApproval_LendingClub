# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
########################## Lending Tree Loan Status ###########################
########################## Classification - Linear ############################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from joblib import parallel_backend
import itertools
import dask.delayed
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
pd.set_option('display.max_columns', None)

path = r'D:\LoanStatus\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_Linear'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Read data
df = pd.read_csv('LendingTree_LoanStatus_final.csv', low_memory=False)
df = df.drop_duplicates()

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
# Set up training and testing sets for oversampling minor class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed_value)

# Concatenate training data back together
df1 = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes of loan status
current = df1[df1.loan_status==0]
default = df1[df1.loan_status==1]

del df1

# Upsample minority
default_upsampled = resample(default,
                             replace=True, # Sample with replacement
                             n_samples=len(current), # Match number in majority 
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

cols = ['loan_status']
y_train = pd.DataFrame(data=y_train, columns=cols)
y_test = pd.DataFrame(data=y_test, columns=cols)

train_US = pd.concat([X_train, y_train], axis=1)
train_US.to_csv('trainDF_US.csv', index=False)

test_US = pd.concat([X_test, y_test], axis=1)
test_US.to_csv('testDF_US.csv', index=False)

del train_US, test_US

###############################################################################
######################## 2. Split over upsampling with SMOTE  #################
###############################################################################
# Set up training and testing sets for upsampling with SMOTE
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=seed_value)

smote = SMOTE(sampling_strategy='minority', random_state=seed_value)
X1_train, y1_train = smote.fit_sample(X1_train, y1_train)

print('\nExamine Loan Status after upsampling with SMOTE') 
print(y1_train.value_counts())
print('======================================================================')

y1_train = pd.DataFrame(data=y1_train, columns=cols)
y1_test = pd.DataFrame(data=y1_test, columns=cols)

train_SMOTE = pd.concat([X1_train, y1_train], axis=1)
train_SMOTE.to_csv('trainDF_SMOTE.csv', index=False)

test_SMOTE = pd.concat([X1_test, y1_test], axis=1)
test_SMOTE.to_csv('testDF_SMOTE.csv', index=False)

del train_SMOTE, test_SMOTE

###############################################################################
# Baseline Lasso for Upsampling
# Scale the data
scaler = MinMaxScaler()
X_trainS = scaler.fit_transform(X_train)
X_testS = scaler.transform(X_test)

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_PKL'
os.chdir(path)

# Set baseline model for Upsampling
lasso = LogisticRegression(penalty='l1', solver='saga', 
                           random_state=seed_value)

# Fit the model
with parallel_backend('threading', n_jobs=-1):
    lasso.fit(X_trainS, y_train)

# Save model
Pkl_Filename = 'Lasso_US_baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(lasso, file)
    
# Predict based on training 
y_pred_US = lasso.predict(X_testS)

print('Results from Lasso baseline model on Upsampled data:')
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
# Baseline Elastic Net for Upsampling
elnet = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)

# Fit the model
with parallel_backend('threading', n_jobs=-1): 
    elnet.fit(X_trainS, y_train)

# Save model
Pkl_Filename = 'Elnet_US_baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(elnet, file)

# Predict based on training 
y_pred_US = elnet.predict(X_testS)

print('Results from Elastic Net baseline model on Upsampled data:')
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
# Baseline Lasso for SMOTE
# Scale the data
scaler = MinMaxScaler()
X1_trainS = scaler.fit_transform(X1_train)
X1_testS = scaler.transform(X1_test)

with parallel_backend('threading', n_jobs=-1):
    lasso.fit(X1_trainS, y1_train)

Pkl_Filename = 'Lasso_SMOTE_baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(lasso, file)
    
y_pred_SMOTE = lasso.predict(X1_testS)

print('Results from Lasso baseline model on SMOTE data:')
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
# Baseline Elastic Net for SMOTE
# Fit the model
with parallel_backend('threading', n_jobs=-1):
    elnet.fit(X1_trainS, y1_train)

Pkl_Filename = 'Elnet_SMOTE_baseline.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(elnet, file)
    
y_pred_SMOTE = elnet.predict(X1_testS)

print('Results from Elastic Net baseline model on SMOTE data:')
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
# Design pipeline
numeric_features = list(X.columns[X.dtypes != 'object'])
numeric_transformer = Pipeline(steps=[ 
    ('mms', MinMaxScaler())])

# Full prediction pipeline
pipe_lasso = Pipeline(steps=[('preprocessor', numeric_transformer), 
                             ('model', LogisticRegression(penalty='l1', 
                                                          solver='saga',
                                                          max_iter=100,
                                                          random_state=seed_value))])

pipe_elnet = Pipeline(steps=[('preprocessor', numeric_transformer), 
                             ('model', LogisticRegression(penalty='elasticnet',
                                                          solver='saga', 
                                                          l1_ratio=0.5,
                                                          max_iter=100, 
                                                          random_state=seed_value))])

pipelines = [pipe_lasso, pipe_elnet]

pipe_dict = {0: 'Lasso', 1: 'Elastic Net'}

###############################################################################
# Define function for paramater grid for GridSearchCV using multiple algorithms
# Different keys will be evaluated sequentially in declared pipeline
def make_param_grids(steps, param_grids):  
    final_params = []
    for estimator_names in itertools.product(*steps.values()):
        current_grid = {}
        for step_name, estimator_name in zip(steps.keys(), estimator_names):
            for param, value in param_grids.get(estimator_name).items():
                if param == 'object':
                    current_grid[step_name] = [value]
                else:
                    current_grid[step_name+'__'+param]=value
        final_params.append(current_grid)

        return final_params

pipeline_steps = {'classifier': ['lasso', 'elnet']}

all_param_grids = {'lasso': {'object': LogisticRegression(), 
                             'penalty': ['l1'], 
                             'solver': ['saga'], 
                             'max_iter': [100, 500, 1000],
                             'C': [10, 1, 0.1, 0.05, 0.01, 0.001]
                            }, 

                   'elnet': {'object': LogisticRegression(), 
                             'penalty': ['elasticnet'], 
                             'solver': ['saga'], 
                             'l1_ratio': [0.5],
                             'max_iter': [100, 500, 1000],
                             'C': [10, 1, 0.1, 0.05, 0.01, 0.001]
                            }
                   }  

param_grids_list = make_param_grids(pipeline_steps, all_param_grids)

pipe = Pipeline(steps=[('classifier', LogisticRegression())])  

# Define gridSearchCV 
grid = GridSearchCV(pipe, param_grid = param_grids_list, cv=3, n_jobs=-1)

###############################################################################
################## Upsampling - Grid Search using Pipelines   #################
###############################################################################
# Fit data using Dask.delayed
print('Start Upsampling - Grid Search..')
search_time_start = time.time()
pipelines_ = [dask.delayed(grid).fit(X_train, y_train.ravel())]
fit_pipelines = dask.compute(*pipelines_, scheduler='processes')

print('Finished Upsampling - Grid Search :', time.time() - search_time_start)
print('======================================================================')

# Compare accuracy to find model with highest accuracy
for idx, val in enumerate(fit_pipelines):
	print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], 
                                            val.score(X_test, y_test)))

# Upsampling - Lasso pipeline test accuracy: 0.984
# Identify the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(fit_pipelines): 
    if val.score(X_test, y_test) > best_acc: 
        best_acc = val.score(X_test, y_test)
        best_pipe = val
        best_clf = idx

print('Classification with highest accuracy: %s' % pipe_dict[best_clf])

# Find best parameters from GridsearchCV in Upsampled data 
print('Lasso/Elastic Net using Upsampling - Best Estimator')
print(best_pipe.best_params_)

LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=1000, 
                                     n_jobs=-1, 
                                     random_state=seed_value)

print('Start fit the best hyperparameters from Upsampling grid search to the data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished fit the best hyperparameters from Upsampling grid search to the data :',
      time.time() - search_time_start)
print('======================================================================')

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_PKL'
os.chdir(path)

# Save model
Pkl_Filename = 'Linear_HPO_Upsampling.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(LassoMod_US_HPO, file)

# =============================================================================
# # To load saved model
# LassoMod_US_HPO = joblib.load('Linear_HPO_Upsampling.pkl')
# print(LassoMod_US_HPO)
# =============================================================================

# Predict based on training 
y_pred_US_HPO = LassoMod_US_HPO.predict(X_testS)

print('Results from LASSO using Upsampling HPO:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y_test, y_pred_US_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred_US_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y_test, y_pred_US_HPO))
print('Precision score : %.3f' % precision_score(y_test, y_pred_US_HPO))
print('Recall score : %.3f' % recall_score(y_test, y_pred_US_HPO))
print('F1 score : %.3f' % f1_score(y_test, y_pred_US_HPO))

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations'
os.chdir(path)

X_train1 = pd.DataFrame(X_train, columns=X_train.columns)
X_test1 = pd.DataFrame(X_test, columns=X_test.columns)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(LassoMod_US_HPO,
                                        random_state=seed_value).fit(X_testS, 
                                                                     y_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X_test1.columns.tolist())

# Write feature weights html object to a file 
with open('D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X_test.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('LassoMod_US_HPO_WeightsExplain.csv', index=False)

# Show prediction
html_obj2 = eli5.show_prediction(LassoMod_US_HPO, X_test1.iloc[1],
                                 show_feature_values=True)

# Write show prediction html object to a file 
with open('D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the show prediction stored HTML file
url2 = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_Prediction.htm'
webbrowser.open(url2, new=2)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X_test1.iloc[1], 
    predict_fn=LassoMod_US_HPO.predict_proba)
exp.save_to_file('LassoMod_US_HPO_LIME.html')

###############################################################################
print('Start Fit best model using gridsearch results on Upsamplimg to SMOTE data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X1_trainS, y1_train)
print('Finished Fit best model using gridsearch results on Upsamplimg to SMOTE data :',
      time.time() - search_time_start)
print('======================================================================')

# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_PKL'
os.chdir(path)

# Save model
Pkl_Filename = 'LassoMod_SMOTEusingUpsamplingHPO.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(LassoMod_US_HPO, file)

# =============================================================================
# # To load saved model
# model = joblib.load('LassoMod_SMOTEusingUpsamplingHPO.pkl')
# print(model)
# =============================================================================

# Predict based on training 
y_pred_US_HPO = LassoMod_US_HPO.predict(X1_test)

print('Results from LASSO using Upsampling HPO on SMOTE Data:')
print('\n')
print('Classification Report:')
clf_rpt = classification_report(y1_test, y_pred_US_HPO)
print(clf_rpt)
print('\n')
print('Confusion matrix:')
print(confusion_matrix(y1_test, y_pred_US_HPO))
print('\n')
print('Accuracy score : %.3f' % accuracy_score(y1_test, y_pred_US_HPO))
print('Precision score : %.3f' % precision_score(y1_test, y_pred_US_HPO))
print('Recall score : %.3f' % recall_score(y1_test, y_pred_US_HPO))
print('F1 score : %.3f' % f1_score(y1_test, y_pred_US_HPO))

###############################################################################
# Set path for ML results
path = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations'
os.chdir(path)

X1_train1 = pd.DataFrame(X1_train, columns=X1_train.columns)
X1_test1 = pd.DataFrame(X1_test, columns=X1_train.columns)

# Model metrics with Eli5
# Compute permutation feature importance (if loaded from saved pickle)
perm_importance = PermutationImportance(LassoMod_US_HPO,
                                        random_state=seed_value).fit(X1_testS,
                                                                     y1_test)

# Store feature weights in an object
html_obj = eli5.show_weights(perm_importance,
                             feature_names=X1_test1.columns.tolist())

# Write feature weights html object to a file 
with open('D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_SMOTE_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode('UTF-8'))

# Open the stored feature weights HTML file
url = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_SMOTE_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli5.explain_weights_sklearn(perm_importance,
                                           feature_names=X1_test1.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('LassoMod_US_HPO_SMOTE_WeightsExplain.csv', index=False)

# Show prediction
html_obj2 = eli5.show_prediction(LassoMod_US_HPO, X1_test1.iloc[1],
                                 show_feature_values=True)

# Write show prediction html object to a file 
with open('D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_SMOTE_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode('UTF-8'))

# Open the stored show prediction HTML file
url2 = r'D:\LoanStatus\Python\Models\ML\Linear\GridSearchCV\Model_Explanations\LassoMod_US_HPO_SMOTE_Prediction.htm'
webbrowser.open(url2, new=2)

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1_train),
    feature_names=X1_train1.columns,
    class_names=['current', 'default'],
    mode='classification')

exp = explainer.explain_instance(
    data_row=X1_test1.iloc[1], 
    predict_fn=LassoMod_US_HPO.predict_proba)
exp.save_to_file('LassoMod_US_HPO_LIME_SMOTE.html')

###############################################################################
##################### SMOTE - Grid Search using Pipelines   ###################
###############################################################################
# Fit data using Dask.delayed
print('Start SMOTE - Grid Search..')
search_time_start = time.time()
pipelines1_ = [dask.delayed(grid).fit(X1_train, y1_train)]
fit_pipelines1 = dask.compute(*pipelines1_, scheduler='processes')

print('Finished SMOTE - Grid Search :', time.time() - search_time_start)
print('======================================================================')

# Compare accuracy to find model with highest accuracy
for idx, val in enumerate(fit_pipelines1): 
    print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X1_test, 
                                                                         y1_test)))

# Identify the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(fit_pipelines1):
	if val.score(X1_test, y1_test) > best_acc:
		best_acc = val.score(X1_test, y1_test)
		best_pipe = val
		best_clf = idx
print('Classification with highest accuracy: %s' % pipe_dict[best_clf])

# Find best parameters from GridsearchCV in Upsampled data 
print('Lasso/Elastic Net using SMOTE - Best Estimator')
print(best_pipe.best_params_)

###############################################################################