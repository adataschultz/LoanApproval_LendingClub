# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:40:28 2021

@author: aschu
"""
###############################################################################
############## Classification - Linear: Test Parallel Processing ##############
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import parallel_backend
import dask.delayed
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import time

# Set path
path = r'D:\Loan-Status\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_Linear_ParallelProcessing'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Read data
df = pd.read_csv('LendingTree_LoanStatus_final.csv', low_memory=False)
df = df.drop_duplicates()

###############################################################################
########################## Resampling Techniques ##############################
###############################################################################
########################   1. Oversample minority class #######################
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

# Separate input features and target of upsampled train data
X_train = upsampled.drop('loan_status', axis=1)
y_train = upsampled.loan_status

del upsampled

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###############################################################################
# Test time for training same model
LassoMod_US_HPO = LogisticRegression(penalty='l1', C=10, solver= 'saga',
                                                   max_iter=10000,
                                                   random_state=seed_value)
# Threading
print('Time for joblib threading..')
search_time_start = time.time()
# Fit the grid search results to the data
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_train, y_train)
print('Finished cleaning with joblib threading :', time.time() - search_time_start)
print('\n')

# Loky
print('Time for joblib loky..')
search_time_start = time.time()
# Fit the grid search restults to the data
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_train, y_train)
print('Finished cleaning with joblib loky :', time.time() - search_time_start)
print('\n')

# Set up for n_jobs=-1 without/with threading or loky
search_time_start = time.time()
LassoMod_US_HPO = LogisticRegression(penalty='l1', C=10, solver= 'saga',
                                                   max_iter=10000, n_jobs=-1,
                                                   random_state=seed_value)
print('Time for joblib n_job=-1..')
search_time_start = time.time()
# Fit the grid search results to the data
LassoMod_US_HPO.fit(X_train, y_train)
print('Finished cleaning with joblibs=-1:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with threading 
print('Time for joblib threading & n_jobs=-1..')
search_time_start = time.time()
# Fit the grid search results to the data
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_train, y_train)
print('Finished cleaning with joblib threading  with n=-1 jobs :', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with loky
print('Time for joblib loky & n_jobs=-1..')
search_time_start = time.time()
# Fit the grid search to the data
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_train, y_train)
print('Finished cleaning with joblib loky with n=-1 jobs :', time.time() - search_time_start)
###############################################################################