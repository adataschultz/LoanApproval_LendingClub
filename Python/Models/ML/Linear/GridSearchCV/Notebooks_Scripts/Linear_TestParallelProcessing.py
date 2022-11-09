# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
############## Classification - Linear: Test Parallel Processing ##############
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
import time

# Set path
path = r'D:\LoanStatus\Data'
os.chdir(path)

# Set seed 
seed_value = 42
os.environ['LoanStatus_ParallelProcessing'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Read data
train_US = pd.read_csv('trainDF_US.csv', low_memory=False)

# Upsampling - Separate input features and target
X_train = train_US.drop('loan_status', axis=1)
y_train = train_US[['loan_status']]

del train_US

# Feature Scaling
sc = StandardScaler()
X_trainS = sc.fit_transform(X_train)

###############################################################################
# 500 Max Iterations
# Define model
LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=500, 
                                     random_state=seed_value)

# Threading
print('Time for joblib threading..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib threading:', time.time() - search_time_start)
print('\n')

# Loky
print('Time for joblib loky..')
search_time_start = time.time()
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib loky:', time.time() - search_time_start)
print('\n')

# Set up model for n_jobs=-1 without/with threading or loky
search_time_start = time.time()
LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=500, 
                                     random_state=seed_value, 
                                     n_jobs=-1)

print('Time for joblib n_job=-1..')
search_time_start = time.time()
LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblibs=-1:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with threading 
print('Time for joblib threading & n_jobs=-1..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib threading  with n=-1 jobs:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with loky
print('Time for joblib loky & n_jobs=-1..')
search_time_start = time.time()
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib loky with n=-1 jobs:', time.time() - search_time_start)

###############################################################################
# 1000 Max Iterations
# Define model
LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=1000, 
                                     random_state=seed_value)

# Threading
print('Time for joblib threading..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib threading:', time.time() - search_time_start)
print('\n')

# Loky
print('Time for joblib loky..')
search_time_start = time.time()
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib loky:', time.time() - search_time_start)
print('\n')

# Set up model for n_jobs=-1 without/with threading or loky
search_time_start = time.time()
LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=1000, 
                                     random_state=seed_value, 
                                     n_jobs=-1)

print('Time for joblib n_job=-1..')
search_time_start = time.time()
LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblibs=-1:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with threading 
print('Time for joblib threading & n_jobs=-1..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib threading  with n=-1 jobs:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with loky
print('Time for joblib loky & n_jobs=-1..')
search_time_start = time.time()
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib loky with n=-1 jobs:', time.time() - search_time_start)

###############################################################################
# 3000 Max Iterations
# Define model
LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=3000, 
                                     random_state=seed_value)

# Threading
print('Time for joblib threading..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib threading:', time.time() - search_time_start)
print('\n')

# Loky
print('Time for joblib loky..')
search_time_start = time.time()
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib loky:', time.time() - search_time_start)
print('\n')

# Set up model for n_jobs=-1 without/with threading or loky
search_time_start = time.time()
LassoMod_US_HPO = LogisticRegression(penalty='l1', 
                                     C=10, 
                                     solver='saga', 
                                     max_iter=3000, 
                                     random_state=seed_value, 
                                     n_jobs=-1)

print('Time for joblib n_job=-1..')
search_time_start = time.time()
LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblibs=-1:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with threading 
print('Time for joblib threading & n_jobs=-1..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib threading  with n=-1 jobs:', time.time() - search_time_start)
print('\n')

# n_jobs=-1 with loky
print('Time for joblib loky & n_jobs=-1..')
search_time_start = time.time()
with parallel_backend('loky', n_jobs=-1):
    LassoMod_US_HPO.fit(X_trainS, y_train)
print('Finished with joblib loky with n=-1 jobs:', time.time() - search_time_start)

###############################################################################