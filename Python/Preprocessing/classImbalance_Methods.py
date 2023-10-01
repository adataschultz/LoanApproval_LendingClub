# -*- coding: utf-8 -*-
"""
@author: aschu
"""
print('\nLoan Status: Methods for Class Imbalance') 
print('======================================================================')

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

seed_value = 42
os.environ['LoanStatus_PreprocessEDA'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

path = r'D:\LoanStatus\Data'
os.chdir(path)

# Read file
df = pd.read_csv('LendingTree_LoanStatus_final.csv', index_col=False,
                 low_memory=False)

print('\nDimensions of Data:', df.shape) 
print('======================================================================')

###############################################################################
########################## Resampling Techniques ##############################
###############################################################################
# Separate input features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

###############################################################################
########################   1. Oversample minority class #######################
###############################################################################
# Setting up testing and training sets for oversampling minor class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=42)

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

# Create train/test sets for later use
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
# Setting up testing and training sets for upsampling with SMOTE
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=seed_value)

smote = SMOTE(sampling_strategy='minority', random_state=42)
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