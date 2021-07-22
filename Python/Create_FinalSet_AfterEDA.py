# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
######################## Create final data set ################################
###############################################################################
import os
import pandas as pd
path = r'D:\Loan-Status\Data'
os.chdir(path)

# Read data
df = pd.read_csv('LendingTree_LoanStatus_EDA.csv', low_memory=False)
df = df.drop_duplicates()

# Drop based off high correlations
df = df.drop(['funded_amnt', 'funded_amnt_inv', 'total_pymnt_inv',
               'collection_recovery_fee', 'avg_cur_bal', 'out_prncp_inv'],
             axis = 1)
df = df.drop_duplicates()

print('\nDimensions of Final Data:', df.shape) 
print('======================================================================')

df.to_csv('LendingTree_LoanStatus_final.csv', index=False)

###############################################################################
######################## Create sample data set  ##############################
###############################################################################
df_sample = df.sample(n=200000)

df_sample.to_csv('LendingTree_LoanStatus_final_sample_2e5.csv', index=False)

###############################################################################

