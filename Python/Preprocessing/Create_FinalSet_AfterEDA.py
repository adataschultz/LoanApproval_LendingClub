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

# Drop based off high correlations and imbalance in cat vars
df = df.drop(['out_prncp_inv', 'funded_amnt', 'funded_amnt_inv', 'total_pymnt_inv',
               'open_acc', 'tot_cur_bal', 'total_rec_prncp', 'num_op_rev_tl',
               'home_ownership_OTHER', 'hardship_flag_Y', 
               'pymnt_plan_y', 'purpose_house', 'purpose_medical',
               'purpose_small_business'],  axis = 1)
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

