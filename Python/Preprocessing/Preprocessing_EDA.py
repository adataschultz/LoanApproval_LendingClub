# -*- coding: utf-8 -*-
"""
@author: aschu
"""
print('\nLoan Status EDA') 
print('======================================================================')

import os
import random
import warnings
import numpy as np
from numpy import sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import parallel_backend, import Parallel, delayed
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
import shap
import time
from datetime import datetime, timedelta
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from group_lasso.utils import extract_ohe_groups
import scipy.sparse
from group_lasso import LogisticGroupLasso
import sweetviz as sv
from ydata_profiling import ProfileReport
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
my_dpi = 96

# Set seed
seed_value = 42
os.environ['LoanStatus_PreprocessEDA'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\LoanStatus\Data'
os.chdir(path)

# Read data
df = pd.read_csv('loan_Master.csv', index_col=False, low_memory=False)
print('- Dimensions of initial data:', df.shape)

###############################################################################
# Create sample of initial data
df_sample = df.sample(n=70000)
df_sample.to_csv('LendingTree_LoanStatus_sample_7e4.csv', index=False)

del df_sample

###############################################################################
# Change path to EDA
path = r'D:\LoanStatus\Python\EDA'
os.chdir(path)

# Remove columns with more than 95% missing
df = df.replace(r'^\s*$', np.nan, regex=True)
df1 = df.loc[:, df.isnull().mean() < 0.05]
print('- Dimensions when columns > 95% missing removed:', df1.shape)

s = set(df1)
varDiff = [x for x in df if x not in s]
print('- Number of features removed due to high missingness:'
      + str(len(varDiff)))

df = df1
del df1
print('======================================================================')

###############################################################################
# Define a function to examine data for data types, percentage missing and unique values
def data_quality_table(df):
    """Returns the characteristics of variables in a Pandas dataframe."""
    var_type = df.dtypes
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    unique_count = df.nunique()
    mis_val_table = pd.concat([var_type, mis_val_percent, unique_count], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0: 'Data Type', 1: 'Percent Missing', 2: 'Number Unique'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] >= 0].sort_values(
            'Percent Missing', ascending=False).round(1)
    print ('- There are ' + str(df.shape[0]) + ' rows and '
           + str(df.shape[1]) + ' columns.\n')
    return mis_val_table_ren_columns

# Categorical variables
# Examine dimensionality
# Drop based on missing and questions
df1 = df.select_dtypes(include = 'object')
print('\n              Data Quality: Qualitative Variables')
display(data_quality_table(df1))
print('\n')
print('\nSample observations of qualitative variables:')
display(df1.head())

df = df.drop(['title', 'last_pymnt_d', 'zip_code', 'earliest_cr_line',
              'last_credit_pull_d', 'issue_d', 'addr_state', 'sub_grade'],
             axis=1)

del df1

# Quantitative variables
# Remove rows with any column having NA/null for some important variables for complete cases
df1 = df.select_dtypes(exclude = 'object')
print('\n              Data Quality: Quantitative Variables')
display(data_quality_table(df1))
print('\n')
print('\nSample observations of quantitative variables:')
display(df1.head())

df = df[df.bc_util.notna() & df.percent_bc_gt_75.notna()
        & df.pct_tl_nvr_dlq.notna() & df.mths_since_recent_bc.notna()
        & df.dti.notna() & df.inq_last_6mths.notna() & df.num_rev_accts.notna()]

del df1

print('\nData Quality Report - Complete Cases') 
print(data_quality_table(df))
print('======================================================================')

print('\nDimensions of Data without any missing data dropped due to being irrelavant:',
      df.shape) 
print('======================================================================')

###############################################################################
# Examine dependent variable: Status of Loan
print('\nExamine Dependent Variable for Classification - Loan Status') 
print(df.loan_status.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print('======================================================================')

###############################################################################
# Convert loan status to binary for classification
# Convert current = 0, default = 1
df['loan_status'] = df['loan_status'].replace(['Fully Paid'], 0)
df['loan_status'] = df['loan_status'].replace(['In Grace Period'], 0)
df['loan_status'] = df['loan_status'].replace(['Current'], 0)

df['loan_status'] = df['loan_status'].replace(['Charged Off'], 1)
df['loan_status'] = df['loan_status'].replace(['Late (31-120 days)'], 1)
df['loan_status'] = df['loan_status'].replace(['Late (16-30 days)'], 1)
df['loan_status'] = df['loan_status'].replace(['Does not meet the credit policy. Status:Fully Paid'], 1)
df['loan_status'] = df['loan_status'].replace(['Does not meet the credit policy. Status:Charged Off'], 1)
df['loan_status'] = df['loan_status'].replace(['Default'], 1)

# After recoding into binary, there is clear class imbalance
print('\nExamine Binary Loan Status for Class Imbalance') 
print(df.loan_status.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print('======================================================================')

###############################################################################
########################## Variable Selection #################################
###############################################################################
# Separate input features and target
X = df.drop('loan_status', axis=1)
y = df.loan_status

# Create dummy variables for categorical variables    
X = pd.get_dummies(X, drop_first=True)

###############################################################################
######################   1. SelectFromModel using XGBoost #####################
###############################################################################
# Fit baseline model on all data
model = XGBClassifier(eval_metric='logloss', 
                      use_label_encoder=False,
                      tree_method='gpu_hist', 
                      gpu_id=0,
                      random_state=seed_value)

model.fit(X, y)

y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y, predictions)
print('Accuracy: %.3f%%' % (accuracy * 100.0)) 
print('======================================================================')

# XGBoost - plot feature importance
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 8.5})
ax = plot_importance(model)
fig = ax.figure
plt.tight_layout()
fig.savefig('xgb_featureImportance_noVIF_AllData.png', dpi=my_dpi*10, 
            bbox_inches='tight')
plt.show();

# Permutation importance
perm_importance = permutation_importance(model, X, y)

plt.rcParams.update({'font.size': 7})
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance')
plt.tight_layout()
plt.savefig('xgb_PermutationfeatureImportance_noVIF_AllData.png', dpi=my_dpi*10, 
            bbox_inches='tight'))
plt.show();

# Visualize feature importance with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.rcParams.update({'font.size': 7})
fig = plt.figure()
shap.summary_plot(shap_values, X, show=False)
fig.savefig('ShapSummary_xgb_noVIF_AllData.png', dpi=my_dpi*10, 
            bbox_inches='tight')
plt.show();

# Fit model using each importance as a threshold
# Run for all features and then repeat for features after VIF to compare (X -> X1)
print('Time for feature selection using XGBoost...')
search_time_start = time.time()
feat_max = X.shape[1]
feat_min = 2
acc_max = accuracy
thresholds = sort(model.feature_importances_)
thresh_goal = thresholds[0]
accuracy_list = []
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X = selection.transform(X)
    
    # Define model
    selection_model = XGBClassifier(eval_metric='logloss',
                                    use_label_encoder=False,
                                    tree_method='gpu_hist',
                                    gpu_id=0,
                                    random_state=seed_value)
    # Train model
    selection_model.fit(select_X, y)
    
    # Evaluate model
    selection_model_pred = selection_model.predict(select_X)
    selection_predictions = [round(value) for value in selection_model_pred]
    accuracy = accuracy_score(y_true=y, y_pred=selection_predictions)
    accuracy = accuracy * 100
    print('Thresh= %.6f, n= %d, Accuracy: %.3f%%' % (thresh, select_X.shape[1],
                                                     accuracy))
    accuracy_list.append(accuracy)
    if(select_X.shape[1] < feat_max) and (select_X.shape[1] >= feat_min) and (accuracy >= acc_max):
      n_min = select_X.shape[1]
      acc_max = accuracy
      thresh_goal = thresh
        
print('\n')
print('Finished feature selection using XGBoost in:',
      time.time() - search_time_start)
print('\n')
print('\nThe optimal threshold is:')
print(thresh_goal)
print('======================================================================')

# Create df for number features and accuracy 
key_list = list(range(X.shape[1], 0, -1))
accuracy_dict = dict(zip(key_list, accuracy_list))
accuracy_df = pd.DataFrame(accuracy_dict.items(), columns=['n_features',
                                                           'Accuracy'])
accuracy_df.to_csv('selectFromModel_xgb_nFeatures_Accuracy.csv',
                   index=False)

# Select features using optimal threshold with least number of features
selection = SelectFromModel(model, threshold=thresh_goal, prefit=True)

# Fit model for feature importance
feature_names = X.columns[selection.get_support(indices=True)]
print('\n- Feature selection using XGBoost resulted in '
      + str(len(feature_names)) + ' features.')
print('\n- Features selected using optimal threshold for accuracy:')
print(X.columns[selection.get_support()]) 

# Create new feature importance chart
X = pd.DataFrame(data=X, columns=feature_names)

model = XGBClassifier(eval_metric='logloss', 
                      use_label_encoder=False, 
                      tree_method='gpu_hist', 
                      gpu_id=0,
                      random_state=seed_value)
model.fit(X, y)
model.save_model('xgb_featureSelection.model')

y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y, predictions)
print('Accuracy: %.3f%%' % (accuracy * 100.0)) 
print('======================================================================') 

# No VIF
# XGBoost - plot feature importance
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 10})
ax = plot_importance(model)
fig = ax.figure
plt.tight_layout()
fig.savefig('xgb_featureImportance_bestThresh.png', dpi=my_dpi*10, 
            bbox_inches='tight')
plt.show();

# Permutation Based Feature Importance (with scikit-learn)
perm_importance = permutation_importance(model, X, y)

# Visualize Permutation Based Feature Importance
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 10})
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance')
plt.savefig('xgb_PermutationfeatureImportance_noVIF_bestThresh.png')
plt.show();

###############################################################################
# Feature Importance Computed with SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize feature importance with SHAP
fig = plt.figure()
plt.rcParams.update({'font.size': 7})
shap.summary_plot(shap_values, X, show=False)
fig.savefig('ShapSummary_xgb_bestThresh.png', dpi=my_dpi*10, 
            bbox_inches='tight')
plt.show();

###############################################################################
#################   Multicollinearity using VIF for Linear  ###################
###############################################################################
# Separate input features and target
X1 = df.drop('loan_status', axis=1)
y = df.loan_status

# Check for multicollinearity using Variance Inflation Factor
# Select numeric data
df_num = X1.select_dtypes(include = ['float64', 'int64'])

# Defining the VIF function for multicollinearity
def calculate_vif(X, threshold=5.0):
    features = [X.columns[i] for i in range(X.shape[1])]
    dropped = True
    while dropped:
        dropped = False
        print('\nThe starting number of quantitative features is: '
              + str(len(features)))
        vif = Parallel(n_jobs=-1,
                       verbose=5)(delayed(variance_inflation_factor)(X[features].values,
                                                                     ix) for ix in range(len(features)))
        maxloc = vif.index(max(vif))
        if max(vif) > threshold:
            print(time.ctime() + ' dropping \'' + X[features].columns[maxloc]
                  + '\' at index: ' + str(maxloc))
          features.pop(maxloc)
          dropped = True
  print('Features Remaining:')
  print([features])
  return X[[i for i in features]]

print('Time for calculating VIF on numerical data using threshold = 5...')
search_time_start = time.time()

X1 = calculate_vif(df_num, 5) 
print('\nNumber of quant features after VIF:', X1.shape[1]) 

print('Finished calculating VIF on numerical data using threshold = 5 in:',
      time.time() - search_time_start)
print ('- There are ' + str(X1.shape[0]) + ' rows and '
       + str(X1.shape[1]) + ' columns.\n')
print('\nQuant features remaining after VIF:')
print(X1.columns)
print('======================================================================')

# Select qual vars to merge with filtered quant
df1 = df.select_dtypes(include='object')

# Concatenate filtered quant, qual and dependent variable
df1 = pd.concat([X1, df1], axis=1)     
df1 = pd.concat([y, df], axis=1)     

###############################################################################
#################   2. Group Lasso for Variable Selection  ####################
###############################################################################
df_num = df1.select_dtypes(include = ['float64', 'int64'])
df_num = df_num.drop(['loan_status'], axis=1)
num_columns = df_num.columns.tolist()

# Scale numerical data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_num)

# Select categorical variables
df_cat = df1.select_dtypes(include = 'object')
cat_columns =  df_cat.columns.tolist()

# One hot encode cat vars
ohe = OneHotEncoder()
onehot_data = ohe.fit_transform(df1[cat_columns])

# Create sparse matrix
X2 = scipy.sparse.hstack([onehot_data, scipy.sparse.csr_matrix(scaled)])
y = df1['loan_status']

# Extract groups
groups = extract_ohe_groups(ohe)
groups = np.hstack([groups, len(cat_columns) + np.arange(len(num_columns))+1])
print('The groups consist of ' + str(groups) + ' for the group lasso.')

# Generate estimator & train model using GridSearch for iterations & tolerance
LogisticGroupLasso.LOG_LOSSES = True

# Create parameter grid
params = { 
    'n_iter': [3000],
    'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }

# Define grid search conditions
grid = GridSearchCV(estimator = LogisticGroupLasso( 
                    groups=groups, group_reg=0.05, l1_reg=0, scale_reg=None, 
                    supress_warning=True, random_state=seed_value), 
                    scoring='accuracy', cv=5, param_grid=params)

print('Time for feature selection using GroupLasso GridSearchCV...')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    grid.fit(X2, y)
print('Finished feature selection using GroupLasso GridSearchCV in:',
      time.time() - search_time_start)

print('\nGroup Lasso GridSearchCV Feature selection')
print('\nBest Estimator:')
print(grid.best_estimator_)
print('\nBest Parameters:')
print(grid.best_params_)
print('\nBest Accuracy:')
print(grid.best_score_)
print('\nResults from GridSearch CV:')
print(grid.cv_results_)
print('======================================================================') 

# Fit the model using results from grid search
gl = LogisticGroupLasso(
    groups=groups,
    group_reg=0.05,
    n_iter=3000,
    tol=0.1, 
    l1_reg=0,
    scale_reg=None,
    supress_warning=True,
    random_state=seed_value,
)

with parallel_backend('threading', n_jobs=-1):
    gl.fit(X2, y)

pred_y = gl.predict(X2)
sparsity_mask = gl.sparsity_mask_ 
accuracy = (pred_y == y).mean()

print(f'Number of total variables: {len(sparsity_mask)}')
print(f'Number of chosen variables: {sparsity_mask.sum()}')
print(f'Accuracy: {accuracy}')

tdf = pd.Series(list(gl.chosen_groups_)).T
tdf = tdf.values.tolist()

X2 = df1.drop('loan_status', axis=1)
X2 = X2.iloc[:,tdf]
variables = X2.columns.tolist()
print(f'Selected variables from group lasso: {variables}')

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams.update({'font.size': 15})
plt.plot(gl.losses_)
plt.tight_layout()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Group Lasso: Loss over the Number of Iterations')
plt.savefig('groupLasso_bestModel_3000iterTol1e-1_loss.png', dpi=my_dpi*10, 
            bbox_inches='tight')
plt.show();
# Accuracy from group lasso not comparable to other methods so not using further

del X2, tdf, variables

###############################################################################
############### Explore Results from Variable Selection Methods ###############
###############################################################################
# Use SelectFromModel=XGBoost X,y results from variable selection 
# As used in the previous sections
X_xgb = X
X_vif = X1

# Separate input features and target to restore original
X2 = df.drop('loan_status', axis=1)

# Create dummy variables for categorical variables    
X2 = pd.get_dummies(X2, drop_first=True)

# Use results from variable selection using MV-SIS completed in R
X_mfs = X2[['num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'pymnt_plan_y',
            'num_accts_ever_120_pd', 'mo_sin_old_rev_tl_op',
            'last_pymnt_amnt', 'percent_bc_gt_75', 'revol_util',
            'tot_hi_cred_lim', 'num_actv_rev_tl', 'disbursement_method_DirectPay',
            'tot_coll_amt', 'term_ 60 months', 'mort_acc', 'funded_amnt_inv',
            'int_rate', 'inq_last_6mths', 'delinq_2yrs', 'installment',
            'collections_12_mths_ex_med', 'open_acc', 'loan_amnt',
            'funded_amnt', 'annual_inc', 'num_tl_op_past_12m',
            'home_ownership_OTHER', 'total_bc_limit']]
X_mfs = X_mfs.drop_duplicates()
print('\nDimensions of Data using Variables selected from MVSIS:', X_mfs.shape) 
print('======================================================================')

###############################################################################
# Find differences in variables from different variable selection methods
# Difference between important features using xgboost=all vs after vif
s = set(X_xgb)
varDiff_vif = [x for x in X_vif if x not in s]
print('\nFeatures using VIF but not in XGB:')
print(varDiff_vif)
print('- Number of different features: ' + str(len(varDiff_vif)))

s1 = set(X_vif)
varDiff_xgb = [x for x in X_xgb if x not in s1]
print('\nFeatures in XGB but not in VIF:')
print(varDiff_xgb)
print('- Number of different features: ' + str(len(varDiff_xgb)))

varDiff_mvsisAll = [x for x in X_mfs if x not in s]
print('\nFeatures in MVSIS but not in XGB:')
print(varDiff_mvsisAll)
print('- Number of different features: ' + str(len(varDiff_mvsisAll)))

varDiff_mvsisVIF = [x for x in X_mfs if x not in s1]
print('\nFeatures in MVSIS but not in VIF:')
print(varDiff_mvsisVIF)
print('- Number of different features: ' + str(len(varDiff_mvsisVIF)))

s1 = set(X_mfs)
varDiff_mvsisAll1 = [x for x in X_xgb if x not in s1]
print('\nFeatures in XGB but not in MV-SIS:')
print(varDiff_mvsisAll1)
print('- Number of different features: ' + str(len(varDiff_mvsisAll1)))

varDiff_mvsisVIF1 = [x for x in X_vif if x not in s1]
print('\nFeatures in VIF but not in MV-SIS:')
print(varDiff_mvsisVIF1)
print('- Number of different features: ' + str(len(varDiff_mvsisVIF1)))
print('======================================================================')

# Add variables found in both MVSIS and xgb_VIF and only MVSIS to set for EDA
df_tmp = df[['num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_accts_ever_120_pd',
             'mo_sin_old_rev_tl_op', 'percent_bc_gt_75', 'revol_util',
             'num_actv_rev_tl', 'tot_coll_amt', 'mort_acc', 'delinq_2yrs',
             'collections_12_mths_ex_med', 'open_acc',
             'num_tl_op_past_12m', 'home_ownership_OTHER']]

df = pd.concat([X_xgb, df_tmp, y], axis=1)
df = df.drop_duplicates()
print('- Dimensions of data using for further EDA:', df.shape)
print('======================================================================')

del X_xgb, X_vif, X_mfs
del df_tmp, y, s, s1, varDiff_vif, varDiff_xgb, varDiff_mvsisAll
del varDiff_mvsisVIF, varDiff_mvsisAll1, varDiff_mvsisVIF1

###############################################################################
# Set path for data
path = r'D:\LoanStatus\Data'
os.chdir(path)

# Write to csv for EDA
df.to_csv('LendingTree_LoanStatus_EDA.csv', index=False)

###############################################################################
######################## Exploratory Data Analysis ############################
###############################################################################
# Set path for EDA results
path = r'D:\LoanStatus\Python\EDA'
os.chdir(path)

# Examine Quantitative vars
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num = df_num.drop(['loan_status'], axis=1)

print('The selected dataframe has ' + str(df_num.shape[1]) +
      ' columns that are quantitative variables.')
print('======================================================================')

# Correlations - avoid duplicate, and self correlations 
# Find the pairs of features on diagonal and lower triangular of correlation matrix
def find_repetitive_pairs(df):
    """Returns the pairs of features on the diagonal and lower triangle of the correlation matrix."""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

def find_top_correlations(df, n):
    """Returns the highest correlations without duplicates."""
    au_corr = df.corr(method='spearman').abs().unstack()
    labels_to_drop = find_repetitive_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print('- The selected dataframe has ' + str(df_num.shape[1]) + ' columns that are quantitative variables.')
print('- The 20 features with the highest correlations:')
print(find_top_correlations(df_num, 20))
print('======================================================================')

# Create correlation matrix
corr = df_num.corr(method='spearman') 

# Create correlation heatmap of highly correlated features
fig = plt.figure()
plt.rcParams['figure.figsize'] = (21, 14)
plt.rcParams.update({'font.size': 6})
ax = sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)],
                 cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                 linecolor='black', annot=True, annot_kws={'size': 9},
                 square=True)
plt.title('Correlation Matrix with Spearman rho')
fig.savefig('EDA_correlationMatrix_spearman.png', dpi=my_dpi*10,
            bbox_inches='tight')
plt.show();

###############################################################################
# Histograms of quant vars
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num = df_num.drop(['loan_status'], axis=1)

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(15,3, figsize=(21,35))
fig.suptitle('Quantitative Features: Histograms', y=1.01, fontsize=30)
for variable, subplot in zip(df_num, ax.flatten()):
    a = sns.histplot(df_num[variable], ax=subplot)
    a.set_yticklabels(a.get_yticks(), size=10)
    a.set_xticklabels(a.get_xticks(), size=9)
fig.tight_layout()
fig.savefig('QuantVar_Histplot.png', dpi=my_dpi*10, bbox_inches='tight')
plt.show();

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(15,3, figsize=(25,35))
fig.suptitle('Quantitative Features: Histograms Grouped by Loan Status', y=1.01,
             fontsize=30)
for variable, subplot in zip(df_num, ax.flatten()):
    a = sns.histplot(x=df_num[variable], data=df_num, hue=df.loan_status,
                     kde=True, ax=subplot)
    a.set_yticklabels(a.get_yticks(), size=10)
    a.set_xticklabels(a.get_xticks(), size=10)
fig.tight_layout()
fig.savefig('QuantVar_Histplot_HueLoanStatus.png', dpi=my_dpi*10, bbox_inches='tight')
plt.show();

# Increase to more granular level
df_num1 = df_num[['loan_amnt', 'funded_amnt', 'funded_amnt_inv',
                  'int_rate', 'installment', 'total_pymnt', 'total_pymnt_inv',
                  'total_rec_prncp', 'total_rec_int', 'mo_sin_old_rev_tl_op',
                  'percent_bc_gt_75', 'revol_util']]

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(4,3, figsize=(21,30))
fig.suptitle('Subset of Quantitative Features: Histograms with Grouped by Loan Status',
             y=1.01, fontsize=30)
for variable, subplot in zip(df_num1, ax.flatten()):
    sns.histplot(x=df_num1[variable], data=df_num1, hue=df.loan_status,
                 kde=True, ax=subplot)
    a.set_yticklabels(a.get_yticks(), size=10)
    a.set_xticklabels(a.get_xticks(), size=9)
fig.tight_layout()
fig.savefig('QuantVar_Histplot_HueLoanStatus_Subset.png', dpi=my_dpi*10, 
            bbox_inches='tight')
plt.show();

###############################################################################    
# Box plots
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(9,5, figsize=(21,30))
fig.suptitle('Quantitative Features: Boxplots', y=1.01, fontsize=30)
for var, subplot in zip(df_num, ax.flatten()):
    a = sns.boxplot(x=df.loan_status, y=df_num[var], data=df_num, ax=subplot)
    a.set_yticklabels(a.get_yticks(), size=12)
    a.set_xticklabels(a.get_xticks(), size=12)
fig.tight_layout()
fig.savefig('QuantVar_Boxplot.png', dpi=my_dpi*10, bbox_inches='tight')
plt.show();

###############################################################################    
# Examine Qualitative vars
df_cat = df.select_dtypes(include = 'uint8')

print('The selected dataframe has ' + str(df_cat.shape[1]) +
       ' columns that are qualitative variables.')
print('======================================================================')

# Count plot
df_cat = df.select_dtypes(include = 'uint8')

print('The selected dataframe has ' + str(df_cat.shape[1])
      + ' columns that are qualitative variables.')
print('\n')
fig, ax = plt.subplots(5, 4, figsize=(21,21))
fig.suptitle('Qualitative Features: Count Plots', y=1.01, fontsize=30)
for variable, subplot in zip(df_cat, ax.flatten()):
    a = sns.countplot(df_cat[variable], ax=subplot)
    a.set_yticklabels(a.get_yticks(), size=10)
fig.tight_layout()
fig.savefig('QualVar_Countplot.png', dpi=my_dpi*10, bbox_inches='tight')
plt.show(); 

###############################################################################
# Automated EDA with Sweetviz after cleaning
sweet_report = sv.analyze(df)
sweet_report.show_html('Loan_Status_AutomatedEDA.html')
sweet_report.show_notebook(layout='widescreen', w=1500, h=1000, scale=0.8)

###############################################################################
# Automated EDA using Pandas Profiling after cleaning
profile = ProfileReport(df, title='Loan Status_EDA')
profile.to_file(output_file='Loan_Status_EDA.html')

###############################################################################