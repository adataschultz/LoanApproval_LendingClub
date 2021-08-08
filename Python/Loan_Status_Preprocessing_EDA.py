# -*- coding: utf-8 -*-
"""
@author: aschu
"""
print('\nLoan Status EDA') 
print('======================================================================')

import os
import pandas as pd
import numpy as np
from numpy import sort
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from joblib import parallel_backend
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.inspection import permutation_importance
import shap
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from group_lasso.utils import extract_ohe_groups
import scipy.sparse
from group_lasso import LogisticGroupLasso
import matplotlib.backends.backend_pdf
from pandas_profiling import ProfileReport
import sweetviz as sv

path = r'D:\Loan-Status\Data'
os.chdir(path)

# Read file
df = pd.read_csv('loan_Master.csv', index_col=False, low_memory=False)

print('\nDimensions of Initial Data:', df.shape) 
print('======================================================================')

###############################################################################
# Create sample of initial data
df_sample = df.sample(n=70000)
df_sample.to_csv('LendingTree_LoanStatus_sample_7e4.csv', index=False)

del df_sample

###############################################################################
# Change path to EDA
path = r'D:\Loan-Status\Python\EDA'
os.chdir(path)

# Replace empty with NA
df = df.replace(r'^\s*$', np.nan, regex=True)
df.isna().sum()

# Remove columns with more than 95% missing
df = df.loc[:, df.isnull().mean() < 0.05]
print('\nDimensions of Data when columns >95% missing removed:', df.shape) 
print('\n63 columns were removed due to high missingness') #
print('======================================================================')

###############################################################################
# Examine Qualitative vars
df1 = df.select_dtypes(include = 'object')
print('The selected dataframe has ' + str(df1.shape[1]) +
       ' columns that are qualitative variables.')
print('======================================================================')

# Examine missing data of qual vars
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        var_type = df.dtypes
        mis_val_table = pd.concat([mis_val, mis_val_percent, var_type], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Data Type'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ('The selected dataframe has ' + str(df.shape[1]) + ' columns.\n'      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              ' columns that have missing values.')
        return mis_val_table_ren_columns

print('\nMissing Data Report') 
pd.set_option('display.max_rows', None)
print(missing_values_table(df1))
print('======================================================================')

del df1

# Drop based on missing and research questions
df = df.drop(['title', 'earliest_cr_line', 'last_pymnt_d',
              'last_credit_pull_d'], axis=1)

# Generate data quality report
def data_quality_table(df):
        mis_val = df.isnull().sum()
        var_type = df.dtypes
        unique_count = df.nunique()
        mis_val_table = pd.concat([mis_val, var_type, unique_count], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Number Missing', 1 : 'Data Type', 2 : 'Number Unique'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        'Number Missing', ascending=False).round(1)
        print ('The selected dataframe has ' + str(df.shape[1]) + ' columns.\n')
        return mis_val_table_ren_columns

pd.set_option('display.max_columns', None)
print('\nData Quality Report - Initial') 
print(data_quality_table(df))
print('======================================================================')

# Drop based on missing and business questions (continued)
df = df.drop(['zip_code', 'sub_grade', 'issue_d', 'addr_state', 'policy_code'],
             axis=1)

###############################################################################
# Remove rows with any column having NA/null for some important variables for complete cases
df = df[df.bc_util.notna() & df.percent_bc_gt_75.notna() & df.pct_tl_nvr_dlq.notna() 
         & df.mths_since_recent_bc.notna() & df.dti.notna()
         & df.inq_last_6mths.notna() & df.num_rev_accts.notna()]

print('\nData Quality Report - Complete Cases') 
print(data_quality_table(df))
print('======================================================================')

print('\nDimensions of Data without any missing data dropped due to being irrelavant:',
      df.shape) 
print('======================================================================')

###############################################################################
# Examine dependent variable: Status of Loan
print('\nExamine Dependent Variable for Classification - Loan Status') 
print((df[['loan_status']].value_counts() / len(df)) * 100)
print('======================================================================')

###############################################################################
# Convert loan status to binary for classification
# Convert current = 0, default = 1
df = df.copy()
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
print((df[['loan_status']].value_counts() / len(df)) * 100)
print('======================================================================')

###############################################################################
########################## Variable Selection #################################
###############################################################################
######################   1. SelectFromModel using Xgboost #####################
########################   2. Group Lasso  ####################################  
###############################################################################
###############################################################################
# Separate input features and target
X = df.drop('loan_status', axis=1)
y = df.loan_status

# Create dummy variables for categorical variables    
X = pd.get_dummies(X, drop_first=True)

###############################################################################
######################   1. SelectFromModel using Xgboost #####################
###############################################################################
# Fit baseline model on all data
model = XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                      random_state=42)

# Train the Xgboost classifier
with parallel_backend('threading', n_jobs=-1):
    model.fit(X, y)

# Make predictions for accuracy
y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y, predictions)
print('Accuracy: %.3f%%' % (accuracy * 100.0)) # Accuracy: NoVIF: 98.947% vs post VIF: 96.84%
print('======================================================================')

# Xgboost - plot feature importance
plt.rcParams['figure.figsize'] = (21, 14)
plt.rcParams.update({'font.size': 7})
ax = plot_importance(model)
fig = ax.figure
pyplot.show()
fig.savefig('xgboost_featureImportance_noVIF_AllData.png')
#fig.savefig('xgboost_featureImportance_afterVIF_AllData.png')

# Fit model using each importance as a threshold
# Run for all features and then repeat for features after VIF to compare (X -> X1)
feat_max = X.shape[1] # max number of features
feat_min = 2 # minimum number of features
acc_max = accuracy
thresholds = sort(model.feature_importances_)
thresh_goal = thresholds[0]
accuracy_list = []
for thresh in thresholds:
    # Select features using threshold:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X = selection.transform(X)
    # Train model:
    selection_model = XGBClassifier(eval_metric='logloss',
                                    use_label_encoder=False)
    with parallel_backend('threading', n_jobs=-1):
        selection_model.fit(select_X, y)
    # Eval model:
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

print('\nThe optimal threshold is:')
print(thresh_goal) #noVIF: 0.00014451929 vs post VIF: 0.00032666008
print('======================================================================')
print('======================================================================')

# Create df for number features and accuracy 
key_list = list(range(X.shape[1], 0, -1))
accuracy_dict = dict(zip(key_list, accuracy_list))
accuracy_df = pd.DataFrame(accuracy_dict.items(), columns=['n_features',
                                                           'Accuracy'])
accuracy_df.to_csv('selectFromModel_Xgboost_nFeatures_Accuracy.csv',
                   index=False)

# Select features using optimal threshold with least number of features
selection = SelectFromModel(model, threshold=thresh_goal, prefit=True)

# Fit model for feature importance
feature_names = X.columns[selection.get_support(indices=True)]
print('\nXgboost resulted in ' + str(len(feature_names)) +
       ' features.')
print('\nFeatures selected using optimal threshold for accuracy from xgboost:')
print(X.columns[selection.get_support()]) 

# Create new feature importance chart
X = pd.DataFrame(data=X, columns=feature_names)

model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

# Train the Xgooost classifier
with parallel_backend('threading', n_jobs=-1):
    model.fit(X, y)

# Make predictions for test data and evaluate
y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y, predictions)
print('Accuracy: %.3f%%' % (accuracy * 100.0)) # NoVIF: 98.953% vs post VIF: 96.854%

# Xgboost - plot feature importance
plt.rcParams['figure.figsize'] = (21, 14)
plt.rcParams.update({'font.size': 7})
ax = plot_importance(model)
fig = ax.figure
pyplot.show()
# No VIF
fig.savefig('xgboost_featureImportance_Thresh_0.0001445.png')
# Removing variables with VIF
#fig.savefig('xgboost_featureImportance_Thresh_0.0003267.png')
print('======================================================================') 

# Permutation Based Feature Importance (with scikit-learn)
perm_importance = permutation_importance(model, X, y)

# Visualize Permutation Based Feature Importance
plt.rcParams['figure.figsize'] = (21, 14)
plt.rcParams.update({'font.size': 7})
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.savefig('xgboost_PermutationfeatureImportance_noVIF_bestThresh.png')

###############################################################################
# Feature Importance Computed with SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize feature importance with SHAP
fig = plt.figure()
plt.rcParams.update({'font.size': 7})
shap.summary_plot(shap_values, X, show=False)
fig.savefig('Shap_summary_Xgboost_bestThresh.png', dpi=fig.dpi, 
            bbox_inches='tight')

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
def calculate_vif_(X, threshold=5.0):
    features = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print('\nThe starting number of quantitative features is: ' + str(len(features)))
        vif = Parallel(n_jobs=-1,
                       verbose=5)(delayed(variance_inflation_factor)(X[features].values,
                                                                     ix) for ix in range(len(features)))
        maxloc = vif.index(max(vif))
        if max(vif) > threshold:
            print(time.ctime() + ' dropping \'' + X[features].columns[maxloc] + '\' at index: ' + str(maxloc))
            features.pop(maxloc)
            dropped=True
    print('Features Remaining:')
    print([features])
    return X[[i for i in features]]

# Calculate VIF on numerical data using threshold = 5
X1 = calculate_vif_(df_num, 5) 
print('\nNumber of quant features after VIF:', X1.shape[1]) 

print(X1.shape) #(2162365, 32)
print('\nQuant features remaining after VIF:')
print(X1.columns )
print('======================================================================')

# Select qual vars to merge with filtered quant
df1 = df.select_dtypes(include = 'object')

# Concatenate filtered quant, qual and dependent variable
df1 = pd.concat([X1, df1], axis=1)     
df1 = pd.concat([y, df], axis=1)     

###############################################################################
#################   2. Group Lasso for Variable Selection  ####################
###############################################################################
# Set seed for numpy array
np.random.seed(0)
LogisticGroupLasso.LOG_LOSSES = True

# Convert categorical var names to list
df_cat = df1.select_dtypes(include = 'object')
cat_columns =  df_cat.columns.tolist()

# Convert numeric var names to list
df_num = df1.select_dtypes(include = ['float64', 'int64'])
num_columns =  df_num.columns.tolist()

# Scale numerical data for linear model fitting
scaler = StandardScaler()
scaler.fit(df1[num_columns].fillna(0))

# Convert caterical to one hot encoding
ohe = OneHotEncoder()
onehot_data = ohe.fit_transform(df1[cat_columns])
groups = extract_ohe_groups(ohe)

# Combine categorical one hot encoded with scaled numerical in sparse matrix
X2 = scipy.sparse.hstack([onehot_data,scipy.sparse.csr_matrix(df1[num_columns])])
y = df1['loan_status']

# Make the groups for the lasso
groups = np.hstack([groups,len(cat_columns) + np.arange(len(num_columns))+1])
print('The groups consist of ' + str(groups) + ' for the group lasso.')
print('======================================================================') 

# Generate estimator & train model using GridSearch for iterations & tolerance
# Create parameter grid
param_grid = {
    'n_iter': [10000, 150000],
    'tol': [1e-05, 1e-06, 1e-07]
    }

# Grid search using accuracy scoring with CV
grid = GridSearchCV(estimator = LogisticGroupLasso(
    groups=groups, group_reg=0.05, l1_reg=0, scale_reg=None, 
    supress_warning=True, random_state=42, param_grid=param_grid,
    scoring='accuracy', cv = 5))

# Fit the models in the grid search
with parallel_backend('threading', n_jobs=-1): 
    grid.fit(X2, y)

print('\nGroup Lasso - Best Estimator from GridSearch CV:')
print(grid.best_estimator_)
print('======================================================================') 

print('\nGroup Lasso - Best Paramters from GridSearch CV:')
print(grid.best_params_)
print('======================================================================') 

print('\nGroup Lasso - Best Accuracy from GridSearch CV:')
print(grid.best_score_)
print('======================================================================') 

print('\nGroup Lasso - Results from GridSearch CV:')
print(grid.cv_results_)
print('======================================================================') 

# Generate estimator and train it
gl = LogisticGroupLasso(
    groups=groups,
    group_reg=0.05,
    n_iter=15000,
    tol=1e-09, 
    l1_reg=0,
    scale_reg=None,
    supress_warning=True,
    random_state=42,
)
# Fit the model using results from grid search
with parallel_backend('threading', n_jobs=-1):
    gl.fit(X2, y)

# Extract results from estimator and compute performance metrics
pred_y = gl.predict(X1)
sparsity_mask = gl.sparsity_mask_ #Boolean mask of features used in prediction
accuracy = (pred_y == y).mean()

# Print results
print(f'Number of total variables: {len(sparsity_mask)}')
print(f'Number of chosen variables: {sparsity_mask.sum()}')
print(f'Accuracy: {accuracy}')

# Convert variable position to series and list to match with location in df
tdf = pd.Series(list(gl.chosen_groups_)).T
tdf = tdf.values.tolist()

# Separate input features and target to combine with vars selected by group lasso
X2 = df.drop('loan_status', axis=1)
X2 = X2.iloc[:,tdf]
variables = X2.columns.tolist()
print(f'Selected variables from group lasso: {variables}')

# Examine loss
plt.figure()
plt.plot(gl.losses_)
plt.show()

# Accuracy from group lasso not comparable to other methods so not using further
del X2, tdf, variables

###############################################################################
############### Explore Results from Variable Selection Methods ###############
###############################################################################
# Use SelectFromModel=Xgboost X,y results from variable selection 
# As used in the previous sections
X_xgb_all = X
X_xgb_vif = X1

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
            'int_rate', 'inq_last_6mths',  'delinq_2yrs', 'installment',
            'collections_12_mths_ex_med',  'open_acc', 'loan_amnt',
            'funded_amnt', 'annual_inc',  'num_tl_op_past_12m',
            'home_ownership_OTHER',  'total_bc_limit']]

X_mfs = X_mfs.drop_duplicates()
print('\nDimensions of Data using Variables selected from MVSIS:', X_mfs.shape) 
print('======================================================================')

###############################################################################
# Find differences in variables from different variable selection methods
# Difference between important features using xgboost=all vs after vif
s = set(X_xgb_all)
varDiff_xgb = [x for x in X_xgb_vif if x not in s]
print('\nFeatures in xgb using VIF but not in Xgb - all:', varDiff_xgb) 
print('\nNumber of different features: ' + str(len(varDiff_xgb)))

s1 = set(X_xgb_vif)
varDiff_xgbVIF = [x for x in X_xgb_all if x not in s1]
print('\nFeatures in xgb_all but not Xgb_VIF: ', varDiff_xgbVIF) 
print('Number of different features: ' + str(len(varDiff_xgbVIF)))

varDiff_mvsisAll = [x for x in X_mfs if x not in s]
print('\nFeatures in MVSIS but not in xgb_all: ', varDiff_mvsisAll) 
print('Number of different features: ' + str(len(varDiff_mvsisAll)))

varDiff_mvsisVIF = [x for x in X_mfs if x not in s]
print('\nFeatures in MVSIS but not in xgb_VIF: ', varDiff_mvsisVIF) 
print('Number of different features: ' + str(len(varDiff_mvsisVIF)))

s1 = set(X_mfs)
varDiff_mvsisAll1 = [x for x in X_xgb_all if x not in s1]
print('\nFeatures in xgb_all but not in MV-SIS: ', varDiff_mvsisAll1) 
print('Number of different features: ' + str(len(varDiff_mvsisAll1)))

varDiff_mvsisVIF1 = [x for x in X_xgb_vif if x not in s1]
print('\nFeatures in xgb_VIF but not MV-SIS: ', varDiff_mvsisVIF1) 
print('Number of different features: ' + str(len(varDiff_mvsisVIF1)))
print('======================================================================')

# Add variables found in both MVSIS and xgb_VIF and only MVSIS to set for EDA
df_tmp = X2[['num_il_tl', 'num_op_rev_tl', 'num_accts_ever_120_pd',
             'mo_sin_old_rev_tl_op', 'percent_bc_gt_75', 'revol_util',
             'num_actv_rev_tl', 'tot_coll_amt', 'mort_acc', 'delinq_2yrs',
             'open_acc', 'num_tl_op_past_12m', 'home_ownership_OTHER']]

df_X = pd.concat([X_xgb_all, df_tmp], axis=1)
df = pd.concat([df_X, y], axis=1)
print('\nDimensions of data using for further EDA:', df.shape) 
print('======================================================================')

del X, X2, s, s1, varDiff_xgb, varDiff_xgbVIF, varDiff_mvsisAll, varDiff_mvsisVIF
del X_mfs, varDiff_mvsisAll1, varDiff_mvsisVIF1, df_tmp

###############################################################################
# Set path for dats
path = r'D:\Loan-Status\Data'
os.chdir(path)

# Write to csv for EDA in Spark
df.to_csv('LendingTree_LoanStatus_EDA.csv', index=False)

###############################################################################
######################## Exploratory Data Analysis ############################
###############################################################################
# Set path for EDA results
path = r'D:\Loan-Status\Python\EDA'
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
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def find_top_correlations(df, n):
    au_corr = df.corr(method='spearman').abs().unstack()
    labels_to_drop = find_repetitive_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print('The 20 features with the highest correlations:')
print(find_top_correlations(df_num, 20))
print('======================================================================')

# Create correlation matrix
corr = df_num.corr(method='spearman') 

# Create correlation heatmap of highly correlated features
my_dpi=96
fig = plt.figure()
plt.rcParams['figure.figsize'] = (21, 14)
plt.rcParams.update({'font.size': 6})
ax = sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={'size': 5}, square=True);

plt.title('Correlation Matrix with Spearman rho')
fig.savefig('EDA_correlationMatrix_spearman.png', dpi=my_dpi * 10,
            bbox_inches='tight')

###############################################################################
# Overlaid histograms of quant vars for Loan_status
pdf = matplotlib.backends.backend_pdf.PdfPages("EDA_Quant_Histograms_output.pdf")
for var in df_num.columns[:]:
  fig = plt.figure(figsize=(8.27, 11.69), dpi=my_dpi * 10)
  sns.histplot(x=var, data=df_num, hue=df.loan_status, kde=True)
  sns.despine(offset=10, trim=True) 
  fig.set_size_inches(22,14)
  pdf.savefig(fig)
pdf.close()

# Box-and-whisker plots of quant vars for Loan_status
pdf = matplotlib.backends.backend_pdf.PdfPages("EDA_Quant_Boxplots_output.pdf")
for var in df_num.columns[:]:
  fig = plt.figure(figsize=(8.27, 11.69), dpi=my_dpi * 10)
  sns.boxplot(x=df.loan_status, y=var, data=df_num)
  fig.set_size_inches(22,14)
  pdf.savefig(fig)
pdf.close()

###############################################################################
###############################################################################    
# Examine Qualitative vars
df_cat = df.select_dtypes(include = 'uint8')

print('The selected dataframe has ' + str(df_cat.shape[1]) +
       ' columns that are qualitative variables.')
print('======================================================================')

# Count plot
fig, ax = plt.subplots(5, 4, figsize=(11.7, 8.27))
for variable, subplot in zip(df_cat, ax.flatten()):
    sns.countplot(df_cat[variable], ax=subplot)
plt.tight_layout()  
fig.savefig('QualVar_Countplot.png', dpi=my_dpi * 10, bbox_inches='tight')

###############################################################################
# Automated EDA with Sweetviz after cleaning
sweet_report = sv.analyze(df)
sweet_report.show_html('Loan_Status_AutomatedEDA.html')

###############################################################################
# Automated EDA using Pandas Profiling after cleaning
profile = ProfileReport(df, title='Loan Status_EDA')
profile.to_file(output_file='Loan_Status_EDA.html')

###############################################################################







