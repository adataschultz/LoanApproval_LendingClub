# `Loan Approval from Historical Data`

## `Background`
Banks routinely lose money based on loans that eventually default. Per the Federal Reserve, at the height of the financial crisis in 2009-2010, the amount lost approached $500 billion. Most recently losses each quarter tend to approach $150 billion. Delinquency rates tend to be around 1.5% most recently. Because of this, it is vitally important for banks to ensure that they keep their delinquencies as low as possible.

- Can we accurately predict loan approval based on historical data?
- How can we confidentially determine whether a loan can be approved?

    - Rationale and Objective:  
        - If a loan is current, the company is making profit and should approve such future loans based on the model.
		- If a loan is late or in default, the company is losing capital and should reject future loans based the model.

- What factors predict loan approval?
- Which variables best predict if a loan will be a loss and how much is the average loss? 


## `Data`
The data was retrieved from [here](https://www.kaggle.com/wendykan/lending-club-loan-data). Lending Club is a peer to peer financial company. Essentially, people can request an unsecured loan between $1,000 and $40,000 while other individuals can visit the site to choose to invest in the loans. So, people are essentially lending to other people directly with Lending Club as a facilitator.


## `Methods`
- Preprocess data in `Python` and `R`
- `Variable selection`
	- ` Python`
 		-  `SelectFromModel` with XGBoost classifier utilizing GPU
		- `VIF` followed by `Group Lasso`
    
   - `R`
		- `Model-Free Screening (MV-SIS)`
		- `Boruta`
- Evaluate methods for class imbalance using:
	- `Upsampling the minority class`
 	- `Synthetic Minority Oversampling Technique (SMOTE)`
- Test selected features using linear and non-linear ML algorithms 
- Tune hyperparameters of different algorithms to increase predictive performance

## `Modeling`


### `Machine Learning`
Models were trained using the following libraries:
- `XGBoost` 
- `Catboost` 
- `LightGBM` 
- `RAPIDS`: `Logistic`/`Ridge`/`Elastic Net Regression`, `LinearSVC`, `Random Forest`, `XGBoost`, `K Nearest Neighbor`
- `SparkML`: `Logistic Regression`, `LinearSVC`, `Decision Trees`, `Random Forest`, `Gradient Boosted Trees`
- `Scikit-learn`: `Linear`, `Naive Bayes`, `Random Forest`


#### `Hyperparameter Optimization`
For hyperparameter tuning, `Optuna`, `Hyperopt`, and `GridSearchCV` were utilized to explore the model parameters that resulted in the lowest error using various metrics for classification. Various trial/experiment sizes were completed to determine which parameters when incorporated into the model resulted in the lowest error.


#### `Model Explanations`
To explain the results from modeling, `ELI5`, `SHAP` and `LIME` were utilized.


