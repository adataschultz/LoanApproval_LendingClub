import os
import random
import numpy as np
import warnings
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from joblib import parallel_backend
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

seed_value = 42
os.environ['LoanStatus_Linear'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def main():
    """Main function of the script."""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='path to input train data')
    parser.add_argument('--test_data', type=str, help='path to input test data')
    parser.add_argument('--penalty', required=False, default='l2', type=str)
    parser.add_argument('--solver', required=False, default='lbfgs', type=str)
    parser.add_argument('--max_iter', required=False, default=100, type=int)
    parser.add_argument('--C', required=False, default=1, type=int)
    parser.add_argument('--tol', required=False, default=1e-4, type=float)
    parser.add_argument('--n_jobs', required=False, default=1, type=int)
    parser.add_argument('--registered_model_name', type=str, help='model name')
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # Enable autologging
    mlflow.sklearn.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print('Input Train Data:', args.train_data)
    print('Input Test Data:', args.test_data)
    
    trainDF = pd.read_csv(args.train_data, low_memory=False)
    testDF = pd.read_csv(args.test_data, low_memory=False)

    train_label = trainDF[['loan_status']]
    test_label = testDF[['loan_status']]

    train_features = trainDF.drop(columns = ['loan_status'])
    test_features = testDF.drop(columns = ['loan_status'])

    print(f"Training with data of shape {train_features.shape}")

    mlflow.log_metric('num_samples', train_features.shape[0])
    mlflow.log_metric('num_features', train_features.shape[1])

    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    # Define model
    model = LogisticRegression(penalty=args.penalty, 
                               solver=args.solver,
                               max_iter=args.max_iter, 
                               C=args.C,
                               tol=args.tol,
                               random_state=seed_value)

    # Fit model
    with parallel_backend('threading', n_jobs=args.n_jobs):
        model.fit(train_features, train_label)    

    ##################
    #</train the model>
    ##################

    #####################
    #<evaluate the model>
    ##################### 
    # Predict
    train_label_pred = model.predict(train_features)
    test_label_pred = model.predict(test_features)

    clr_train = classification_report(train_label, train_label_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(clr_train).iloc[:-1,:].T, annot=True)
    plt.savefig('clr_train.png')
    mlflow.log_artifact('clr_train.png')
    plt.close()

    clr_test = classification_report(test_label, test_label_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(clr_test).iloc[:-1,:].T, annot=True)
    plt.savefig('clr_test.png')
    mlflow.log_artifact('clr_test.png')
    plt.close()

    cm_train = confusion_matrix(train_label, train_label_pred)
    cm_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    cm_train.plot()
    plt.savefig('cm_train.png')
    mlflow.log_artifact('cm_train.png')
    plt.close()

    cm_test = confusion_matrix(test_label, test_label_pred)
    cm_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    cm_test.plot()
    plt.savefig('cm_test.png')
    mlflow.log_artifact('cm_test.png')
    plt.close()

    train_accuracy = accuracy_score(train_label, train_label_pred)
    train_precision = precision_score(train_label, train_label_pred)
    train_recall = recall_score(train_label, train_label_pred)
    train_f1 = f1_score(train_label, train_label_pred)

    test_accuracy = accuracy_score(test_label, test_label_pred)
    test_precision = precision_score(test_label, test_label_pred)
    test_recall = recall_score(test_label, test_label_pred)
    test_f1 = f1_score(test_label, test_label_pred)

    mlflow.log_metric('train_accuracy', train_accuracy)
    mlflow.log_metric('train_precision', train_precision)
    mlflow.log_metric('train_recall', train_recall)
    mlflow.log_metric('train_f1', train_f1)
    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_precision', test_precision)
    mlflow.log_metric('test_recall', test_recall)
    mlflow.log_metric('test_f1', test_f1)

    #####################
    #</evaluate the model>
    ##################### 

    ##########################
    #<save and register model>
    ##########################
    # Registering the model to the workspace
    print('Registering the model via MLFlow')
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.registered_model_name, 'trained_model'),
    )

    ###########################
    #</save and register model>
    ###########################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
