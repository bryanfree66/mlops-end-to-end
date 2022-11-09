import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb
import hypertune
import argparse
import logging
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.cloud import bigquery

# SET UP TRAINING SCRIPT ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument("--project-id", dest="project_id",
                    type=str, help="Project id for bigquery client.")
parser.add_argument("--bq-table", dest="bq_table",
                    type=str, help="Download url for the training data.")
parser.add_argument("--boost-rounds", dest="boost_rounds",
                    default=20, type=int, help="Number of boosted rounds")
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

# Function to retrieve data from BigQuery
def get_data():
    logging.info("Downloading training data from BigQuery: {}, {}".format(args.project_id, args.bq_table))
    logging.info("Creating BigQuery client")
    bqclient = bigquery.Client(project=args.project_id)
    
    logging.info("Loading table data")
    table = bigquery.TableReference.from_string(args.bq_table)
    rows = bqclient.list_rows(table)
    dataframe = rows.to_dataframe()
    
    logging.info("Preparing data for training")
    dataframe.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)
    X = pd.concat([dataframe.drop('type', axis=1), pd.get_dummies(dataframe['type'])], axis=1)
    y = X[['isFraud']]
    X = X.drop(['isFraud'],axis=1)
    
    logging.info("Splitting data for training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42, shuffle=True)
    data_train = xgb.DMatrix(X_train, label=y_train)
    
    logging.info("Finishing get_data")
    return data_train, X_test, y_test

# Function to train the model
def train_model(data_train):
    logging.info("Start training ...")
    params = {
        'objective': 'multi:softprob',
        'num_class': 2
    }
    model = xgb.train(params, data_train, num_boost_round=args.boost_rounds)
    
    logging.info("Training completed")
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    logging.info("Preparing test data ...")
    data_test = xgb.DMatrix(X_test)
    
    logging.info("Getting test predictions ...")
    pred = model.predict(data_test)
    predictions = [np.around(value) for value in pred]
    
    logging.info("Evaluating predictions ...")
    try:
        accuracy = accuracy_score(y_test, predictions)
    except:
        accuracy = 0.0
    logging.info(f"Evaluation completed with model accuracy: {accuracy}")

    logging.info("Report metric for hyperparameter tuning ...")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=accuracy
    )
    
    logging.info("Finishing ...")
    return accuracy

data_train, X_test, y_test = get_data()
model = train_model(data_train)
accuracy = evaluate_model(model, X_test, y_test)

# GCSFuse conversion
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'
if args.model_dir.startswith(gs_prefix):
    args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
    dirpath = os.path.split(args.model_dir)[0]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

# Export the classifier to a file
gcs_model_path = os.path.join(args.model_dir, 'model.bst')
logging.info("Saving model artifacts to {}". format(gcs_model_path))
model.save_model(gcs_model_path)

logging.info("Saving metrics to {}/metrics.json". format(args.model_dir))
gcs_metrics_path = os.path.join(args.model_dir, 'metrics.json')
with open(gcs_metrics_path, "w") as f:
    f.write(f"{'accuracy: {accuracy}'}")
