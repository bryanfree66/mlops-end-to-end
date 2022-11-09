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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test
