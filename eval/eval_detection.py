import numpy as np
import torch 
import pandas as pd
import os 
import sys

import json
import pickle
import xgboost as xgb

# Metrics
from sdmetrics import load_demo
from sdmetrics.single_table import LogisticDetection
from sdmetrics.single_table.detection.sklearn import ScikitLearnClassifierDetectionMetric

from matplotlib import pyplot as plt

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='real')

args = parser.parse_args()

class XgboostDetection(ScikitLearnClassifierDetectionMetric):
    """ScikitLearnClassifierDetectionMetric based on a XgboostClassifier.

    This metric builds a Xgboost Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.
    """

    name = 'Xgboost Detection'

    @staticmethod
    def _get_classifier():
        return xgb.XGBClassifier(
            max_depth=6,
            n_estimators=500,
        )
        
def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    syn_path = f'synthetic/{dataname}/{model}.csv'
    real_path = f'synthetic/{dataname}/test.csv'

    data_dir = f'data/{dataname}' 
    print(syn_path)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    real_data = pd.read_csv(real_path)
    syn_data = (
        pd.read_csv(syn_path)
        .sample(n=real_data.shape[0], replace=False)
        .reset_index(drop=True)
    )
    save_dir = f'eval/detection/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    # qual_report.generate(new_real_data, new_syn_data, metadata)

    score_logistic = LogisticDetection.compute(
        real_data=new_real_data,
        synthetic_data=new_syn_data,
        metadata=metadata
    )
    score_xgboost = XgboostDetection.compute(
        real_data=new_real_data,
        synthetic_data=new_syn_data,
        metadata=metadata
    )
    overall_scores = {
        "dataname": dataname,
        "model": model,
        "logistic": score_logistic,
        "xgboost": score_xgboost
    }
    save_path = f'{save_dir}/{model}.json'
    print(f'{dataname}, {model}, Logistic: {score_logistic:.6f},  Xgboost: {score_xgboost:.6f}')
    print('Saving scores to ', save_path)

    with open(save_path, "w") as json_file:
        json.dump(overall_scores, json_file)