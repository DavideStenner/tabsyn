import os
import torch
import time
import pandas as pd

import argparse
import warnings
import json
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file

warnings.filterwarnings('ignore')

def main(args):
    
    dataname = args.dataname
    device = args.device

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    X_train = pd.read_csv(
        os.path.join(dataset_dir, 'train.csv')
    )
    target_col_name = X_train.columns[info['target_col_idx'][0]]
    loader = GenericDataLoader(
        X_train,
        target_column=target_col_name,
    )
    syn_model = Plugins().get("ctgan")

    syn_model.fit(loader)
    start_time = time.time()

    save_to_file(f'{ckpt_dir}/model.pkl', syn_model)
    end_time = time.time()

    print(f'Training time: {end_time - start_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GOGGLE')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'