import os
import pandas as pd
import torch

import argparse
import warnings
import json
import time
from synthcity.utils.serialization import load_from_file

import json

warnings.filterwarnings('ignore')

def main(args):
    dataname = args.dataname
    device = args.device
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    syn_model = load_from_file(f'{ckpt_dir}/model.pkl')
    
    start_time = time.time()    
    syn_df = syn_model.generate(count=info['train_num']).dataframe()

    syn_df.to_csv(save_path, index = False)

    end_time = time.time()  
    print(f'Sampling time = {end_time - start_time}')
    print('Saving sampled data to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training ')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'