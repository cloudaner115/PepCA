# -*- coding: utf-8 -*-

from model.models import PepCA
from model.models import PeptideComplexes, PepCA_dataset
from time import time
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import datetime
import os
import random
import pickle
from trainer import *
from sklearn.metrics import roc_auc_score
device = "cuda" if torch.cuda.is_available() else "cpu"
with open('./data/protein.pkl', 'rb') as f:
    protein_data = pickle.load(f)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    parser.add_argument("-d", dest="dataset", required=False, type=str, default="pepnn",
                        help="Which dataset to train on, pepnn, pepbind, bitenet or interpep")
    
    args = parser.parse_args()
    
    if args.dataset != "pepnn" and args.dataset != "pepbind" and args.dataset != "interpep" and args.dataset != "bitenet":
        raise ValueError("-d must be set to pepnn, pepbind, bitenet or interpep")


    if not os.path.exists(f'./out_model/{args.dataset}/'):
        os.makedirs(f'./out_model/{args.dataset}/')
    idx = 0
    idxs = []
    for file in os.listdir(f'./out_model/{args.dataset}/'):
        if 'ipy' in file:
            continue
        idxs.append(int(file.split('_')[1]))
        
    if idxs == []:
        idx = 0
    else:
        idx = max(idxs) + 1
    n_layers = 6, d_model = 64, n_head = 6, d_k = 64, d_v = 128, d_inner = 64
    output_file = f'./out_model/{args.dataset}/{args.dataset}_{idx}_{n_layers}_{d_model}_{n_head}_{d_k}_{d_v}_{d_inner}'
    
    model = PepCA(n_layers, d_model, n_head, d_k, d_v, d_inner, dropout=0.1)
    model = model.to(device)

    if args.dataset == "interpep" or args.dataset == "pepbind" or args.dataset == 'bitenet':
        train(model, 15, output_file, args.dataset,protein_data)
    elif args.dataset == "pepnn":
        train_pepnn(model, 30, output_file,protein_data)