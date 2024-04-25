from model.models import PepCA
from model.models import PeptideComplexes, PepCA_dataset
from transformers import BertModel, BertTokenizer, pipeline
from time import time
import numpy as np
import torch.nn as nn
import torch
import argparse
import torch.nn.functional as F
import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from sklearn.metrics import roc_auc_score
from Bio.PDB import Polypeptide
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
device = 'cuda'
with open('./data/protein.pkl', 'rb') as f:
    protein_data = pickle.load(f)
def val_pepnn():
    model = PepCA(6, 64, 6, 
                  64, 128, 64,return_attention=False, dropout=0.1)
    model.load_state_dict(torch.load('out_model/pepnn_final.pth'))
    model = model.to(device)
    model = model.eval()
    loader = torch.utils.data.DataLoader(PeptideComplexes(mode="test"),
                                         batch_size=1, num_workers=16,
                                         shuffle=True)
    all_targets = []
    all_auc = []
    all_outputs = []
    prot_seqs = []
    with torch.no_grad():
        for value_val, (pep_sequence, prot_seq, target,weight) in enumerate(loader):
            pep_sequence = pep_sequence.to(device)
            target = target.to(device)
            #weight = weight.to(device)
            
            prot_seq = prot_seq[0].replace(" ", "")
            prot_seqs.append(prot_seq)
    
            prot_mix = protein_data[prot_seq]
            prot_mix = prot_mix.float().unsqueeze(0).to(device)
    
            outputs_nodes = model(pep_sequence, prot_mix)
            outputs = torch.exp(outputs_nodes.squeeze(0)[:,1]).detach().cpu()
            target = target.detach().cpu()
            try:
                all_auc.append(roc_auc_score(target.numpy().squeeze(), outputs.numpy()))
            except:
                continue
            all_targets.append(target.squeeze())
            all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    all_targets = all_targets.detach().cpu()
    all_outputs = all_outputs.detach().cpu()
    auroc = roc_auc_score(all_targets, all_outputs)
    return auroc

def val(dataset):
    model = PepCA(6, 64, 6, 
                  64, 128, 64,return_attention=False, dropout=0.1)
    model.load_state_dict(torch.load(f'out_model/{dataset}_final.pth'))
    model = model.to(device)
    model = model.eval()
    loader = torch.utils.data.DataLoader(PepCA_dataset(data_set = dataset,mode="test"),
                                         batch_size=1, num_workers=16,
                                         shuffle=True)
    all_targets = []
    all_auc = []
    all_outputs = []
    prot_seqs = []
    with torch.no_grad():
        for value_val, (pep_sequence, prot_seq, target) in enumerate(loader):
            pep_sequence = pep_sequence.to(device)
            target = target.to(device)     
            prot_seq = prot_seq[0].replace(" ", "")
            prot_seqs.append(prot_seq)
    
            prot_mix = protein_data[prot_seq]
            prot_mix = prot_mix.float().unsqueeze(0).to(device)
    
            outputs_nodes = model(pep_sequence, prot_mix)
            outputs = torch.exp(outputs_nodes.squeeze(0)[:,1]).detach().cpu()
            target = target.detach().cpu()
            try:
                all_auc.append(roc_auc_score(target.numpy().squeeze(), outputs.numpy()))
            except:
                continue
            all_targets.append(target.squeeze())
            all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    all_targets = all_targets.detach().cpu()
    all_outputs = all_outputs.detach().cpu()
    auroc = roc_auc_score(all_targets, all_outputs)
    return auroc
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    parser.add_argument("-d", dest="dataset", required=False, type=str, default="pepnn",
                        help="Which dataset to train on, pepnn, pepbind, bitenet or interpep")
    
    args = parser.parse_args()


    if args.dataset == "interpep" or args.dataset == "pepbind" or args.dataset == 'bitenet':
        auroc = val(args.dataset)
        print(f'{args.dataset} auroc is {auroc}')
    elif args.dataset == "pepnn":
        auroc = val_pepnn()
        print(f'{args.dataset} auroc is {auroc}')