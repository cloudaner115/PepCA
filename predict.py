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

model = PepCA(6, 64, 6, 
              64, 128, 64,return_attention=False, dropout=0.1)
model.load_state_dict(torch.load('out_model/pepnn_final.pth'))
model = model.to(device)
model = model.eval()
df = pd.read_csv('./PepPI.csv')
all_targets = []
all_auc = []
all_outputs = []
prot_seqs = []
with torch.no_grad():
    for value_val, (pep_sequence, prot_seq, target) in enumerate(df.iloc):
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
        all_targets.append(target)

df['target'] = all_targets
df.to_csv('./PepPI.csv')


        

    