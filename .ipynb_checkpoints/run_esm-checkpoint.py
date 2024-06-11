import torch
#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from tqdm import tqdm
import torch
import esm
import numpy as np
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()  # disables dropout for deterministic results

df = pd.read_csv('./protein.csv')
all_seq = df['seq'].tolist()
data = []
for seq in all_seq:
    data.append(('1',seq))

all_labels, all_strs, all_tokens = batch_converter(data)
batch_size = 1
num = len(all_tokens) // batch_size + 1
sequence_representations = {}
for i in tqdm(range(num)):

    batch_tokens = all_tokens[i*batch_size:(i+1)*batch_size]
    batch_lens = (batch_tokens != 1).sum(1)
    batch_tokens = batch_tokens.to(device)
    batch_str = all_strs[i*batch_size:(i+1)*batch_size]
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)    
    token_representations = results["representations"][33]

    for j, tokens_len in enumerate(batch_lens):
        res = token_representations[j, 1 : tokens_len - 1].cpu().numpy()
        sequence_representations[batch_str[j]] = res

import pickle

with open('./protein.pkl', 'wb') as f:
    pickle.dump(sequence_representations, f)