from model.models import PepCA
from model.models import PeptideComplexes, PepCA_dataset
from Bio import SeqIO
from Bio.PDB import Polypeptide
from time import time
import numpy as np
import torch.nn as nn
import torch
import argparse
import torch.nn.functional as F
import datetime
import warnings
import esm
warnings.filterwarnings('ignore')
import os
import pickle
def aa2tensor(seq):
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    aa2tensor_dict = {}
    for i,a in enumerate(aa):
        aa2tensor_dict[a] = i
    output = []
    for a in seq:
        output.append(aa2tensor_dict[a])   
    return torch.tensor(output)
    
def esm2tensor(prot_seq,esm_model):
    all_labels, all_strs, all_tokens = batch_converter([('1',prot_seq)])
    batch_size = 1
    num = len(all_tokens) // batch_size + 1
    sequence_representations = {}
    i=0
    batch_tokens = all_tokens[i*batch_size:(i+1)*batch_size]
    batch_lens = (batch_tokens != 1).sum(1)
    batch_tokens = batch_tokens.to(device)
    batch_str = all_strs[i*batch_size:(i+1)*batch_size]
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)    
    token_representations = results["representations"][33]

    for j, tokens_len in enumerate(batch_lens):
        res = token_representations[j, 1 : tokens_len - 1].cpu()
    return res

def predict_binding_site(pep_seq,prot_seq,model,esm_model):
    with torch.no_grad():
        
        pep_sequence = aa2tensor(pep_seq).to(device).unsqueeze(0)
 
        prot_mix = esm2tensor(prot_seq,esm_model)

        prot_mix = prot_mix.float().unsqueeze(0).to(device)

        outputs_nodes = model(pep_sequence, prot_mix)
        outputs = torch.exp(outputs_nodes.squeeze(0)[:,1]).detach().cpu()
    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

   
    parser.add_argument("-prot", dest="input_protein_file", required=False, type=str,
                        help="Fasta file with protein sequence")
    
    parser.add_argument("-pep", dest="input_peptide_file", required=False, type=str,
                        help="Fasta file with peptide sequence")

    args = parser.parse_args()
    
    prot_records = SeqIO.parse(args.input_protein_file, format="fasta")
    
    prot_seq = ' '.join(list(prot_records)[0].seq)
    
    pep_records = SeqIO.parse(args.input_peptide_file, format="fasta")

    pep_seq = str(list(pep_records)[0].seq).replace("X", "")    
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()  # disables dropout for deterministic results
    
    model = PepCA(6, 64, 6, 
                  64, 128, 64,return_attention=False, dropout=0.1)
    model.load_state_dict(torch.load('out_model/pepnn_final.pth'))
    model = model.to(device)
    model = model.eval()
    
    output = predict_binding_site(pep_seq,prot_seq,model,esm_model)    
    print(f'Binding site in this protein is {output}')

    