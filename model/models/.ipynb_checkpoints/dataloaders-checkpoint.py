# -*- coding: utf-8 -*-


from torch.utils.data import Dataset
from Bio.PDB import Polypeptide
import numpy as np
import torch
import pandas as pd

    
class PepCA_dataset(Dataset):
    
    def __init__(self, mode,data_set,df_dir = './data/'):
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        aa2tensor_dict = {}
        for i,a in enumerate(aa):
            aa2tensor_dict[a] = i
        self.aa2tensor_dict = aa2tensor_dict
        self.mode = mode
        self.data_set = data_set
        self.df = pd.read_csv(df_dir+data_set+'_'+mode+'.csv')

    def aa2tensor(self,seq):
        output = []
        for a in seq:
            output.append(self.aa2tensor_dict[a])
        return torch.tensor(output) 
        
    def __getitem__(self, index):
        pdb_id,prot_sequence, pep_seq, target= self.df.iloc[index]

        pep_sequence = self.aa2tensor(pep_seq)
        
        number_list = [int(char) for char in target]
        target = torch.tensor(number_list)

        return pep_sequence, prot_sequence, target
            
    def __len__(self):
        return len(self.df)
    
class PeptideComplexes(Dataset):
    
    def __init__(self, mode,df_dir = './data/'):
        data_set = 'pepnn'
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        aa2tensor_dict = {}
        for i,a in enumerate(aa):
            aa2tensor_dict[a] = i
        self.aa2tensor_dict = aa2tensor_dict
        self.mode = mode
        self.data_set = data_set
        self.df = pd.read_csv(df_dir+data_set+'_'+mode+'.csv')

    def aa2tensor(self,seq):
        output = []
        for a in seq:
            output.append(self.aa2tensor_dict[a])
        return torch.tensor(output) 
        
    def __getitem__(self, index):
        pdb_id,prot_sequence, pep_seq, target,weight= self.df.iloc[index]

        pep_sequence = self.aa2tensor(pep_seq)
        
        number_list = [int(char) for char in target]
        target = torch.tensor(number_list)

        return pep_sequence, prot_sequence, target,torch.tensor(weight)
            
    def __len__(self):
        return len(self.df)