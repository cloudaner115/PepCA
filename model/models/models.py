import numpy as np
import torch
import torch.nn as nn
from .layers import *
from .modules import *



class RepeatedModule(nn.Module):
    
    def __init__(self, n_layers, d_model,
                 n_head, d_k, d_v, d_inner, dropout=0.1):
        
        super().__init__()
        
        self.linear = nn.Linear(1280, d_model)
        self.sequence_embedding = nn.Embedding(20, d_model)
        self.d_model = d_model 
        
        self.multimodal_layer_stack = nn.ModuleList([
                MultiModalLayer(d_model,  d_inner,  n_head, d_k, d_v) 
                for _ in range(n_layers)])
    
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
       
    
    def _positional_embedding(self, number,device):
        
        result = torch.exp(torch.arange(0, self.d_model,2,dtype=torch.float32)*-1*(np.log(10000)/self.d_model))
        
        numbers = torch.arange(0, number, dtype=torch.float32)
        
        numbers = numbers.unsqueeze(0)
        
        numbers = numbers.unsqueeze(2)
       
        result = numbers*result
        
        result = torch.cat((torch.sin(result), torch.cos(result)),2)
        result = result.to(device)
       
        return result
    
    def forward(self, peptide_sequence, protein_sequence):
        
        
        peptide_attention_list = []
        
        prot_attention_list = []
        
        prot_pep_attention_list = []
        
        pep_prot_attention_list = []
        
        pep_enc = self.sequence_embedding(peptide_sequence)
        
        pep_enc += self._positional_embedding(peptide_sequence.shape[1],peptide_sequence.device)
        pep_enc = self.dropout(pep_enc)

        prot_enc = self.dropout_2(self.linear(protein_sequence))

        for reciprocal_layer in self.multimodal_layer_stack:
            
            prot_enc, pep_enc, prot_attention, peptide_attention, prot_pep_attention, pep_prot_attention =\
                reciprocal_layer(pep_enc, prot_enc)
            
            peptide_attention_list.append(peptide_attention)
            
            prot_attention_list.append(prot_attention)
            
            prot_pep_attention_list.append(prot_pep_attention)
            
            pep_prot_attention_list.append(pep_prot_attention)
            
        
        
        return prot_enc, pep_enc, peptide_attention_list, prot_attention_list,\
            prot_pep_attention_list, pep_prot_attention_list
    

class PepCA(nn.Module):
    
    def __init__(self, n_layers, d_model, n_head,
                 d_k, d_v, d_inner, return_attention=False, dropout=0.2):
        
        super().__init__()
        self.repeated_module = RepeatedModule(n_layers, d_model,
                               n_head, d_k, d_v, d_inner, dropout=dropout)
        
        self.final_attention_layer = MultiHeadAttention(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)
        
        self.final_ffn = FFN(d_model, d_inner, dropout=dropout) 
        self.output_projection_prot = nn.Linear(d_model, 2)     
        self.softmax_prot =nn.LogSoftmax(dim=-1)
   
                
        self.return_attention = return_attention
        
    def forward(self, peptide_sequence, protein_sequence):
        
      
        prot_enc, pep_enc, peptide_attention_list, prot_attention_list,\
            prot_pep_attention_list, pep_prot_attention_list = self.repeated_module(peptide_sequence,
                                                                                    protein_sequence)
            
        prot_enc, final_prot_pep_attention  = self.final_attention_layer(prot_enc, pep_enc, pep_enc)
        
        prot_enc = self.final_ffn(prot_enc)

        prot_enc = self.softmax_prot(self.output_projection_prot(prot_enc))

        if not self.return_attention:
            return prot_enc
        else:
            return prot_enc, peptide_attention_list, prot_attention_list,\
            prot_pep_attention_list, pep_prot_attention_list, final_prot_pep_attention
        
