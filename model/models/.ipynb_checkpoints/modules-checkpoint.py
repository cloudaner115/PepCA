from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v    
        self.W_Q = nn.Linear(d_model, n_head*d_k)
        self.W_K = nn.Linear(d_model, n_head*d_k)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
 

    def forward(self, q, k, v):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()

        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)
        attention = torch.matmul(Q, K)
           
        attention = attention /np.sqrt(self.d_k)

        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, V)           
        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])            
        output = self.W_O(output)      
        output = self.dropout(output)
        
        output = self.layer_norm(output + q)
        
        return output, attention
        
class MultiHeadCoAttention(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        
        self.Linear_pro_1 = nn.Linear(d_model, n_head*d_k)
        self.Linear_pro_2 = nn.Linear(d_model, n_head*d_v)
        self.Linear_pep_1 = nn.Linear(d_model, n_head*d_k)
        self.Linear_pep_2 = nn.Linear(d_model, n_head*d_v)
        self.Linear_output_prot = nn.Linear(n_head*d_v, d_model)
        self.Linear_output_pep = nn.Linear(n_head*d_v, d_model)
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, prot_mix, pep_seq):
        
        batch, len_prot, _ = prot_mix.size()
        batch, len_pep, _ = pep_seq.size()
         
        Prot_1 = self.Linear_pro_1(prot_mix).view([batch, len_prot, self.n_head, self.d_k])
        Prot_2 = self.Linear_pro_2(prot_mix).view([batch, len_prot, self.n_head, self.d_v])
        Pep_1 = self.Linear_pep_1(pep_seq).view([batch, len_pep, self.n_head, self.d_k])
        Pep_2 = self.Linear_pep_2(pep_seq).view([batch, len_pep, self.n_head, self.d_v])
        
        Prot_1 = Prot_1.transpose(1, 2)
        Prot_2 = Prot_2.transpose(1,2) 
        Pep_1 = Pep_1.transpose(1, 2).transpose(2, 3)
        Pep_2 = Pep_2.transpose(1, 2)
   
        similarity_matrix = torch.matmul(Prot_1, Pep_1)

        attention_1 = F.softmax(similarity_matrix /np.sqrt(self.d_k), dim=-1)
        attention_2 = F.softmax(similarity_matrix.transpose(-2, -1) / np.sqrt(self.d_k), dim=-1)
        
        output_prot = torch.matmul(attention_1, Pep_2)
        output_pep = torch.matmul(attention_2, Prot_2)
        output_prot = output_prot.transpose(1, 2).reshape([batch, len_prot, self.d_v*self.n_head])
        output_pep = output_pep.transpose(1, 2).reshape([batch, len_pep, self.d_v*self.n_head])
        output_prot = self.dropout_1(self.Linear_output_prot(output_prot))
        output_pep = self.dropout_2(self.Linear_output_pep(output_pep))
        output_prot = self.layer_norm_1(output_prot + prot_mix)
        output_pep = self.layer_norm_2(output_pep + pep_seq)
        return output_prot, output_pep, attention_1, attention_2
        
    
class FFN(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        
        self.layer_1 = nn.Conv1d(d_in, d_hid,1)
        self.layer_2 = nn.Conv1d(d_hid, d_in,1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_in)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        residual = x 
        output = self.layer_1(x.transpose(1, 2))
        
        output = self.relu(output)
        
        output = self.layer_2(output)
        
        output = self.dropout(output)
        
        output = self.layer_norm(output.transpose(1, 2)+residual)
        
        return output

