from torch import nn
from .modules import *

class MultiModalLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        
        super().__init__()
        
        self.peptide_attention_layer = MultiHeadAttention(n_head, d_model,
                                                                d_k, d_v)
        
        self.protein_attention_layer = MultiHeadAttention(n_head, d_model,
                                                               d_k, d_v)
        
        self.multi_co_attention_layer = MultiHeadCoAttention(n_head, d_model,
                                                                           d_k, d_v)
        
        self.dense_1 = nn.Linear(d_model*2, d_model)
        self.dense_2 = nn.Linear(d_model*2, d_model)
        
        self.ffn_seq = FFN(d_model, d_inner)
        
        self.ffn_protein = FFN(d_model, d_inner)

    def forward(self, peptide_seq_enc, protein_seq_enc):
        
        prot_enc, prot_attention = self.protein_attention_layer(protein_seq_enc, protein_seq_enc, protein_seq_enc)
        pep_enc, pep_attention = self.peptide_attention_layer(peptide_seq_enc, peptide_seq_enc, peptide_seq_enc)
        prot_enc, pep_enc, prot_pep_attention, pep_prot_attention = self.multi_co_attention_layer(prot_enc,pep_enc)
        prot_enc = self.ffn_protein(prot_enc)

        pep_enc = self.ffn_seq(pep_enc)
    
        prot_enc = prot_enc + self.dense_1(torch.cat([prot_enc, protein_seq_enc], dim=-1))
        pep_enc = pep_enc + self.dense_2(torch.cat([pep_enc, peptide_seq_enc],dim=-1))
        
        return prot_enc, pep_enc, prot_attention, pep_attention, prot_pep_attention, pep_prot_attention

