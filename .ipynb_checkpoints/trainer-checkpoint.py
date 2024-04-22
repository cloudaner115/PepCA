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
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
device = 'cuda' 
def train_pepnn(model, epochs, output_file,protein_data):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001)
    loader = torch.utils.data.DataLoader(PeptideComplexes(mode="train"),
                                         batch_size=1, num_workers=16,
                                         shuffle=True)
    validation_loader = torch.utils.data.DataLoader(PeptideComplexes(mode="test"),
                                                    batch_size=1, num_workers=16,
                                                    shuffle=True)
    binding_weights = torch.FloatTensor(np.load("./model/params/binding_weights.npy")).to(device)
    start_time = time()
    validation_losses = []
    iters = len(loader)
    iters_validation = len(validation_loader)
    all_acc = []
    pbar = tqdm(range(0, epochs))
    for e in pbar:
        pbar.set_description(f"training epoch {e}") 
        model.train()
        run_loss = 0
        accuracy = 0
        
        for i, (pep_sequence, prot_seq, target, weight) in enumerate(loader):
            pep_sequence = pep_sequence.to(device)
            target = target.to(device)
            weight = weight.to(device)
            prot_mix = protein_data[prot_seq[0]]
            prot_mix = prot_mix.unsqueeze(0)
            prot_mix = prot_mix.to(device)
            optimizer.zero_grad()
            outputs_nodes = model(pep_sequence, prot_mix)
            
            loss = F.nll_loss(outputs_nodes.transpose(-1,-2), target,
                              ignore_index=-100, weight=binding_weights)

            loss = loss*weight
            run_loss += loss.item()
            loss.backward()
            optimizer.step()
            predict = torch.argmax(outputs_nodes, dim=-1)
            predict = predict.cpu().detach().numpy()
            actual = target.cpu().detach().numpy()  
            correct = np.sum(predict == actual)
            total = len(actual[0])
            accuracy +=  correct/total
                  

            if i == iters - 1:
                model.eval()
                run_loss = 0
                accuracy = 0
                all_outputs = []
                all_target = []
                with torch.no_grad():
                    for i_val, (pep_sequence, prot_seq, target, weight) in enumerate(validation_loader):
                        pep_sequence = pep_sequence.to(device)
                        target = target.to(device)
                        weight = weight.to(device)
                        prot_mix = protein_data[prot_seq[0]]
                        prot_mix = prot_mix.unsqueeze(0)
                        prot_mix = prot_mix.to(device)

                        weight = weight.to(device)
                        outputs_nodes = model(pep_sequence, prot_mix)
   
                        loss = F.nll_loss(outputs_nodes.transpose(-1,-2), target,
                              ignore_index=-100, weight=binding_weights)

                        outputs = outputs_nodes[:,0]
                        all_outputs.append(outputs)
                        all_target.append(target)
                        loss = loss*weight                       
                        run_loss += loss.item()              
                        predict = torch.argmax(outputs_nodes, dim=-1)
                        predict = predict.cpu().detach().numpy()
                        actual = target.cpu().detach().numpy()                        
                        correct = np.sum(predict == actual)
                        total = len(actual[0])
         
                        accuracy += correct/total                                      
                        del loss

                validation_losses.append(run_loss / iters_validation)
                all_acc.append(accuracy / (iters_validation))
     
                pbar.set_postfix({'loss' : '{:.5f}'.format(validation_losses[-1])})

                        
                if validation_losses[-1]  == min(validation_losses):
                        print("Saving model with new minimum validation loss")
                        torch.save(model.state_dict(),output_file+'_loss.pth')  
                        print("Saved model successfully!")

            
                if all_acc[-1]  == max(all_acc):
                        print("Saving model with new max acc")
                        torch.save(model.state_dict(),output_file+'_acc.pth')  
                        print("Saved model successfully!")                              
                torch.save(model.state_dict(),output_file+'_final.pth')      
                run_loss = 0
                accuracy = 0
            
                model.train() 

                
def train(model, epochs, output_file, dataset,protein_data):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
    
    if dataset == "pepbind":
        binding_weights = torch.FloatTensor(np.load("./model/params/binding_weights_pepbind.npy")).to(device)
    elif dataset == "interpep":
        binding_weights = torch.FloatTensor(np.load("./model/params/binding_weights_interpep.npy")).to(device)
    elif dataset == 'bitenet':
        binding_weights = torch.FloatTensor(np.load("./model/params/binding_weights_bitenet.npy")).to(device)
    loader = torch.utils.data.DataLoader(PepCA_dataset(data_set=dataset,mode="train"),
                                         batch_size=1, num_workers=16,
                                         shuffle=True)
    
    validation_loader = torch.utils.data.DataLoader(PepCA_dataset(data_set=dataset ,mode="val"),
                                                    batch_size=1, num_workers=16,
                                                    shuffle=True)        
    start_time = time()
    validation_losses = []
    all_acc = []
    iters = len(loader)    
    iters_validation = len(validation_loader)
    pbar = tqdm(range(0, epochs))
    for e in pbar:
        pbar.set_description(f"training step {e}") 
        model.train()
        run_loss = 0
        accuracy = 0
        pbar = tqdm(loader)
        for i, (pep_sequence, prot_seq, target) in enumerate(pbar):
            
            pep_sequence = pep_sequence.to(device)
            target = target.to(device)
            prot_mix = protein_data[prot_seq[0]]
            prot_mix = prot_mix.unsqueeze(0)
            prot_mix = prot_mix.to(device)           
            optimizer.zero_grad()          
            outputs_nodes = model(pep_sequence, prot_mix)
            loss = F.nll_loss(outputs_nodes.transpose(-1,-2), target,
                              ignore_index=-100, weight=binding_weights)

            run_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
            predict = torch.argmax(outputs_nodes, dim=-1)
          
            
            predict = predict.cpu().detach().numpy()
            actual = target.cpu().detach().numpy()
            
            
            correct = np.sum(predict == actual)
            total = len(actual[0])
            
            
            accuracy +=  correct/total
                            
            if i == iters - 1:
                model.eval()
                run_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for i_val, (pep_sequence, prot_seq, target) in enumerate(validation_loader):
                        
                        pep_sequence = pep_sequence.to(device)
                        target = target.to(device)
                        prot_mix = protein_data[prot_seq[0]]
                        prot_mix = prot_mix.unsqueeze(0)
                        prot_mix = prot_mix.to(device)                    
                        outputs_nodes = model(pep_sequence, prot_mix)
      
                        loss = F.nll_loss(outputs_nodes.transpose(-1,-2), target,
                              ignore_index=-100, weight=binding_weights)

                        run_loss += loss.item()
                        predict = torch.argmax(outputs_nodes, dim=-1)       
                        predict = predict.cpu().detach().numpy()
                        actual = target.cpu().detach().numpy()
                        correct = np.sum(predict == actual)
                        total = len(actual[0])
                        accuracy += correct/total                        
                    
                        del loss
                
                validation_losses.append(run_loss / iters_validation)
                all_acc.append(accuracy / (iters_validation))
                
                pbar.set_postfix({'loss' : '{:.5f}'.format(validation_losses[-1])})
                if validation_losses[-1]  == min(validation_losses):
                        print("Saving model with new minimum validation loss")
                        torch.save(model.state_dict(),output_file+'_loss.pth')  
                        print("Saved model successfully!")

            
                if all_acc[-1]  == max(all_acc):
                        print("Saving model with new max acc")
                        torch.save(model.state_dict(),output_file+'_acc.pth')  
                        print("Saved model successfully!")                   
                torch.save(model.state_dict(),output_file+'_final.pth')  
                run_loss = 0
                accuracy = 0
            
                model.train() 