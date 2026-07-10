import random
import torch
import numpy as np
from torch.utils.data import Dataset
from .fasta_dataset import FastaDataset

class TrainingDataset(Dataset):
    def __init__(self, positive_path, negative_path, dynamic_negative_sampling=False, neg_pos_ratio=1):
        super(TrainingDataset, self).__init__()
        self.dynamic_negative_sampling = dynamic_negative_sampling
        self.neg_pos_ratio = neg_pos_ratio
        
        
        self.positive_dataset = FastaDataset(positive_path, label=1)
        self.all_negative_dataset = FastaDataset(negative_path, label=0)
        
        self.num_pos = len(self.positive_dataset)
        self.num_neg_total = len(self.all_negative_dataset)

       
        if not self.dynamic_negative_sampling:
            
            self.fixed_negatives = self.all_negative_dataset
        else:
           
            self.fixed_negatives = []

    def __len__(self):
        if self.dynamic_negative_sampling:
           
            return int(self.num_pos * (1 + self.neg_pos_ratio))
        else:
          
            return self.num_pos + len(self.fixed_negatives)
    
    def __getitem__(self, idx):
       
        if idx < self.num_pos:
            return self.positive_dataset[idx]
        
        
        else:
            if self.dynamic_negative_sampling:
               
                rand_idx = random.randint(0, self.num_neg_total - 1)
                return self.all_negative_dataset[rand_idx]
            else:
               
                neg_idx = idx - self.num_pos
                
                if neg_idx >= len(self.fixed_negatives):
                    neg_idx = neg_idx % len(self.fixed_negatives)
                    
                return self.fixed_negatives[neg_idx]
    
    def collate_fn(self, batch):
        sequences_and_labels = list(batch)
        sequences = [item[0] for item in sequences_and_labels]
        labels = [item[1] for item in sequences_and_labels]
        return sequences, torch.tensor(labels)