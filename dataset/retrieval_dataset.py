import torch
from torch.utils.data import Dataset
from .fasta_dataset import FastaDataset

class RetrievalDataset(Dataset):
    def __init__(self, candidate_path, seed_path):
        super(RetrievalDataset, self).__init__()
        self.candidate_path = candidate_path
        self.seed_path = seed_path
        
        self.candidate_dataset = FastaDataset(candidate_path, label=1)
        self.seed_dataset = FastaDataset(seed_path, label=0)
        
    def __len__(self):
        return len(self.candidate_dataset) + len(self.seed_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.candidate_dataset):
            return self.candidate_dataset[idx]
        else:
            return self.seed_dataset[idx - len(self.candidate_dataset)]
    
    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        return sequences, torch.tensor(labels)
