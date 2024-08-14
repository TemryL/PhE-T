import torch
import pandas as pd
import lightning as L
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .tokenizer import PhenotypeTokenizer


class MHMDataset(Dataset):
    def __init__(self, df, tokenizer, mlm_probability=0.15):
        self.data = []
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self._tokenize(df)
    
    def _tokenize(self, df):
        for _, row in tqdm(df.iterrows(), desc='Tokenize data', total=len(df)):
            row = row.to_dict()
            self.data.append(self.tokenizer.encode(row))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        # Create value_ids, phenotype_ids and labels for MHM
        value_ids = torch.tensor(self.data[idx]['value_ids'])
        phenotype_ids = torch.tensor(self.data[idx]['phenotype_ids'])
        labels = value_ids.clone()
        
        # Create probability matrix for masking
        prob_matrix = torch.full(labels.shape, self.mlm_probability)
                
        # Create mask for tokens to be predicted
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100    # only compute loss on masked tokens
        
        # Replace masked input tokens with tokenizer.mask_token_id
        value_ids[masked_indices] = self.tokenizer.mask_token_id
        
        return {
            'value_ids': value_ids,
            'phenotype_ids': phenotype_ids,
            'labels': labels
        }


class MHMDataModule(L.LightningDataModule):
    def __init__(self, train_data: str, test_data: str,
                 num_features: List[str], cat_features: List[str],
                 batch_size: int = 32, n_workers: int = 4, 
                 train_val_split: float = 0.95,
                 n_bins: int = 100,
                 mlm_probability: float = 0.15,
                 pin_memory: bool = True):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.num_features = num_features
        self.cat_features = cat_features
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.train_val_split = train_val_split
        self.n_bins = n_bins
        self.mlm_probability = mlm_probability
        self.pin_memory = pin_memory
        self.tokenizer = PhenotypeTokenizer(n_bins=n_bins)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.train_data).drop('eid', axis=1)
            self.tokenizer.fit(train_df, self.num_features, self.cat_features)
            
            train_df, val_df = train_test_split(train_df, train_size=self.train_val_split, random_state=42)
            train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
            
            self.train_dataset = MHMDataset(train_df, self.tokenizer, mlm_probability=self.mlm_probability)
            self.val_dataset = MHMDataset(val_df, self.tokenizer, mlm_probability=self.mlm_probability)
        
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(self.test_data).drop('eid', axis=1)
            self.test_dataset = MHMDataset(test_df, self.tokenizer, mlm_probability=self.mlm_probability)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )