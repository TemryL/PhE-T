import torch
import pandas as pd
import lightning as L
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader
from .tokenizer import PhenotypeTokenizer


class MHMDataset(Dataset):
    def __init__(self, df, tokenizer, mhm_probability=0.15):
        self.eids = []
        self.data = []
        self.tokenizer = tokenizer
        self.mhm_probability = mhm_probability
        self._tokenize(df)
    
    def _tokenize(self, df):
        for _, row in tqdm(df.iterrows(), desc='Tokenize data', total=len(df)):
            row = row.to_dict()
            if 'eid' in row:
                eid = row.pop('eid')
                self.eids.append(eid)
            self.data.append(self.tokenizer.encode(row))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        # Create value_ids, phenotype_ids and labels for MHM and boolean trait prediction
        phenotype_ids = torch.tensor(self.data[idx]['phenotype_ids'])
        hm_value_ids = torch.tensor(self.data[idx]['value_ids'])
        pred_value_ids = torch.tensor(self.data[idx]['value_ids'])
        hm_labels = hm_value_ids.clone()
        pred_labels = pred_value_ids.clone()
        
        # Create probability matrix for masking
        prob_matrix = torch.full(hm_labels.shape, self.mhm_probability)
                
        # Create mask for health modeling tokens prediction
        hm_mask = torch.bernoulli(prob_matrix).bool()
        hm_labels[~hm_mask] = -100    # only compute loss on masked tokens
        hm_value_ids[hm_mask] = self.tokenizer.mask_token_id
        
        # Creat mask for boolean trait prediction
        bool_trait_ids = torch.tensor(list(self.tokenizer.boolean_traits.keys()))
        pred_mask = (phenotype_ids[..., None] == bool_trait_ids).any(dim=-1)
        pred_value_ids[pred_mask] = self.tokenizer.mask_token_id
        
        return {
            'phenotype_ids': phenotype_ids,
            'hm_value_ids': hm_value_ids,
            'hm_labels': hm_labels,
            'pred_value_ids': pred_value_ids,
            'pred_labels': pred_labels,
            'eid': self.eids[idx]
        }


class MHMDataModule(L.LightningDataModule):
    def __init__(self, train_data: str, val_data: str, test_data: str,
                 num_features: List[str], cat_features: List[str],
                 batch_size: int = 32, n_workers: int = 4,
                 n_bins: int = 100, binning: str = 'uniform',
                 mhm_probability: float = 0.15,
                 pin_memory: bool = True):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_features = num_features
        self.cat_features = cat_features
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_bins = n_bins
        self.binning = binning
        self.mhm_probability = mhm_probability
        self.pin_memory = pin_memory
        self.tokenizer = PhenotypeTokenizer(n_bins=n_bins, binning=binning)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._has_setup = False

    def setup(self, stage=None):
        if not self._has_setup and (stage == 'fit' or stage is None):
            train_df = pd.read_csv(self.train_data)
            val_df = pd.read_csv(self.val_data)
            train_df = train_df[self.num_features + self.cat_features]
            val_df = val_df[self.num_features + self.cat_features]
            self.tokenizer.fit(pd.concat([train_df, val_df]), self.num_features, self.cat_features)
            
            self.train_dataset = MHMDataset(train_df, self.tokenizer, mhm_probability=self.mhm_probability)
            self.val_dataset = MHMDataset(val_df, self.tokenizer, mhm_probability=self.mhm_probability)
            self._has_setup = True
        
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(self.test_data)
            test_df = test_df[self.num_features + self.cat_features]
            self.test_dataset = MHMDataset(test_df, self.tokenizer, mhm_probability=self.mhm_probability)
        
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