import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from importlib.machinery import SourceFileLoader

from src.phet import PhET
from src.datasets import MHMDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="Path to the model checkpoint.", type=str, required=True)
    parser.add_argument("--data_path", help="Path to the data split.", type=str, required=True)
    parser.add_argument("--out_dir", help="Path to the save the predition", type=str, required=True)
    parser.add_argument("--config", help="Path to the config file.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for data loading", default=False)
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision('high')
    torch.set_warn_always(False)
    
    # Parse arguments:
    args = parse_args()
    cfg = SourceFileLoader("config", args.config).load_module()
    ckpt_path = args.ckpt_path
    data_path = args.data_path
    out_dir = args.out_dir
    nb_workers = args.nb_workers
    pin_memory = args.pin_memory
    
    # Load tokenizer:
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    tokenizer = checkpoint['hyper_parameters']['tokenizer']
    
    # Load dataset:
    df = pd.read_csv(data_path)
    df = df[['eid'] + cfg.num_features + cfg.cat_features + cfg.diseases]
    dataset = MHMDataset(df, tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size = cfg.batch_size,
        shuffle = False,
        num_workers = nb_workers,
        pin_memory = pin_memory
    )
    
    # Load model:
    model = PhET.from_lightning_checkpoint(ckpt_path).eval()
    
    # Predict:
    outputs = []
    for batch in tqdm(data_loader, desc='Prediction'):
        scores = model.predict(
            value_ids = batch['pred_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            bool_traits = tokenizer.boolean_traits
        )
        
        outputs.append({
            'eids': batch['eid'],
            'scores': scores,
            'labels': batch['pred_labels'],
            'phenotype_ids': batch['phenotype_ids']
        })
    
    y = defaultdict(lambda: defaultdict(list))
    for x in outputs:
        eids = x['eids']
        for trait, y_scores in x['scores'].items():
            p_id = tokenizer.get_phenotype_id(trait)
            info = tokenizer.get_boolean_trait_info(p_id)
            true_id = info['true_id']
            false_id = info['false_id']
            
            phenotype_ids = x['phenotype_ids']
            labels = x['labels'][phenotype_ids == p_id]
            y_true = labels.clone()
            y_true[labels == false_id] = 0
            y_true[labels == true_id] = 1
            y[trait]['eids'].append(eids)
            y[trait]['y_scores'].append(y_scores)
            y[trait]['y_true'].append(y_true)

    # Save output in JSON file:
    os.makedirs(out_dir, exist_ok=True)
    for trait, value in y.items():
        eids = torch.cat(value['eids'])
        y_true = torch.cat(value['y_true'])
        y_scores = torch.cat(value['y_scores'])
            
        trait_name = trait.lower().replace(" ", "-")
        filename = f"rs_{trait_name}"
        filename += ".json"
        full_path = os.path.join(out_dir, filename)

        with open(full_path, "w") as f:
            f.write(json.dumps({
                "eids": eids.tolist(),
                "y_true": y_true.tolist(),
                "y_scores": y_scores.tolist()
            }))
    print(f"Scores saved successfully in {out_dir}")


if __name__ == '__main__':
    main()