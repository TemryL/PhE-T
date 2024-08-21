import torch
import lightning as L
from collections import defaultdict
from torch.optim import AdamW
from torcheval.metrics.functional import binary_auroc, binary_auprc
from transformers import get_linear_schedule_with_warmup


class MHMTransformer(L.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        adamw_epsilon: float = 1e-8,
        adamw_betas: tuple = (0.9, 0.98),
        warmup_steps: int = 10000,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.tokenizer = tokenizer
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(
            value_ids = batch['hm_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            labels = batch['hm_labels'],
        )
        loss = outputs['loss']
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            value_ids = batch['hm_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            labels = batch['hm_labels'],
        )
        loss = outputs['loss']
        
        scores = self.model.predict(
            value_ids = batch['pred_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            bool_traits = self.tokenizer.boolean_traits
        )
        
        self.validation_step_outputs.append({
            'loss': loss,
            'scores': scores,
            'labels': batch['pred_labels'],
            'phenotype_ids': batch['phenotype_ids']
        })
        return loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.tensor([x['loss'] for x in outputs], device=self.device).mean()
        y = defaultdict(lambda: defaultdict(list))
        for x in outputs:
            for trait, y_pred in x['scores'].items():
                p_id = self.tokenizer.get_phenotype_id(trait)
                info = self.tokenizer.get_boolean_trait_info(p_id)
                true_id = info['true_id']
                false_id = info['false_id']
                
                phenotype_ids = x['phenotype_ids']
                labels = x['labels'][phenotype_ids == p_id]
                y_true = labels.clone()
                y_true[labels == false_id] = 0
                y_true[labels == true_id] = 1
                y[trait]['y_pred'].append(y_pred)
                y[trait]['y_true'].append(y_true)
        
        results = {}
        for trait, value in y.items():
            pred_values = torch.cat(value['y_pred'])
            true_values = torch.cat(value['y_true'])
            
            # Compute AUROC and AUPRC
            auroc = binary_auroc(pred_values, true_values)
            auprc = binary_auprc(pred_values, true_values)
            results[trait] = {"AUROC": auroc, "AUPRC": auprc}

        # Log results
        self.log("val/loss", loss, sync_dist=True)
        for name, metrics in results.items():
            self.log(f"val/auroc/{name}", metrics['AUROC'], sync_dist=True)
            self.log(f"val/auprc/{name}", metrics['AUPRC'], sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adamw_epsilon, betas=self.hparams.adamw_betas)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]