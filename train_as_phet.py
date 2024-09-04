import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from importlib.machinery import SourceFileLoader
from src.datasets import TabSpiroDataModule
from src.models import AsthmaPhET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file.", type=str, required=True)
    parser.add_argument("--nb_epochs", help="Number of epochs.", type=int, required=True)
    parser.add_argument("--nb_gpus", help="Number of GPUs per node.", type=int, required=True)
    parser.add_argument("--nb_nodes", help="Number of nodes.", type=int, required=True)
    parser.add_argument("--run_name", help="Name of the run.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=1)
    parser.add_argument("--ckpt_path", help="Path to a checkpoint to resume from.", type=str, default=None)
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for data loading", default=False)
    return parser.parse_args()

    
def main():
    L.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    
    # Parse arguments:
    args = parse_args()
    cfg = SourceFileLoader("config", args.config).load_module()
    nb_epochs = args.nb_epochs
    nb_gpus = args.nb_gpus
    nb_nodes = args.nb_nodes
    run_name = args.run_name
    ckpt_path = args.ckpt_path
    nb_workers = args.nb_workers
    pin_memory = args.pin_memory
        
    # Create data module: 
    dm = TabSpiroDataModule(
        tab_train = cfg.tab_train,
        tab_val = cfg.tab_val,
        tab_test = cfg.tab_test,
        spiro_train = cfg.spiro_train,
        spiro_val = cfg.spiro_val,
        spiro_test = cfg.spiro_test,
        ckpt_tokenizer = cfg.ckpt_phet,
        num_features = cfg.num_features,
        cat_features = cfg.cat_features,
        batch_size = cfg.batch_size,
        n_workers = nb_workers, 
        pin_memory = pin_memory
    )
    
    # Create model:
    model = AsthmaPhET(
        ckpt_phet = cfg.ckpt_phet,
        ckpt_resnet = cfg.ckpt_resnet,
        n_embeds = cfg.n_embeds,
        freeze_phet = cfg.freeze_phet,
        freeze_resnet = cfg.freeze_resnet,
        learning_rate = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        warmup_steps = cfg.warmup_steps
    )
    
    # Set callbacks:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    val_ckpt = ModelCheckpoint(
        dirpath = f'ckpts/AsthmaPhE-T/{run_name}',
        filename = 'best-{epoch}-{step}',
        monitor = 'val/loss',
        mode = 'min',
        save_top_k = 1,
        save_on_train_epoch_end = False,
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10
    )

    # Set logger:
    logger = WandbLogger(project='AsthmaPhE-T', name=run_name)
    logger.watch(model)
    
    # Set trainer:
    trainer = L.Trainer(
        max_epochs = nb_epochs,
        devices = nb_gpus,
        num_nodes = nb_nodes,
        log_every_n_steps = 10,
        val_check_interval = 50,
        strategy = "ddp",
        logger = logger, 
        # callbacks = [lr_monitor, val_ckpt, early_stop],
        callbacks = [lr_monitor, val_ckpt],
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True,
        fast_dev_run = False
    )

    # Fit model:
    #trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()