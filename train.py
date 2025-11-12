import pytorch_lightning as L
import yaml
import wandb
import time
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning import seed_everything

from navdiffusion import NavDiffsionLightning, NavigationDataset

def main(config, wandb_logger):

    seed_everything(42, workers=True)

    train_set = NavigationDataset(config['data_params'], config['model_params'], split='train')
    val_set = NavigationDataset(config['data_params'], config['model_params'], split='val')
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_params']["batch_size"],
        shuffle=True,
        num_workers=config['train_params']["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config['train_params']["batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    
    print("\033[32m [ Sucess build Dataset! ] \033[0m" + f"num of train datapoints : {len(train_set)}, num of val datapoints : {len(val_set)}")

    #-------------------------- model --------------------------#

    navdiff = NavDiffsionLightning(config["model_params"], config["train_params"])

    #-------------------------- train & val --------------------------#

    trainer = L.Trainer( 
        accelerator = 'gpu', 
        devices = 1,
        logger = wandb_logger,
        deterministic = True,
        max_epochs = config['train_params']["max_epochs"],
        gradient_clip_val = config['train_params']["grad_clip_max_norm"],
        log_every_n_steps = 5,
        callbacks=[
            ModelSummary(max_depth=1),
            ModelCheckpoint(dirpath="/home/iunone/forest_nav_diffusion/results")
        ]
    )

    trainer.fit(
        model=navdiff, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":

    cfg_path = "config/lightning.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    config["wandb_params"]["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    wandb_params = config["wandb_params"]
    wandb_logger = WandbLogger(
        project=wandb_params["project_name"],
        log_model=True,     # Log model checkpoints at the end of training
        entity=wandb_params["entity"],
        settings=wandb.Settings(start_method="fork"),
        name=wandb_params["run_name"],
        )
    
    main(config, wandb_logger)