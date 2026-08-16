import time

import pytorch_lightning as L
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from navdiffusion import NavigationDataset, NavDiffsionLightning


def main(config, wandb_logger):
    train_params = config["train_params"]
    seed_everything(train_params["seed"], workers=True)

    train_set = NavigationDataset(
        config["data_params"], config["model_params"], split="train"
    )

    train_workers = train_params["num_workers"]
    train_loader = DataLoader(
        train_set,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=train_workers,
        drop_last=False,
        persistent_workers=train_workers > 0,
    )

    print(
        "\033[32m[Successfully built dataset]\033[0m "
        "train samples: {} (all valid samples, no validation split)".format(
            len(train_set)
        )
    )

    navdiff = NavDiffsionLightning(config["model_params"], train_params)
    trainer = L.Trainer(
        accelerator=train_params.get("accelerator", "auto"),
        devices=train_params.get("devices", 1),
        strategy=train_params.get("strategy"),
        precision=train_params.get("precision", 32),
        logger=wandb_logger,
        deterministic=train_params.get("deterministic", False),
        max_epochs=train_params["max_epochs"],
        gradient_clip_val=train_params["grad_clip_max_norm"],
        log_every_n_steps=5,
        callbacks=[
            ModelSummary(max_depth=1),
            ModelCheckpoint(dirpath=train_params["checkpoint_dir"]),
        ],
    )
    trainer.fit(
        model=navdiff,
        train_dataloaders=train_loader,
    )


if __name__ == "__main__":
    with open("config/lightning.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    config["wandb_params"]["run_name"] += "_" + time.strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    wandb_params = config["wandb_params"]
    wandb_logger = WandbLogger(
        project=wandb_params["project_name"],
        log_model=True,
        entity=wandb_params["entity"],
        name=wandb_params["run_name"],
    )
    main(config, wandb_logger)
