import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import List

from .diffusion_policy import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from warmup_scheduler import GradualWarmupScheduler

from .vision_transformer.transformer_encoder import VisualGoalTransformer, replace_bn_with_gn

class NavDiffsionLightning(L.LightningModule):

    def __init__(self, model_params, train_params):
        super().__init__()

        # vision transformer
        self.vision_encode_param = model_params["vision_encoder"]
        self.transformer_encoder = VisualGoalTransformer(
            time_horizon=self.vision_encode_param["time_horzion"],
            embedding_dim=self.vision_encode_param["embedding_dim"]
        )
        self.transformer_encoder = replace_bn_with_gn(self.transformer_encoder) 

        # noise diffusion
        self.noise_pred_net_param = model_params["noise_pred_net"]        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim = 2,  # x,y
            global_cond_dim = self.vision_encode_param["embedding_dim"],
            down_dims = self.noise_pred_net_param["down_dims"],
            cond_predict_scale = self.noise_pred_net_param["cond_predict_scale"],
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_pred_net_param["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # planning horizon
        self.len_traj_pred = model_params["len_traj_pred"]
        self.traj_norm_lower = [model_params["traj_norm_x"][0], model_params["traj_norm_y"][0]]
        self.traj_norm_higher = [model_params["traj_norm_x"][1], model_params["traj_norm_y"][1]]

        self.optimizer_params = train_params["optimizer"]

        self.epochs = train_params["max_epochs"]
        self.save_hyperparameters()

    def forward(self, 
        batch_imgs: torch.Tensor, 
        batch_goals: torch.Tensor, 
        batch_future_wpts: torch.Tensor = None
    ):
        '''
            - batch_obs_images: [B, context, H, W]
            - batch_goal_pos: [B, 1, 2]
            - batch_future_wpts: [B, len_traj_pred, 2]
        '''
        bs = batch_imgs.shape[0]

        obsgoal_cond = self.transformer_encoder.forward(batch_imgs, batch_goals)
        
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,(bs,)).long().to(self.device)
        if batch_future_wpts is not None:
            noise = torch.randn(batch_future_wpts.shape).to(self.device)
            noisy_action = self.noise_scheduler.add_noise(batch_future_wpts, noise, timesteps)
            noise_pred = self.noise_pred_net(sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)
        else:
            naction = torch.randn((bs, self.len_traj_pred, 2)).to(self.device)
            self.noise_scheduler.set_timesteps(self.noise_pred_net_param["num_diffusion_iters"])
            # denoise in k steps
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obsgoal_cond,
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            noise_pred = naction
            noise = None

        return noise_pred, noise

    def configure_optimizers(self):

        # build optimizer
        lr = float(self.optimizer_params["lr"])
        self.optimizer_params["optimizer"] = self.optimizer_params["optimizer"].lower()
        if self.optimizer_params["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98))
        elif self.optimizer_params["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif self.optimizer_params["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.optimizer_params['optimizer']} not supported")

        # build lr scheduler
        scheduler_params = self.optimizer_params["scheduler"]
        if scheduler_params['type'] is not None: 
            if scheduler_params['type'] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
                    optimizer, 
                    T_max = self.epochs
                )
            elif scheduler_params['type'] == "cyclic":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=lr / 10.,
                    max_lr=lr,
                    step_size_up=scheduler_params["cyclic_period"] // 2,
                    cycle_momentum=False,
                )
            elif scheduler_params['type'] == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=scheduler_params["plateau_factor"],
                    patience=scheduler_params["plateau_patience"],
                    verbose=True,
                    )
            else:
                raise ValueError(f"Scheduler {scheduler_params['type']} not supported")

            if scheduler_params["warmup"]:
                print("Using warmup scheduler")
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=scheduler_params["warmup_epochs"],
                    after_scheduler=scheduler,
                )
        else:
            scheduler = None

        return {
                "optimizer": optimizer, 
                "lr_scheduler": scheduler, 
            } 
    
    def training_step(self, batch, batch_idx):
        '''
            forward model and return loss
            
            - Batch from dataset:
                imgs : [bs, T, 4, H, W]
                goals : [B, 2]
                future_wpts : [B, len_traj_pred, 2]
            
            - Loss:
                diffusion_loss
        '''

        imgs, goals, future_wpts = batch
        
        traj_lower = torch.tensor(self.traj_norm_lower, device=self.device)
        traj_higher = torch.tensor(self.traj_norm_higher, device=self.device)
        future_wpts = (future_wpts - traj_lower) / (traj_higher - traj_lower) * 2 - 1
        noise_pred, noise = self.forward(imgs, goals, future_wpts)

        noise_loss = F.mse_loss(noise_pred, noise, reduction="none")
        while noise_loss.dim() > 1:
            noise_loss = noise_loss.mean(dim=-1)
        diffusion_loss = noise_loss.mean()

        self.log("train/loss", diffusion_loss)

        return diffusion_loss

    def validation_step(self, batch, batch_idx):
        
        imgs, goals, future_wpts = batch

        pred_wpts, _ = self.forward(imgs, goals)
        traj_lower = torch.tensor(self.traj_norm_lower, device=self.device)
        traj_higher = torch.tensor(self.traj_norm_higher, device=self.device)
        pred_wpts = (pred_wpts + 1) / 2 * (traj_higher - traj_lower) + traj_lower
        l2_metric = F.mse_loss(pred_wpts, future_wpts, reduction="mean")

        self.log("val/l2_metric", l2_metric)

    def predict(self, batch):
        
        imgs, goals, _ = batch

        pred_wpts, _ = self.forward(imgs, goals)
        traj_lower = torch.tensor(self.traj_norm_lower, device=self.device)
        traj_higher = torch.tensor(self.traj_norm_higher, device=self.device)
        pred_wpts = (pred_wpts + 1) / 2 * (traj_higher - traj_lower) + traj_lower

        return pred_wpts
