import torch
import torch.nn.functional as F
import pytorch_lightning as L

from .diffusion_policy import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from warmup_scheduler import GradualWarmupScheduler

from .vision_transformer.transformer_encoder import VisualEncoder

class NavDiffsionLightning(L.LightningModule):

    def __init__(self, model_params, train_params):
        super().__init__()

        # visual conditioning encoder
        self.vision_encode_param = model_params["vision_encoder"]
        self.transformer_encoder = VisualEncoder(
            obs_encoder=self.vision_encode_param.get(
                "obs_encoder", "efficientnet-b0"
            ),
            embedding_dim=self.vision_encode_param["embedding_dim"],
            input_channels=self.vision_encode_param.get("input_channels", 3),
            pretrained=self.vision_encode_param.get("pretrained", False),
            replace_batch_norm=self.vision_encode_param.get(
                "replace_batch_norm", True
            ),
            spatial_pool_size=self.vision_encode_param.get("spatial_pool_size", 4),
            spatial_channels=self.vision_encode_param.get("spatial_channels", 128),
            dropout=self.vision_encode_param.get("dropout", 0.0),
        )

        # noise diffusion
        self.noise_pred_net_param = model_params["noise_pred_net"]        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,  # x,y
            global_cond_dim=self.vision_encode_param["embedding_dim"],
            diffusion_step_embed_dim=self.noise_pred_net_param.get(
                "diffusion_step_embed_dim", 256
            ),
            down_dims=self.noise_pred_net_param["down_dims"],
            trajectory_horizon=model_params["len_traj_pred"],
            kernel_size=self.noise_pred_net_param.get("kernel_size", 3),
            n_groups=self.noise_pred_net_param.get("n_groups", 8),
            cond_predict_scale=self.noise_pred_net_param["cond_predict_scale"],
        )
        self.num_train_timesteps = int(
            self.noise_pred_net_param.get(
                "num_train_timesteps",
                self.noise_pred_net_param.get("num_diffusion_iters", 10),
            )
        )
        self.num_inference_steps = int(
            self.noise_pred_net_param.get(
                "num_inference_steps",
                self.noise_pred_net_param.get("num_diffusion_iters", 10),
            )
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # planning horizon
        self.len_traj_pred = model_params["len_traj_pred"]
        self.register_buffer(
            "traj_norm_lower",
            torch.tensor(
                [model_params["traj_norm_x"][0], model_params["traj_norm_y"][0]],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "traj_norm_higher",
            torch.tensor(
                [model_params["traj_norm_x"][1], model_params["traj_norm_y"][1]],
                dtype=torch.float32,
            ),
        )

        self.optimizer_params = train_params["optimizer"]

        self.epochs = train_params["max_epochs"]
        self.save_hyperparameters()

    def forward(self, 
        batch_imgs: torch.Tensor, 
        batch_future_wpts: torch.Tensor = None
    ):
        '''
            - batch_obs_images: [B, 3, H, W]
            - batch_future_wpts: [B, len_traj_pred, 2]
        '''
        bs = batch_imgs.shape[0]

        image_cond = self.transformer_encoder(batch_imgs)
        
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=self.device,
        ).long()
        if batch_future_wpts is not None:
            noise = torch.randn_like(batch_future_wpts)
            noisy_action = self.noise_scheduler.add_noise(batch_future_wpts, noise, timesteps)
            noise_pred = self.noise_pred_net(sample=noisy_action, timestep=timesteps, global_cond=image_cond)
        else:
            naction = torch.randn(
                (bs, self.len_traj_pred, 2),
                device=self.device,
                dtype=image_cond.dtype,
            )
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            # denoise in k steps
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=image_cond,
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
        backbone_lr = self.optimizer_params.get("backbone_lr")
        if backbone_lr is None:
            optimizer_parameters = self.parameters()
        else:
            backbone_parameters = list(
                self.transformer_encoder.obs_encoder.parameters()
            )
            backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
            new_parameters = [
                parameter
                for parameter in self.parameters()
                if id(parameter) not in backbone_parameter_ids
            ]
            optimizer_parameters = [
                {"params": backbone_parameters, "lr": float(backbone_lr)},
                {"params": new_parameters, "lr": lr},
            ]
        weight_decay = float(self.optimizer_params.get("weight_decay", 0.0))
        optimizer_name = self.optimizer_params["optimizer"].lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_parameters,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

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

        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    
    def training_step(self, batch, batch_idx):
        '''
            forward model and return loss
            
            - Batch from dataset:
                imgs : [bs, 3, H, W]
                future_wpts : [B, len_traj_pred, 2]
            
            - Loss:
                diffusion_loss
        '''

        imgs, future_wpts = batch
        
        future_wpts = (
            (future_wpts - self.traj_norm_lower)
            / (self.traj_norm_higher - self.traj_norm_lower)
            * 2
            - 1
        )
        noise_pred, noise = self.forward(imgs, future_wpts)

        noise_loss = F.mse_loss(noise_pred, noise, reduction="none")
        while noise_loss.dim() > 1:
            noise_loss = noise_loss.mean(dim=-1)
        diffusion_loss = noise_loss.mean()

        self.log(
            "train/loss",
            diffusion_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=imgs.shape[0],
        )

        return diffusion_loss

    def predict(self, batch):
        
        imgs, _ = batch

        pred_wpts, _ = self.forward(imgs)
        pred_wpts = (
            (pred_wpts + 1)
            / 2
            * (self.traj_norm_higher - self.traj_norm_lower)
            + self.traj_norm_lower
        )

        return pred_wpts
