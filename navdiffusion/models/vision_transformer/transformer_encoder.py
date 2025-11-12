import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from navdiffusion.models.vision_transformer.base_module import PositionalEncoding
from navdiffusion.models.vision_transformer.base_module import MLP

class VisualGoalTransformer(nn.Module):

    def __init__(
        self,
        obs_encoder: Optional[str] = "efficientnet-b0",
        time_horizon: int = 5,
        embedding_dim: Optional[int] = 128,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 4,
        mha_ff_dim_factor: Optional[int] = 4,
    ):

        super().__init__()
        self.embedding_dim = embedding_dim

        #------------------------------------- vision embedding

        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=4)
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError

        if self.num_obs_features != self.embedding_dim:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.embedding_dim)
        else:
            self.compress_obs_enc = nn.Identity()
        
        #------------------------------------- goal embedding

        self.goal_encoder = MLP(2, 128, self.embedding_dim, 2)

        #------------------------------------ position embedding

        self.positional_encoding = PositionalEncoding(self.embedding_dim, max_seq_len=1+time_horizon)

        self.transformer = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=mha_num_attention_heads, 
                dim_feedforward=mha_ff_dim_factor*self.embedding_dim, 
                activation="gelu", 
                batch_first=True, 
                norm_first=True
            ), 
            num_layers = mha_num_attention_layers
        )

    def forward(self, 
        imgs: torch.tensor, 
        goals: torch.tensor
    ) -> torch.Tensor:

        bs, t = imgs.shape[:2]
        
        # [bs, t, ch, h, w] -> [bs*t, ch, h, w]
        imgs = imgs.view(bs * t, *imgs.shape[2:])
        
        vision_emb = self.obs_encoder.extract_features(imgs)
        vision_emb = self.obs_encoder._avg_pooling(vision_emb)
        if self.obs_encoder._global_params.include_top:
            vision_emb = vision_emb.flatten(start_dim=1)
            vision_emb = self.obs_encoder._dropout(vision_emb)
        vision_emb = self.compress_obs_enc(vision_emb)
        
        # [bs*t, emb_dim] -> [bs, t, emb_dim]
        vision_emb = vision_emb.view(bs, t, -1)

        # [bs, 2] --> [bs, 1, self.embedding_dim]
        goal_emb = self.goal_encoder.forward(goals).unsqueeze(dim=1)

        # --> [bs, T+1, self.embedding_dim]
        query = torch.cat((vision_emb, goal_emb), dim=1)            
        if self.positional_encoding:
            query = self.positional_encoding(query)

        # [bs, T+1, self.embedding_dim]--> [bs, 256]
        tokens = self.transformer(query, src_key_padding_mask=None)
        tokens = torch.mean(tokens, dim=1)         

        return tokens


# -------------------------------------------------------------------------------------

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module
