import torch
from torch import nn
import copy
from chimera.models.vanp.positional_encoder import PositionalEncoding

class Projector(nn.Module):
    """Projector for Barlow Twins"""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    

class MultimodalPretrainModel(nn.Module):
    def __init__(self, encoders: dict, cfg: dict, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict(encoders)
        
        self.action_encoder_type = cfg.get("action_encoder_type", "attn")
        feature_size = cfg.get("feature_size", 512)
        hidden_dim = cfg.get("hidden_dim", 8192)
        projection_dim = cfg.get("projection_dim", 8192)
        dropout = cfg.get("dropout", 0.4)
        nhead = cfg.get("nhead", 4)
        num_layers = cfg.get("num_layers", 4)
        obs_len = cfg.get("obs_len", 6)
        n_registers = cfg.get("n_registers", 4)

        img_out = cfg['img_backbone'].get("out_dim", 2048)
        act_out = cfg.get('d_model', 128)

        self.image_compressor = (
            nn.Identity() if img_out == feature_size
            else nn.Sequential(nn.Linear(img_out, feature_size), nn.LeakyReLU())
        )
        self.action_compressor = (
            nn.Identity() if act_out == feature_size
            else nn.Sequential(nn.Linear(act_out, feature_size), nn.LeakyReLU())
        )

        self.projector = Projector(feature_size, hidden_dim, projection_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.positional_encoding = PositionalEncoding(feature_size, obs_len + 1 + n_registers)

        self.ctx_token_emb = nn.Parameter(torch.randn(1, 1, feature_size), requires_grad=True)

        self.n_registers = n_registers
        if n_registers > 0:
            self.reg_token_emb = nn.Parameter(torch.randn(1, n_registers, feature_size))

    def forward(self, frames, future_frame, actions):
        B, T, L = actions.size()

        future_frame = self.image_compressor(self.encoders['img'](future_frame))

        frames = torch.stack(
            [self.image_compressor(self.encoders['img'](frame)) for frame in frames], dim=1
        )

        img_embed = frames[:, -1].clone()
        ctx_token_emb = self.ctx_token_emb.expand(B, -1, -1)
        tokens = [ctx_token_emb, frames]
        if self.n_registers > 0:
            reg_token_emb = self.reg_token_emb.expand(B, -1, -1)
            tokens.append(reg_token_emb)

        x = torch.cat(tokens, dim=1)

        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)[:,0,:]

        img_z = self.projector(x)
        future_frame_z = self.projector(future_frame)

        if self.action_encoder_type == "mlp":
            actions_flatten  = actions.view(B, -1)
            action_embed = self.encoders["action"](actions_flatten)
        elif self.action_encoder_type == "attn":
            action_embed, _ = self.encoders["action"](actions)
        else:
            raise ValueError(f"Unknown action_encoder_type {self.action_encoder_type}")

        action_z = self.projector(self.action_compressor(action_embed))

        return img_embed, img_z, action_z, future_frame_z


