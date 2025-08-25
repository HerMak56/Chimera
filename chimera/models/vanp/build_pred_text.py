from typing import Literal, Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from torchvision import models

from chimera.models.vanp.pred_text import MultimodalPretrainModel
from chimera.models.vanp.encoders import Transformer 

@torch.no_grad()
def _set_requires_grad(module: nn.Module, requires: bool):
    for p in module.parameters():
        p.requires_grad = requires

def _build_action_encoder(
    encoder_type: Literal["mlp", "attn"],
    action_size: int,
    pred_len: int,
    mcfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if encoder_type.lower() == "mlp":
        # dims_cfg = mcfg.get("action_backbone", {}).get("dims", None)
        # if not dims_cfg:
        #     raise ValueError("For action_encoder_type=mlp you must set model.action_backbone.dims in YAML.")
        # in_dim = pred_len * action_size
        # dims = [in_dim] + list(dims_cfg)
        # activation = mcfg.get("action_backbone", {}).get("act", "relu")
        # dropout = float(mcfg.get("action_backbone", {}).get("dropout", 0.0))
        # mlp = make_mlp(dims=dims, activation=activation, dropout=dropout).to(device)
        # return mlp
        raise NotImplementedError

    elif encoder_type.lower() == "attn":
        attn = Transformer(
            n_layers=mcfg["attn"]["num_layers"],
            d_model=mcfg["attn"]["context_size"],
            n_head=mcfg["attn"]["nhead"],
            n_action=action_size,
            d_hidden=mcfg["attn"]["d_hid"],
            pred_len=pred_len,
            action_type=mcfg["action_type"],
            dropout=mcfg["attn"]["dropout"],
            n_registers=mcfg["attn"]["n_registers"],
        ).to(device)
        return attn

    else:
        raise ValueError(f"Unknown action_encoder_type={encoder_type}")

def _build_image_backbone(name: str, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    enc = models.get_model(name.lower(), weights=weights, zero_init_residual=True)
    # выкинем классификатор
    if hasattr(enc, "fc"):
        enc.fc = nn.Identity()
    elif hasattr(enc, "classifier"):
        enc.classifier = nn.Identity()
    return enc

def build_pretext_model(
    cfg: Dict[str, Any],
    device
) -> MultimodalPretrainModel:
    device = device
    img_name = cfg["img_backbone"]["name"]
    img_pretrained = bool(cfg["img_backbone"].get("pretrained", False))
    img_backbone = _build_image_backbone(img_name, img_pretrained).to(device)
    
    action_encoder = _build_action_encoder(encoder_type=cfg["action_encoder_type"],
                      action_size=cfg['action_size'],
                      pred_len=cfg['pred_len'],
                      mcfg=cfg,
                      device=device)
    
    if cfg.get("freeze_action_backbone", False):
        _set_requires_grad(action_encoder, False)
        action_encoder.eval()

    # 4) собрать PretextModel с нужными гиперпараметрами
    pretext = MultimodalPretrainModel(
        encoders={"img": img_backbone, "action": action_encoder},
        cfg=cfg
    ).to(device)

    return pretext
    