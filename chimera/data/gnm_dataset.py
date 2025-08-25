# chimera/data/gnm2_socialnav_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

# ---- egowalk / GNM imports (как в твоём окружении) ----
from egowalk_dataset.datasets.gnm.gnm_dataset import (
    GNMDataset,
    GNMFeature,
    GNMTuple,
    GNMRGBFeature,
    GNMWaypointFeature,
    DefaultGNMDataset,
)
from egowalk_dataset.datasets.gnm.gnm_indexing import index_gnm
from egowalk_dataset.datasets.gnm.cutters import SpikesCutter, StuckCutter, BackwardCutter


# =========================
#  Augmentations (минимум)
# =========================

class GaussianBlur:
    def __init__(self, p: float) -> None:
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(float(sigma)))
        return img


class Solarization:
    def __init__(self, p: float) -> None:
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        return img


# ==================================
#  Traversability mask as a feature
# ==================================

class GNMTraversabilityMaskFeature(GNMFeature):
    """
    Загружает PNG маску проходимости для последнего obs кадра траектории.
    Ожидаемый путь: {mask_root}/{trajectory_name}/frame_{idx:05d}.png

    Возвращает torch.LongTensor формы [1, H, W] (после pil_to_tensor).
    """
    def __init__(
        self,
        name: str = "obs_traversability",
        mask_root: Union[str, Path] = "/home/oversir/workspace/dataset/aux_channels/traversability_mask",
    ) -> None:
        super().__init__(name)
        self._mask_root = Path(mask_root)

    def __call__(self, root: Path, gnm_tuple: GNMTuple) -> Tensor:
        traj = gnm_tuple.trajectory_name
        obs_idx = gnm_tuple.obs_idxs[-1]
        mask_path = self._mask_root / traj / f"frame_{obs_idx:05d}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Traversability mask not found: {mask_path}")
        mask_pil = Image.open(mask_path).convert("L")
        # [1,H,W] long
        return TF.pil_to_tensor(mask_pil).long()


class GNMExtendedDataset(GNMDataset):
    """GNMDataset c дополнительной фичей трэвёрс‑маски."""
    def __init__(
        self,
        index: Dict[str, Any],
        image_transform: Optional[Callable[[np.ndarray], Union[np.ndarray, Tensor]]] = None,
        angle_format: Literal["none", "yaw", "sincos"] = "none",
        data_path: Union[str, Path] = "/home/oversir/workspace/dataset",
        mask_root: Union[str, Path] = "/home/oversir/workspace/dataset/aux_channels/traversability_mask",
    ) -> None:
        obs_f = GNMRGBFeature(name="obs", field="obs", transform=image_transform)
        goal_f = GNMRGBFeature(name="goal", field="goal", transform=image_transform)
        act_f  = GNMWaypointFeature(name="action", angle_format=angle_format)
        trav_f = GNMTraversabilityMaskFeature(name="obs_traversability", mask_root=mask_root)
        super().__init__(index, [obs_f, goal_f, act_f, trav_f], data_path)


# =========================
#  Main dataset
# =========================

class GNM2SocialNavDataset(Dataset):
    """
    Drop-in датасет под твой VANP-пайплайн.

    Возвращаемые ключи sample:
      - past_frames: List[T] of Tensor[3,H,W]  (time-major список)
      - future_frame: Tensor[3,H,W]
      - original_frame: np.ndarray[H,W,3] uint8 (для быстрого plt.imshow)
      - (опц.) obs_traversability: Tensor[1,H,W] float32 в [0,1] (NEAREST)
      - future_positions: Tensor[pred_len,2] float32
      - future_yaw: Tensor[pred_len] float32 (если use_yaw, иначе нули)
      - past_positions: Tensor[obs_len,2] float32 (заглушка нули, если нет в GNM)
      - past_yaw: Tensor[obs_len] float32 (нули)
      - past_vw:  Tensor[obs_len,2] float32 (нули)
      - future_vw: Tensor[pred_len,2] float32 (нули)
      - goal_direction: Tensor[2] float32 = [radius, angle] нормированный по radius
      - dt: Tensor[1] float32 = 1 - 1/(dt_idx+1), где dt_idx ∈ [pred_len//2, pred_len)
    """

    def __init__(
        self,
        obs_len: int = 6,
        pred_len: int = 20,
        use_yaw: bool = False,
        train: bool = True,
        resize: Union[Tuple[int, int], List[int]] = (224, 224),  # (H, W)
        transform_original_frame: Optional[Callable[[np.ndarray], Any]] = None,
        use_mask: bool = False,
        data_path: Union[str, Path] = "/home/oversir/workspace/dataset",
        mask_root: Union[str, Path] = "/home/oversir/workspace/dataset/aux_channels/traversability_mask",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.use_yaw = bool(use_yaw)
        self.train = bool(train)
        self.resize = (int(resize[0]), int(resize[1]))  # (H, W) для torchvision
        self.use_mask = bool(use_mask)
        self.data_path = Path(data_path)
        self.mask_root = Path(mask_root)
        self.transform_original_frame = transform_original_frame

        # индексируем GNM (твои параметры)
        gnm_index = index_gnm(
            cutters=[
                StuckCutter(eps=1e-2),
                BackwardCutter(backwards_eps=1e-2, stuck_eps=1e-2, ignore_stuck=True),
                SpikesCutter(spike_threshold=2.0),
            ],
            window_step=30,
            context_length=5,
            goal_offset=(3, 20),
            goal_offset_mode="sampled",
            action_length=self.pred_len,
            context_step=1,
            action_step=1,
            n_workers=12,
            data_path=str(self.data_path),   # у тебя теперь это поле есть
            use_tqdm=True,
        )

        if self.use_mask:
            self.ds = GNMExtendedDataset(
                index=gnm_index,
                angle_format="yaw",
                data_path=str(self.data_path),
                mask_root=str(self.mask_root),
            )
        else:
            self.ds = DefaultGNMDataset(
                index=gnm_index, angle_format="yaw", data_path=str(self.data_path)
            )

        # — цветовые аугментации отдельно (без геометрии) —
        if self.train:
            self.color_tf = transforms.Compose(
                [
                    transforms.RandomAutocontrast(p=0.4),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.6),
                    Solarization(p=0.5),
                ]
            )
        else:
            self.color_tf = None

        # нормализация отдельно
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # локальный RNG только для dt, если хочешь стабильность
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ds)

    # ---------- helpers ----------

    @staticmethod
    def _np_to_pil_uint8(frame: np.ndarray) -> Image.Image:
        """(H,W,3) ndarray -> PIL RGB, гарантируя uint8."""
        if frame.dtype != np.uint8:
            f = frame
            if f.max() <= 1.0:
                f = (f * 255.0).clip(0, 255).astype(np.uint8)
            else:
                f = f.clip(0, 255).astype(np.uint8)
        else:
            f = frame
        return Image.fromarray(f)

    def _sample_geom(self, pil_img: Image.Image) -> Tuple[int, int, int, int, bool]:
        """
        Сэмплим параметры RandomResizedCrop + флип ОДИН РАЗ на весь сэмпл,
        чтобы синхронно применить к всем past кадрам, goal и маске.
        """
        if self.train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                pil_img, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)
            )
            do_flip = bool(np.random.rand() < 0.5)
        else:
            i, j, h, w = 0, 0, pil_img.height, pil_img.width
            do_flip = False
        return i, j, h, w, do_flip

    def _apply_geom(self, img: Image.Image, geom: Tuple[int, int, int, int, bool], is_mask: bool) -> Image.Image:
        i, j, h, w, do_flip = geom
        img = TF.resized_crop(
            img,
            top=i, left=j, height=h, width=w,
            size=self.resize,  # (H, W)
            interpolation=InterpolationMode.NEAREST if is_mask else InterpolationMode.BICUBIC,
            antialias=not is_mask,
        )
        if do_flip:
            img = TF.hflip(img)
        return img

    # ---------- main ----------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]  # {'obs', 'goal', 'action', ...}

        obs_np: np.ndarray = item["obs"]    # (T, H, W, 3)
        goal_np: np.ndarray = item["goal"]  # (H, W, 3)
        action = item["action"]             # (pred_len, 3) обычно [x, y, yaw]

        # один раз сэмплим параметры геом. аугментации от последнего кадра
        probe = self._np_to_pil_uint8(obs_np[-1])
        geom = self._sample_geom(probe)

        # past_frames: список [3,H,W]
        past_frames: List[Tensor] = []
        for t in range(obs_np.shape[0]):
            pil = self._np_to_pil_uint8(obs_np[t])
            pil = self._apply_geom(pil, geom, is_mask=False)
            if self.color_tf is not None:
                pil = self.color_tf(pil)
            img = TF.to_tensor(pil)
            img = self.normalize(img)
            past_frames.append(img)

        pil_goal = self._np_to_pil_uint8(goal_np)
        pil_goal = self._apply_geom(pil_goal, geom, is_mask=False)
        if self.color_tf is not None:
            pil_goal = self.color_tf(pil_goal)
        future_frame: Tensor = self.normalize(TF.to_tensor(pil_goal))

        original_frame: np.ndarray = np.array(self._apply_geom(self._np_to_pil_uint8(obs_np[-1]), geom, is_mask=False))

        if "future_positions" in item:
            future_positions = torch.as_tensor(item["future_positions"], dtype=torch.float32)
        else:
            future_positions = torch.as_tensor(np.asarray(action)[:, :2], dtype=torch.float32)

        if self.use_yaw:
            if isinstance(action, Tensor):
                yaw_np = action.detach().cpu().numpy()
            else:
                yaw_np = np.asarray(action)
            future_yaw = torch.as_tensor(yaw_np[:, 2] if yaw_np.shape[1] >= 3 else np.zeros(self.pred_len, np.float32), dtype=torch.float32)
        else:
            future_yaw = torch.zeros(self.pred_len, dtype=torch.float32)

        past_positions = torch.zeros(self.obs_len, 2, dtype=torch.float32)
        past_yaw = torch.zeros(self.obs_len, dtype=torch.float32)
        past_vw = torch.zeros(self.obs_len, 2, dtype=torch.float32)
        future_vw = torch.zeros(self.pred_len, 2, dtype=torch.float32)

        dt_idx = int(self._rng.integers(low=max(1, self.pred_len // 2), high=self.pred_len))
        goal_xy = future_positions[dt_idx]  # [2]
        radius = torch.linalg.vector_norm(goal_xy) + 1e-6
        angle = torch.atan2(goal_xy[1], goal_xy[0])
        goal_direction = torch.tensor([radius.item(), angle.item()], dtype=torch.float32)
        goal_direction /= (radius + 1e-6)
        dt_tensor = torch.tensor([1.0 - (1.0 / (dt_idx + 1))], dtype=torch.float32)

        sample: Dict[str, Any] = {
            "past_frames": past_frames,                # list[T] of [3,H,W]
            "future_frame": future_frame,              # [3,H,W]
            "original_frame": original_frame,          # HWC uint8 (numpy)
            "past_positions": past_positions,          # [obs_len,2]
            "future_positions": future_positions,      # [pred_len,2]
            "past_yaw": past_yaw,                      # [obs_len]
            "future_yaw": future_yaw,                  # [pred_len]
            "past_vw": past_vw,                        # [obs_len,2]
            "future_vw": future_vw,                    # [pred_len,2]
            "goal_direction": goal_direction,          # [2]
            "dt": dt_tensor,                           # [1]
        }

        if self.use_mask and "obs_traversability" in item:
            mask_raw: Tensor = item["obs_traversability"]
            mask_np = mask_raw.squeeze(0).cpu().numpy().astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode="L")
            mask_pil = self._apply_geom(mask_pil, geom, is_mask=True)
            mask_t = TF.pil_to_tensor(mask_pil).float() / 255.0 
            sample["obs_traversability"] = mask_t

        return sample