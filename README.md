# Chimera 🧬⚡

**Chimera** is a research framework for exploring multimodal self-supervised learning.  
It combines ideas from [Barlow Twins](https://arxiv.org/abs/2103.03230), [VICReg](https://arxiv.org/abs/2105.04906), and [Decoupled Contrastive Learning](https://arxiv.org/abs/2309.05300), extending them for visual navigation and real-world datasets.  

The goal is to build a flexible prototype for reproducing known experiments and testing new hypotheses in multimodal SSL.

---

## 🚀 Quickstart

1. **Build and run the container**
   ```bash
   cd Docker
   docker compose up --build -d
   ```

2.	Install dependencies inside the container
(this is done automatically on first run, but you can re-run if needed):
    ```bash
    uv pip install -e ".[pl,clearml viz,dev]"
    ```


3.	Test the setup
    ```python
    chimera-train
     ```
    Expected output:
    ```
    Hello
    Cuda : True
    ```




## Project Structure
```
chimera/        # Main Python package
  data/         # Dataset loading and preparation
  losses/       # Loss functions (VICReg, Barlow Twins, etc.)
  models/       # Encoder architectures
  train/        # Training scripts (PyTorch / Lightning)
  scripts/      # Utilities and CLI entrypoints
  utils/        # Helpers (reproducibility, logging, etc.)
config/         # YAML configuration files
Docker/         # Dockerfile + docker-compose
dataset/        # Datasets (mounted inside the container)

```


##  Logging & Monitoring

1. **ClearML** — experiment tracking (requires a valid `.clearml.conf`, mounted automatically inside the container).  
2. **CometML** — optional additional logging.  
3. **TensorBoard** + [`rich`](https://github.com/Textualize/rich) — visualization and console-friendly monitoring.  

---

## 📌 Roadmap

- [ ] Reproduce VICReg / Barlow Twins experiments  
- [ ] Add multimodal data support  
- [ ] Compare alternative regularization methods (HSIC, CKA)  
- [ ] Provide pytest coverage for core modules  

---

## ⚡ Entry Points

- `chimera-train` — basic training script  
- `chimera-pl-train` — PyTorch Lightning training loop  
- `chimera-eval` — evaluation script  

---

## 💡 Notes

- ClearML requires a valid `~/.clearml.conf` (already mounted from `Docker/configs/clearml.conf`).  
- Use [`uv`](https://github.com/astral-sh/uv) (instead of pip) for fast dependency management.  
- CUDA **12.6 runtime** is included in the base image with PyTorch `cu121` wheels.  