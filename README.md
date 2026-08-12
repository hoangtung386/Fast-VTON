# Fast-VTON

One-step virtual try-on powered by SwiftEdit diffusion editing.

## Overview

Fast-VTON adapts [SwiftEdit](https://github.com/Qualcomm-AI-research/SwiftEdit)'s one-step text-guided image editing for high-fidelity virtual try-on. Instead of text prompts, the system takes a garment image as conditioning and composites it onto a person photo in a single diffusion step.

### Key Features

- **One-step inference** — single UNet pass (after optional inversion network ablation)
- **ARaM** (Adaptive Region-aware Masking) — spatial control via three scaling factors (`s_y`, `s_edit`, `s_non-edit`)
- **Garment branch** — DINOv2 patch tokens replace the text prompt via cross-attention K/V
- **Inpainting conditioning** — 9-channel `conv_in` (4 noisy + 4 masked + 1 mask) for explicit mask guidance
- **Stage 1 training** — only ~33 M of 1.76 B parameters are unfrozen (on A100 40 GB)

## Architecture

| Component | Role |
|---|---|
| **Generator (SBv2)** | Frozen UNet, inpainting channels zero-initialized |
| **Inversion network (F_θ)** | Predicts inverted noise; frozen in Stage 1, empty-prompt trick |
| **Garment encoder** | DINOv2-large → 257 patch tokens via projection (~3.1 M) |
| **IP-Adapter branch** | CLIP image embedding (4 tokens) from agnostic person photo |
| **Prompt branch K/V** | Fine-tuned to accept garment features (~25.56 M) |

## Project Structure

```
src/
├── attention/          # ARaM mask controller & processors
├── models/             # InverseModel, IPSBV2Model, AuxiliaryModel
├── pipelines/          # End-to-end edit_image()
├── vton/               # Virtual try-on adaptation
│   ├── config.py       # DataConfig, Stage1Config, CheckpointConfig
│   ├── masking.py      # Agnostic mask construction
│   ├── garment_encoder.py
│   ├── precompute.py   # Cache frozen features
│   ├── trainer.py      # Stage 1 training loop
│   └── freezing.py     # Selective parameter groups
└── utils/
scripts/
├── run_edit.py         # Inference script
├── train_vton_stage1.py
├── build_vton_cache.py
├── make_null_embedding.py
└── dissect_checkpoints.py
```

## Installation

```bash
# Clone
git clone <your-fork> && cd Fast-VTON

# Install with virtual try-on extras
pip install -e '.[vton]'

# Or install dev dependencies
pip install -e '.[dev]'
```

**Requirements:** Python ≥ 3.12, CUDA 11.8+ (for PyTorch cu118 wheel).

## Pretrained Weights

Download from [SwiftEdit releases](https://github.com/Qualcomm-AI-research/SwiftEdit/releases):

```bash
# Multi-part download
for p in "aa ab ac ad ae"; do
    wget -q -c https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-${p}
done
cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
tar zxf swiftedit_weights.tar.gz && rm swiftedit_weights.tar.gz*

# The archive extracts to swiftedit_weights/; this project expects weights/
mv swiftedit_weights weights
```

Expected directory: `weights/`

## Inference

```bash
python scripts/run_edit.py \
    --image path/to/person.jpg \
    --source-prompt "woman" \
    --edit-prompt "Taylor Swift" \
    --output result.png
```

### ARaM Scaling Factors

| Parameter | Default | Effect |
|---|---|---|
| `--scale-text` | 1.0 | Prompt alignment inside edit region |
| `--scale-edit` | 0.2 | Source image influence inside edit region |
| `--scale-non-edit` | 1.0 | Background preservation outside edit region |
| `--mask-threshold` | 0.5 | Mask binarization threshold |

## Training (Stage 1)

Stage 1 warms up the garment branch while keeping the generator and inversion network frozen.

### 1. Build Cache

```bash
python scripts/make_null_embedding.py   # ~2 min
python scripts/build_vton_cache.py \
    --output outputs/vton_cache \
    --batch-size 8
```

### 2. Train

```bash
python scripts/train_vton_stage1.py \
    --cache outputs/vton_cache \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --output-dir outputs/vton_stage1
```

### Colab (A100 40 GB)

```python
!pip install -q torch==2.2.1 torchvision==0.17.1
!git clone <your-fork> /content/Fast-VTON
%cd /content/Fast-VTON
!pip install -q -e '.[vton]'

# Download checkpoints and dataset...
!python scripts/train_vton_stage1.py --cache outputs/vton_cache --batch-size 8
```

### Training Config

| Setting | A100 40 GB | A100 80 GB |
|---|---|---|
| Batch size | 8 | 16 |
| Gradient accumulation | 2 | 1 |
| Mixed precision | bf16 | bf16 |
| Max steps | 40,000 | 40,000 |

### Trainable Parameters (~33 M)

| Group | Parameters |
|---|---|
| `prompt_kv` (attn2.to_k/to_v) | 25.56 M |
| `image_projection` | 4.20 M |
| `garment_projection` | ~3.1 M |
| `conv_in` (expanded) | ~26 k |

## Dataset

**VITON-HD** — 11,647 training pairs, 512×384 resolution.

The dataset (`forgeml/viton_hd`) provides `image`, `agnostic`, `cloth`, and `cloth_mask` columns. Masks are constructed via pixel differencing (`|image − agnostic| > threshold`) with morphological cleanup.

## Tests

```bash
pytest                    # Run all tests
pytest -m "not slow"      # Skip tests requiring pretrained weights
```

| Test | What it verifies |
|---|---|
| `test_checkpoint_compatibility` | state_dict keys match `ip_adapter.bin` |
| `test_conv_in_expansion` | Zero-init `conv_in` is neutral |
| `test_masking` | Mask covers correct region |
| `test_mask_controller` | ARaM runs on latent 64×48 |
| `test_config` | DataConfig / Stage1Config validation |

## License

MIT License — Copyright (c) 2026 Lê Vũ Hoàng Tùng

See [LICENSE](LICENSE) for details.
