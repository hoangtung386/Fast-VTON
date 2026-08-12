"""Regenerate ``train_vton_stage1_colab.ipynb``.

The notebook is generated rather than hand-edited so cell content stays diffable and
the JSON stays valid. Run ``python notebooks/_build_notebook.py`` after editing.
"""

import json
from pathlib import Path

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")})


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").split("\n"),
        }
    )


# Add trailing newlines the way nbformat stores them.
def finish(source: list[str]) -> list[str]:
    return [line + "\n" for line in source[:-1]] + [source[-1]]


md("""
# SwiftEdit → Virtual Try-On — Stage 1 trên Colab A100

Notebook này **chỉ điều phối**: mọi logic nằm trong `src/`, ở đây chỉ gọi.
Đọc `docs/VTON_PLAN.md` để hiểu vì sao mỗi bước lại như vậy.

**Luồng chạy**

| Bước | Việc | Thời gian |
|---|---|---|
| 1–3 | Môi trường, repo, checkpoint | ~25 phút |
| 4 | Dữ liệu VITON-HD + **cổng chặn: kiểm tra mask** | ~20 phút |
| 5 | Embedding rỗng cho F_θ | ~2 phút |
| 6 | **Cổng chặn: overfit 8 mẫu** | ~10 phút |
| 7 | Cache đầy đủ 11,647 mẫu | ~30 phút |
| 8 | Train Stage 1 | nhiều giờ |
| 9 | Đóng gói 1 file → Google Drive | ~5 phút |

Hai cổng chặn có dấu ⛔ — sai ở đó thì mọi thứ phía sau vô nghĩa, **đừng chạy tiếp**.

> **Runtime:** Runtime → Change runtime type → **A100 GPU**.
> Colab Pro thường cấp A100 **40 GB**; bản 80 GB không đảm bảo. Cell 1 sẽ tự dò và
> chọn batch size phù hợp.
""")

md("## 1. Kiểm tra GPU")

code("""
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
""")

md("""
## 2. Cài môi trường và lấy code

Sửa `REPO_URL` thành remote git mới của bạn. Nếu chưa push lên đâu cả, đổi
`USE_DRIVE_COPY = True` để copy thẳng từ Drive.
""")

code("""
REPO_URL = "https://github.com/<your-account>/SwiftEdit.git"  # <-- SỬA
USE_DRIVE_COPY = False
DRIVE_REPO_PATH = "/content/drive/MyDrive/SwiftEdit"   # dùng khi USE_DRIVE_COPY = True

from google.colab import drive

drive.mount("/content/drive")
""")

code("""
import shutil
from pathlib import Path

PROJECT = Path("/content/SwiftEdit")

if PROJECT.exists():
    shutil.rmtree(PROJECT)

if USE_DRIVE_COPY:
    shutil.copytree(DRIVE_REPO_PATH, PROJECT)
else:
    !git clone -q {REPO_URL} {PROJECT}

%cd {PROJECT}
!ls src
""")

code("""
# Pin đúng stack đã kiểm chứng. Torch trước, phần còn lại sau, numpy cuối cùng.
!pip install -q torch==2.2.1 torchvision==0.17.1
!pip install -q -e '.[vton]'
!pip install -q numpy==1.26.4

print("\\nKhởi động lại runtime NẾU Colab báo cần, rồi chạy tiếp từ cell dưới.")
""")

code("""
%cd /content/SwiftEdit
import src, torch

print("swiftedit", src.__version__)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
""")

md("""
## 3. Tải checkpoint SwiftEdit (9.1 GB)

Release chính thức gồm 5 part. `wget -c` cho phép chạy lại nếu đứt mạng.

> `stabilityai/stable-diffusion-2-1-base` đã bị gỡ khỏi Hub (404 kể cả có token).
> Code dùng mirror `Manojb/stable-diffusion-2-1-base`, đã verify sha256 khớp bản gốc.
""")

code("""
BASE = "https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0"

for part in "aa ab ac ad ae".split():
    !wget -q -c {BASE}/swiftedit_weights.tar.gz.part-{part}

!cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
!tar zxf swiftedit_weights.tar.gz
!rm -f swiftedit_weights.tar.gz*
!find swiftedit_weights -name '._*' -delete
!mv swiftedit_weights weights          # archive giải nén ra swiftedit_weights/, code chờ weights/

from src.vton import CheckpointConfig

CHECKPOINTS = CheckpointConfig()
CHECKPOINTS.validate()          # ném FileNotFoundError nếu thiếu bất cứ thứ gì
print("checkpoint OK:", CHECKPOINTS.root.resolve())
""")

code("""
# Xác nhận checkpoint nguyên vẹn: 686/686 tensor của G phải bit-identical.
!python scripts/dissect_checkpoints.py --compare-generator
""")

md("""
## 4. ⛔ Cổng chặn 1 — dữ liệu VITON-HD và kiểm tra mask

`forgeml/viton_hd` là 11,647 cặp train chính thức, có sẵn cột `agnostic` nên không cần
chạy human parsing. Nhưng **không có `parse`**, nên mask dựng bằng hiệu ảnh
(`build_agnostic_mask`) — cách này không phụ thuộc bảng label LIP/CIHP vốn hay sai.

**Nhìn kỹ lưới ảnh bên dưới.** Mask (hàng 3) phải phủ thân áo + hai cánh tay,
**không** lấn xuống quần, **không** ăn vào mặt. Coverage lành mạnh: 0.10 – 0.35.
""")

code("""
from datasets import load_dataset

dataset = load_dataset("forgeml/viton_hd", split="train")
print(dataset)
""")

code("""
import matplotlib.pyplot as plt

from src.vton import DataConfig, build_agnostic_mask, mask_coverage

DATA = DataConfig()          # 512x384, ngưỡng 12, kernel 9
print(f"latent {DATA.latent_height}x{DATA.latent_width}, pil size {DATA.pil_size}")

fig, axes = plt.subplots(3, 6, figsize=(16, 9))
for column in range(6):
    sample = dataset[column * 1900]
    mask = build_agnostic_mask(
        sample["image"], sample["agnostic"], DATA.pil_size,
        DATA.mask_diff_threshold, DATA.mask_morph_kernel,
    )
    axes[0, column].imshow(sample["image"].resize(DATA.pil_size))
    axes[1, column].imshow(sample["agnostic"].resize(DATA.pil_size))
    axes[2, column].imshow(mask, cmap="gray")
    axes[0, column].set_title(f"coverage {mask_coverage(mask):.3f}", fontsize=9)
    for row in range(3):
        axes[row, column].axis("off")
plt.tight_layout()
plt.show()
""")

code("""
# Kiểm tra định lượng trên 200 mẫu trước khi cache cả 11k.
import numpy as np

coverages = [
    mask_coverage(build_agnostic_mask(
        dataset[i]["image"], dataset[i]["agnostic"], DATA.pil_size,
        DATA.mask_diff_threshold, DATA.mask_morph_kernel))
    for i in range(0, 2000, 10)
]
coverages = np.array(coverages)
outliers = ((coverages < 0.05) | (coverages > 0.50)).mean()

print(f"coverage: trung vị {np.median(coverages):.3f}, "
      f"khoảng {coverages.min():.3f}–{coverages.max():.3f}")
print(f"tỉ lệ bất thường: {outliers:.1%}")
assert outliers < 0.05, (
    "Quá nhiều mask bất thường. Chỉnh mask_diff_threshold (lấn thì tăng) "
    "hoặc mask_morph_kernel (thủng thì tăng) trong DataConfig rồi chạy lại."
)
print("\\n✅ CỔNG CHẶN 1 ĐẠT")
""")

md("""
## 5. Embedding rỗng cho F_θ

Try-on giao nhánh prompt cho quần áo, nên text encoder chỉ còn một việc: sinh hằng số này.

> Phải lấy từ **`stabilityai/sd-turbo`** — đó là text encoder mà `InverseModel` được
> huấn luyện cùng. Lấy nhầm từ SD 2.1-base sẽ ra tensor trông hợp lệ nhưng làm giảm
> chất lượng inversion một cách âm thầm. Script đã tự cảnh báo nếu bạn đổi `--model`.
""")

code("""
!python scripts/make_null_embedding.py --output outputs/null_embedding.pt

import torch

print("shape:", tuple(torch.load("outputs/null_embedding.pt").shape))
""")

md("""
## 6. ⛔ Cổng chặn 2 — overfit 8 mẫu

Cache 8 mẫu rồi train 400 step. Loss **phải** tụt gần về 0. Không tụt nghĩa là có bug
trong đường ống (mask sai chỗ, token nối nhầm, tham số không nhận gradient) —
**không phải** thiếu dữ liệu. Rẻ hơn nhiều so với phát hiện sau 6 giờ A100.
""")

code("""
!python scripts/build_vton_cache.py \\
    --limit 8 --batch-size 4 --output outputs/smoke_cache
""")

code("""
!python scripts/train_vton_stage1.py \\
    --cache outputs/smoke_cache \\
    --batch-size 4 --gradient-accumulation-steps 1 \\
    --max-steps 400 --log-every 25 \\
    --num-workers 0 \\
    --output-dir outputs/smoke_run
""")

md("""
Đọc dòng log cuối. Cần thấy:

- `trainable parameters: ~33 M` — nếu ra hàng trăm M là đã lỡ mở đóng băng G.
- `loss` giảm đều và về gần 0.

Nếu loss đứng yên, dừng lại và soát: token có tách đúng ở vị trí 257 không, mask latent
có phải nhị phân không, `conv_in` đã mở lên 9 kênh chưa.
""")

code("""
from pathlib import Path

import torch

ckpt = torch.load(sorted(Path("outputs/smoke_run").glob("*.pt"))[-1], map_location="cpu")
trainable = sum(t.numel() for group in ckpt["weights"].values() for t in group.values())
print(f"step {ckpt['step']}, tham số trainable {trainable / 1e6:.2f} M")
assert 25e6 < trainable < 60e6, "số tham số trainable không đúng kỳ vọng ~33 M"
print("\\n✅ CỔNG CHẶN 2 ĐẠT (nếu loss cũng đã hội tụ)")
""")

md("""
## 7. Cache đầy đủ

Mọi module Stage 1 đóng băng — F_θ, VAE, DINOv2, CLIP — đều nhận input cố định, nên chạy
một lần rồi cache. Vòng lặp train sau đó bỏ hẳn được một lượt UNet và ~1.7 GB VRAM.

Cache ghi ra thư mục `.npy` memmap: file DINOv2 6 GB không cần nằm vừa RAM.
""")

code("""
!python scripts/build_vton_cache.py \\
    --output outputs/vton_cache --batch-size 8

!du -sh outputs/vton_cache && ls -la outputs/vton_cache
""")

code("""
from src.vton import CachedVtonDataset

cache = CachedVtonDataset("outputs/vton_cache")
print(f"{len(cache)} mẫu")
for name, tensor in cache[0].items():
    print(f"  {name:20s} {tuple(tensor.shape)}  {tensor.dtype}")
""")

md("""
## 8. Train Stage 1

Checkpoint chỉ lưu ~33 M tensor trainable (≈130 MB) chứ không lưu 1.76 B tham số đóng
băng — quan trọng vì Colab hay đứt phiên và checkpoint phải ghi ra Drive thật nhanh.

**Bị đứt giữa chừng?** Chạy lại cell này với `--resume <đường dẫn checkpoint mới nhất>`.
""")

code("""
import subprocess

VRAM_GB = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
).decode().split()[0]) / 1024

# Bảng 2.7 của VTON_PLAN.md
if VRAM_GB >= 70:
    BATCH_SIZE, ACCUM = 16, 1
elif VRAM_GB >= 38:
    BATCH_SIZE, ACCUM = 8, 2
else:
    BATCH_SIZE, ACCUM = 4, 4
    print("CẢNH BÁO: dưới 38 GB VRAM, batch nhỏ lại — sẽ chậm hơn đáng kể.")

print(f"VRAM {VRAM_GB:.0f} GB -> batch_size={BATCH_SIZE}, accumulation={ACCUM} "
      f"(effective {BATCH_SIZE * ACCUM})")
""")

code("""
RUN_DIR = "/content/drive/MyDrive/vton_stage1"
MAX_STEPS = 40000

!mkdir -p {RUN_DIR}
!python scripts/train_vton_stage1.py \\
    --cache outputs/vton_cache \\
    --batch-size {BATCH_SIZE} \\
    --gradient-accumulation-steps {ACCUM} \\
    --max-steps {MAX_STEPS} \\
    --output-dir {RUN_DIR} \\
    --num-workers 2
""")

code("""
# Chạy lại sau khi Colab ngắt kết nối:
# latest = sorted(Path(RUN_DIR).glob("step_*.pt"))[-1]
# !python scripts/train_vton_stage1.py --cache outputs/vton_cache \\
#     --batch-size {BATCH_SIZE} --gradient-accumulation-steps {ACCUM} \\
#     --max-steps {MAX_STEPS} --output-dir {RUN_DIR} --resume {latest}
""")

md("""
## 9. Kiểm tra G không bị đụng

Nguyên tắc cốt lõi của Stage 1: generator đóng băng. Chỉ `attn2.to_k/to_v` và `conv_in`
được phép đổi. Kiểm chứng bằng chính công cụ đã dùng để chứng minh tác giả đóng băng G.
""")

code("""
!python scripts/dissect_checkpoints.py --compare-generator
""")

md("""
## 10. Đóng gói toàn bộ model thành 1 file → Google Drive

`export_bundle` ghi **một** file chứa mọi module cần cho inference cộng config để dựng
lại chúng: generator (đã mở 9 kênh) + garment encoder + mạng nghịch đảo + VAE + CLIP
vision + embedding rỗng. Máy đích **không cần** tải Hugging Face, **không cần**
`weights/`, chỉ cần file này.

Vì sao là bundle state-dict chứ không phải `torch.save(model)`: pickle một module sống sẽ
ghi lại đường dẫn class của từng submodule, nên file chết ngay khi có ai đổi tên class —
đúng thứ mà refactor làm. Bundle chỉ chứa tensor và config nên sống sót.

Kích thước fp16: khoảng **4.9 GB** (~2.46 B tham số).
""")

code("""
from pathlib import Path

import torch

from src.constants import INPAINTING_LATENT_CHANNELS
from src.models import AuxiliaryModel, InverseModel, IPSBV2Model
from src.vton import CheckpointConfig, GarmentEncoder, Stage1Config, Stage1Trainer

CHECKPOINTS = CheckpointConfig()
config = Stage1Config(output_dir=Path(RUN_DIR), max_steps=MAX_STEPS)

aux_model = AuxiliaryModel(device="cuda", load_text_encoder=False)
generator = IPSBV2Model(
    CHECKPOINTS.generator_dir,
    CHECKPOINTS.ip_adapter_path,
    aux_model,
    device="cuda",
    with_ip_mask_controller=True,
    inpainting_channels=INPAINTING_LATENT_CHANNELS,
)
garment_encoder = GarmentEncoder(CHECKPOINTS.garment_backbone).to("cuda")
inverse_model = InverseModel(CHECKPOINTS.inversion_dir, device="cuda", load_text_encoder=False)

# Nạp trọng số đã train vào bộ khung vừa dựng.
trainer = Stage1Trainer(generator, garment_encoder, config)
trainer.load_checkpoint(sorted(Path(RUN_DIR).glob("*.pt"))[-1])
print("đã nạp checkpoint tại step", trainer.state.step)
""")

code("""
from src.vton import export_bundle

BUNDLE_PATH = "/content/drive/MyDrive/vton_stage1/swiftedit_vton_full.pt"

export_bundle(
    BUNDLE_PATH,
    generator=generator,
    garment_encoder=garment_encoder,
    inverse_model=inverse_model,
    step=trainer.state.step,
    height=config.data.height,
    width=config.data.width,
    include_frozen=True,        # tự chứa: máy đích không cần tải gì thêm
    dtype="fp16",
    null_embedding=torch.load("outputs/null_embedding.pt"),
)
""")

code("""
from src.vton import bundle_summary

summary = bundle_summary(BUNDLE_PATH)
print(f"file: {summary['file_size_bytes'] / 1e9:.2f} GB")
print(f"tổng tham số: {summary['total_parameters'] / 1e6:.1f} M\\n")
for group, count in sorted(summary["parameters_by_group"].items()):
    print(f"  {group:18s} {count / 1e6:9.2f} M")
print("\\nmanifest:")
for key, value in summary["manifest"].items():
    if not key.endswith("_config"):
        print(f"  {key:22s} {value}")
""")

md("""
### Kiểm tra bundle nạp lại được

Chạy trên chính Colab để chắc file không hỏng trước khi mang về server 3090.
""")

code("""
import gc

# Giải phóng VRAM trước khi dựng lại từ bundle.
del generator, garment_encoder, inverse_model, aux_model, trainer
gc.collect()
torch.cuda.empty_cache()

from src.vton import load_bundle

bundle = load_bundle(BUNDLE_PATH, device="cuda")
print("step:", bundle.manifest.step)
print("độ phân giải:", bundle.manifest.height, "x", bundle.manifest.width)
print("kênh conv_in:", bundle.manifest.inpainting_channels)
print("có module đóng băng:", bundle.manifest.includes_frozen)
print("mạng nghịch đảo:", type(bundle.inversion_unet).__name__)
print("VAE:", type(bundle.vae).__name__)
print("\\n✅ Bundle nạp lại thành công")
""")

md("""
## Xong

`swiftedit_vton_full.pt` đã nằm trên Drive. Trên server RTX 3090 chỉ cần:

```python
from src.vton import load_bundle

bundle = load_bundle("swiftedit_vton_full.pt", device="cuda")
```

fp16 toàn bộ chiếm khoảng 3.2 GB VRAM — thoải mái trong 24 GB.

**Việc còn lại (Stage 2, xem mục 2.10 của `docs/VTON_PLAN.md`)**

- Mở F_θ (865.91 M), thêm DISTS + `L_regu` kiểu SDS theo Eq. 8 của paper.
- Bật `DataConfig.horizontal_flip` và cache cả hai chiều.
- **Ablation bỏ F_θ:** thay `inverted_noise` bằng nhiễu Gauss rồi đo lại. Nếu chất lượng
  tụt không đáng kể thì bỏ hẳn F_θ — inference còn một lượt UNet, nhanh gấp đôi. Đây là
  quyết định đáng giá nhất còn lại, và phải đo chứ không đoán.
""")

for cell in cells:
    cell["source"] = finish(cell["source"])

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).parent / "train_vton_stage1_colab.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} with {len(cells)} cells")
