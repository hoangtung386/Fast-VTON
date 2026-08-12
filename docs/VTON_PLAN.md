# SwiftEdit → Virtual Try-On: Phân tích kiến trúc & Kế hoạch Stage 1

> Tài liệu ghi lại toàn bộ phân tích kiến trúc và hướng dẫn sửa code để huấn luyện
> Stage 1 trên Google Colab Pro (A100) với dataset VITON-HD.
>
> Mọi số liệu trong Phần 1 đều lấy từ việc mổ trực tiếp checkpoint, không phải trích paper.

---

# PHẦN 1 — Phân tích kiến trúc

## 1.1 Bằng chứng thực nghiệm từ checkpoint

Repo không phát hành code training, nên mọi kết luận về chiến lược huấn luyện phải suy ra từ
trọng số. Kết quả mổ tĩnh (CPU, đọc header safetensors + mmap):

### G bị đóng băng tuyệt đối

So từng tensor `unet.*` trong `ip_adapter.bin` với `sbv2_0.5/diffusion_pytorch_model.safetensors`:

```
matched by name : 686
bit-identical   : 686      ← không một weight nào thay đổi
differing       : 0
```

Không phải "gần giống" mà **bit-identical 686/686**. Kiểm chứng bằng số học:

```
file ip_adapter.bin = 3,582,997,840 byte
(865.91M UNet + 25.56M IP-adapter + 4.20M image_proj) × 4 byte = 3.583 GB   ✓ khớp
```

**Kết luận: tác giả chưa từng cập nhật một tham số nào của generator G.** Luận điểm
"không được full fine-tune G" có bằng chứng thực nghiệm, không còn là suy diễn từ paper.

Hệ quả phụ: `sbv2_0.5/` (3.3 GB) bị `load_state_dict` ghi đè bằng chính nó tại `models.py:199`
— 3.3 GB trong 9.8 GB checkpoint là dữ liệu trùng lặp.

### F_θ trôi rất xa khỏi điểm khởi tạo

```
tensor count      G=686   F=686   same key set=True
identical tensors : 0/686
differing tensors : 686/686
largest relative drift: 0.984 at down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.weight
```

F_θ init từ G nhưng đã học lại gần như hoàn toàn cách sử dụng điều kiện `c_y`. Độ trôi lớn nhất
nằm ở lớp **cross-attention** — chi tiết này quyết định lập luận ở mục 1.3.

Ghi chú nhỏ: thư mục tên `inverse_ckpt-120k` nhưng config EMA ghi `optimization_step: 110000`.

### Sức chứa hai nhánh điều kiện

| Nhánh | Nguồn | Số token | Tham số K/V |
|---|---|---|---|
| Text `c_y` | CLIP text, 77 × 1024 | **77** | 25.56 M (16 site `attn2`) |
| IP `c_x` | `image_proj.proj: 1024→4096` | **4** | 25.56 M (`to_k_ip`/`to_v_ip`) |

Giới hạn 4 token thuộc nhánh **IP (ảnh người)**, không phải nhánh text. Quần áo đi vào nhánh text
đã có sẵn 77 slot, và cross-attention **không có positional embedding phía K/V** nên nhận độ dài
tùy ý — đưa 257 patch token DINOv2 vào thẳng được, **không cần sửa kiến trúc**.

### Ngân sách tham số

| Thành phần | Tham số |
|---|---|
| G (UNet generator) | 865.91 M |
| F_θ (inversion net) | 865.91 M |
| `image_proj_model` | 4.20 M |
| `to_k_ip` / `to_v_ip` | 25.56 M |
| `attn2.to_k/to_v` trong G (W_y) | 25.56 M |
| **Full fine-tune (G+F)** | **1757 M** |
| **Adapter-only (proj + W_y)** | **29.76 M** |

### Cấu hình kiến trúc

```
cross_attention_dim  = 1024      in_channels  = 4      sample_size = 64  (512px)
block_out_channels   = [320, 640, 1280, 1280]          out_channels = 4
attention_head_dim   = [5, 10, 20, 20]                 16 site attn2
```

`out_channels = 4` ⇒ nhánh `if model_pred.shape[1] == noise.shape[1] * 2` trong `models.py`
không bao giờ kích hoạt.

## 1.2 Ánh xạ SwiftEdit → VTON

ARaM (Eq. 9) phân vai hai đường điều kiện rất rõ:

```
h_l = s_y       · M     · Attn(Q, K_y, V_y)   ← text: chỉ tác động TRONG vùng edit
    + s_edit    · M     · Attn(Q, K_x, V_x)   ← ảnh nguồn, trong vùng edit
    + s_non-edit· (1−M) · Attn(Q, K_x, V_x)   ← ảnh nguồn, giữ nguyên nền
```

| SwiftEdit | VTON |
|---|---|
| `c_x` (IP) = ảnh nguồn | ảnh người **agnostic** → giữ mặt, da, nền |
| `c_y` (text) = mô tả thay đổi | **ảnh quần áo** (DINOv2) → nội dung đổ vào mask |
| `M` = mask tự sinh từ 2 prompt | agnostic mask dựng sẵn |

Số hạng `s_non-edit·(1−M)` chính là cơ chế bảo toàn nền mà VTON cần.

## 1.3 Vì sao không đóng băng F_θ theo cách hiển nhiên

Phương án "stage 1 đóng băng cả UNet lẫn F_θ, chỉ train projection" có một lỗ hổng:
**F_θ cũng tiêu thụ `c_y`**. Vì cross-attn của F_θ đã trôi tới 98% khỏi G, một projection duy nhất
`DINO → 1024` sẽ phải đồng thời làm hài lòng hai bộ cross-attention đã rất khác nhau.

**Cách gỡ:** chỉ đưa quần áo vào `G^IP`, còn F_θ cho ăn **embedding rỗng**. README ghi source prompt
là optional nên empty prompt nằm trong phân phối F_θ đã quen. Precompute một lần embedding của
chuỗi rỗng, lưu thành hằng số, rồi **xóa hẳn text encoder**.

> ⚠️ Phải precompute từ text encoder của **`stabilityai/sd-turbo`** (cái mà `InverseModel` dùng),
> không phải của SD 2.1-base. F_θ được train với sd-turbo text encoder.

## 1.4 Hai cạm bẫy dễ bỏ sót

**(a) Nhánh IP sẽ chống lại bạn.** Số hạng `s_edit·M·Attn(Q,K_x,V_x)` bơm đặc trưng ảnh nguồn vào
*trong* vùng edit — mà vùng đó chứa **quần áo cũ**. Không ngẫu nhiên `infer.py:35` đặt
`scale_edit=0.2`. → Cho nhánh IP ăn **ảnh agnostic**, không phải ảnh người gốc.

**(b) Nhiễu nghịch đảo cũng mang theo áo cũ.** `input_sb = alpha_t·latents + sigma_t·ε̂`, mà ε̂ được
huấn luyện để **tái tạo** ảnh đầu vào. Thay áo dài tay bằng áo ba lỗ là đổi cấu trúc — ε̂ sẽ cưỡng lại.
→ **Nghịch đảo ảnh agnostic**, và thêm điều kiện inpainting: mở `conv_in` từ 4 lên 9 kênh
(4 latent + 4 masked-latent + 1 mask), khởi tạo 5 kênh mới bằng 0.

Zero-init đảm bảo tại bước 0, G hành xử **y hệt** bản gốc. Đây là ngoại lệ duy nhất cho nguyên tắc
đóng băng G (~26k tham số mới).

## 1.5 Dataset và độ phân giải

**VITON-HD** trước (11,647 cặp train, chính diện, thân trên) — sạch, là benchmark chuẩn, một
category nên ít biến thiên. **DressCode** (53,795 cặp, 3 category) để mở rộng sau.

Độ phân giải: **512×384**. Đây là setting low-res chuẩn của VITON-HD, đúng tỉ lệ 3:4, và UNet conv
xử lý được mọi kích thước chia hết cho 8. Đừng letterbox về vuông — phí 25% pixel.

Ràng buộc code: 3 dòng `H = W = int(np.sqrt(...))` tại `src/mask_ip_controller.py:29`, `:58`, `:119`
giả định latent vuông. **Chỉ ảnh hưởng inference** (ARaM là cơ chế inference-time), Stage 1 training
không dùng `MaskController` nên chưa cần sửa.

## 1.6 Kiến trúc chốt

| Thành phần | Xử lý |
|---|---|
| G (SBv2) | Đóng băng, trừ `conv_in` (4→9, zero-init) và `attn2.to_k/to_v` |
| Text encoder | **Xóa**, thay bằng embedding rỗng precompute cho F_θ |
| Nhánh `c_y` | DINOv2 patch token của ảnh quần áo + projection (257 token) |
| Nhánh `c_x` | Ảnh **agnostic** qua CLIP image encoder (4 token) |
| Mask | Dựng từ dataset, bỏ self-guided mask |
| F_θ | Stage 1 đóng băng + null embedding; Stage 2 mới mở |

## 1.7 Điểm quyết định cần đo, không đoán

**Inference hiện tại là 2 lượt UNet, không phải 1.** "One-step" nói về số bước khử nhiễu, còn chi phí
thực = F_θ (865.91M) + G (865.91M).

Nếu đã có mask tường minh và điều kiện inpainting 9 kênh, **F_θ có thể trở nên thừa**: khởi tạo từ
nhiễu Gauss, cấu trúc người đi vào qua kênh conv, composite lại vùng ngoài mask. Bỏ F_θ =
**nhanh gấp đôi**, khỏi train Stage 2, khỏi lo `L_regu` và chuyện ε̂ trôi.

Cách chốt: dựng bản có F_θ trước (rủi ro thấp, kế thừa nguyên SwiftEdit), rồi **ablate** — thay ε̂
bằng nhiễu ngẫu nhiên, đo lại. Nếu chất lượng tụt không đáng kể, bỏ F_θ.

---
# PHẦN 2 — Stage 1 trên Colab A100 + VITON-HD

> Phần này đã được **hiện thực hoá thành code** trong `src/vton/`. Các đoạn
> snippet trong bản nháp trước đã được thay bằng tham chiếu tới module thật, để tài liệu
> không trôi khỏi code. Ba chỗ bản nháp sai đã được sửa, ghi rõ ở mục 2.8.

## 2.1 Bản đồ code

| Việc | Module | Script chạy |
|---|---|---|
| Dựng mask từ cặp person/agnostic | `vton/masking.py` | — |
| Encode quần áo bằng DINOv2 | `vton/garment_encoder.py` | — |
| Mở `conv_in` 4→9 kênh | `models/generator.py::expand_conv_in` | — |
| Đóng băng có chọn lọc | `vton/freezing.py` | — |
| Embedding rỗng cho F_θ | — | `scripts/make_null_embedding.py` |
| Cache đặc trưng | `vton/precompute.py` | `scripts/build_vton_cache.py` |
| Dataset trên cache | `vton/data.py` | — |
| Vòng lặp huấn luyện | `vton/trainer.py` | `scripts/train_vton_stage1.py` |
| Kiểm tra checkpoint | — | `scripts/dissect_checkpoints.py` |

Cấu hình tập trung ở `vton/config.py` (`DataConfig`, `CheckpointConfig`, `Stage1Config`),
đều là dataclass đóng băng có validate trong `__post_init__`.

## 2.2 Chiến lược: precompute mọi thứ đóng băng

Stage 1 giữ nguyên F_θ, VAE, DINOv2 backbone và CLIP vision tower — input của chúng cố
định, nên tính trước một lần rồi cache. Vòng lặp huấn luyện khi đó chỉ còn projection + G:
bỏ được một lượt UNet và ~1.7 GB VRAM mỗi step.

Cache là một thư mục `.npy` memmap (`vton/precompute.py`), không phải một file `.pt` khổng
lồ — nhờ vậy file DINOv2 6 GB không cần nằm vừa RAM và nhiều dataloader worker dùng chung
page cache.

| Mảng | Kích thước (11,647 mẫu, fp16) |
|---|---|
| `z_person`, `z_agnostic`, `inverted_noise` | 3 × 273 MB |
| `mask_latent` | 68 MB |
| `clip_image_embeds` | 23 MB |
| `garment_features` (tuỳ chọn) | 6.1 GB |

Đánh đổi: hflip phải cache cả hai chiều. `DataConfig.horizontal_flip` mặc định `False` —
bật lại ở Stage 2.

## 2.3 Colab: cài môi trường

```python
# Cell 1
!pip install -q torch==2.2.1 torchvision==0.17.1     # cu121 mặc định, A100 chạy tốt
!git clone <your-fork> /content/SwiftEdit
%cd /content/SwiftEdit
!pip install -q -e '.[vton]'
```

```python
# Cell 2 — checkpoint + Drive
from google.colab import drive; drive.mount('/content/drive')
for p in "aa ab ac ad ae".split():
    !wget -q -c https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-{p}
!cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz && tar zxf swiftedit_weights.tar.gz
!rm -f swiftedit_weights.tar.gz*
!find swiftedit_weights -name '._*' -delete
!mv swiftedit_weights weights
```

**Bẫy môi trường Colab: phải gỡ `peft`.** Colab cài sẵn peft đời mới, còn ta ghim
`transformers==4.37.2` cho khớp checkpoint. diffusers tính
`USE_PEFT_BACKEND = _required_peft_version and _required_transformers_version` lúc import;
thấy peft có mặt là bật, rồi giữa lượt UNet gọi `scale_lora_layers` → `import peft` →
`ImportError: cannot import name 'EncoderDecoderCache'` (lớp đó chỉ có từ transformers
~4.45). Dự án không dùng peft ở đâu cả, nên `pip uninstall -y peft` là xong — và khi đó môi
trường Colab khớp đúng bản local vốn chạy được (local không có peft, `USE_PEFT_BACKEND`
bằng `False`, nhánh kia không bao giờ chạy).

Notebook assert `not USE_PEFT_BACKEND` ngay ở cell kiểm môi trường, vì nếu để lọt thì lỗi
chỉ nổ sau khi job cache đã chạy một lúc, với traceback toàn nội bộ peft không hé lộ gì về
nguyên nhân.

## 2.4 Dữ liệu VITON-HD

`forgeml/viton_hd` chứa đúng 11,647 cặp train chính thức (3.4 GB) với các cột
`image`, `agnostic`, `cloth`, `cloth_mask`, `pose`, `caption`. Có sẵn `agnostic` nên không
cần chạy SCHP, nhưng **không có `parse`** — vì vậy mask được dựng bằng hiệu ảnh.

`build_agnostic_mask` lấy `|image − agnostic| > threshold` rồi morphology close/open và giữ
thành phần liên thông lớn nhất. Cách này không phụ thuộc bảng label LIP/CIHP, vốn khác nhau
giữa các bản parsing và là nguồn sai âm thầm phổ biến.

**Bắt buộc kiểm tra bằng mắt trước khi cache 11k mẫu:**

```python
import matplotlib.pyplot as plt
from datasets import load_dataset
from src.vton import build_agnostic_mask, mask_coverage

ds = load_dataset("forgeml/viton_hd", split="train")
fig, ax = plt.subplots(3, 6, figsize=(16, 9))
for i in range(6):
    s = ds[i * 977]
    m = build_agnostic_mask(s["image"], s["agnostic"], (384, 512))
    print(f"coverage {mask_coverage(m):.3f}")     # đo được: trung vị 0.35, dải 0.18–0.63
    ax[0, i].imshow(s["image"].resize((384, 512)));    ax[0, i].axis("off")
    ax[1, i].imshow(s["agnostic"].resize((384, 512))); ax[1, i].axis("off")
    ax[2, i].imshow(m, cmap="gray");                   ax[2, i].axis("off")
```

Mask phải phủ thân áo + hai cánh tay, không lấn xuống quần, không ăn vào mặt.

**Đừng đánh giá bằng coverage.** Bản nháp đặt dải lành mạnh 0.10–0.35 và khuyên tăng
`mask_diff_threshold` khi mask lấn. Cả hai đều sai, và đo mới ra:

- Coverage thật có trung vị **0.35**, p95 **0.54**, tối đa **0.63** (200 mẫu). Cột
  `agnostic` là **agnostic-v3.2**: nó sơn một vùng bao rộng — áo, hai tay, tóc xoã vai,
  cộng một quầng nền quanh silhouette. Quầng nền đó chạy dọc xuống hai bên đùi, nên vừa
  đẩy coverage vừa đẩy mép dưới lên cao mà **không** hề lấn vào quần.
- Quét `mask_diff_threshold` từ 12 lên 35 chỉ đổi trung vị 0.347 → 0.341. Chênh lệch
  person/agnostic trong vùng mask lớn hơn 35 rất nhiều vì đó là vùng bị sơn đè thật, nên
  ngưỡng gần như không có tác dụng. Nó là nút sai.

Phép đo phân biệt được "mask to nhưng đúng" với "mask lấn quần" là vùng quần thật —
1/3 giữa của 1/4 khung dưới — chỉ bị phủ **8% ở trung vị**. Kèm theo
`mask_vertical_extent` cho mép trên (trung vị 0.20; dưới 0.05 là đang ăn vào đầu).
Cổng chặn ở notebook kiểm đúng ba đại lượng đó thay vì coverage đơn thuần.

Còn tồn tại, ghi để Stage 2 biết: khoảng **17%** mẫu có mask phủ quá nửa vùng quần (áo
dài, váy), nên model phải bịa lại phần trên của quần. Mọi paper VITON-HD đều sống chung
với đặc tính này của agnostic-v3.2.

Split test không có ở mirror này — lấy từ `SaffalPoosh/VITON-HD-test` (2,032 cặp).

## 2.5 Chạy Stage 1

```bash
python scripts/make_null_embedding.py                       # ~2 phút
python scripts/build_vton_cache.py --output outputs/vton_cache --batch-size 8
python scripts/train_vton_stage1.py --cache outputs/vton_cache \
       --batch-size 8 --gradient-accumulation-steps 2 --output-dir /content/drive/MyDrive/vton_stage1
```

Tham số nào cũng override được qua CLI; `--resume` nhận đường dẫn checkpoint. Checkpoint chỉ
lưu tensor trainable (~33 M, khoảng 130 MB) chứ không lưu 1.76 B tham số đóng băng — quan
trọng trên Colab vì phiên có thể đứt bất cứ lúc nào.

## 2.6 Nhóm tham số được mở khoá

`TrainableGroups` trong `vton/freezing.py`:

| Nhóm | Mặc định | Tham số |
|---|---|---|
| `prompt_kv` — `attn2.to_k/to_v` (W_y) | bật | 25.56 M |
| `image_projection` — `image_proj_model` | bật | 4.20 M |
| `garment_projection` | bật | ~3.1 M |
| `conv_in` (đã mở rộng) | bật | ~26 k |
| `image_kv` — `to_k_ip/to_v_ip` | **tắt** | 25.56 M |

Tổng mặc định ≈ **33 M / 1757 M**. `image_kv` là knob đầu tiên nên thử nếu identity giữ kém
(`--train-image-kv`).

## 2.7 Cấu hình theo VRAM

Colab Pro thường cấp **A100 40 GB**, không đảm bảo bản 80 GB. Kiểm tra bằng `!nvidia-smi`:

| | A100 40 GB | A100 80 GB |
|---|---|---|
| `--batch-size` | 8 | 16 |
| `--gradient-accumulation-steps` | 2 | 1 |
| gradient checkpointing | bật | bật |
| precision | bf16 | bf16 |
| `--mixed-precision` | `auto` | `auto` |
| cache DINO feats | tuỳ (6.1 GB đĩa) | nên bật |

```
G đóng băng bf16                    1.7 GB
trainable 33 M (fp32 + Adam ×2)     0.53 GB
activations (bs=8, 512×384, ckpt)   ~12 GB
──────────────────────────────────────────
tổng                                ~15 GB
```

Inference trên RTX 3090 24 GB: fp16 toàn bộ (F_θ + G + VAE + DINOv2) ≈ 3.2 GB.

**bf16 không chạy trên T4.** `torch.cuda.is_bf16_supported()` đòi compute capability ≥ 8.0
(Ampere); T4 là 7.5. Và `torch.autocast` chỉ kiểm điều đó ở **step đầu tiên**, nên ép bf16
trên T4 làm job chết sau khi đã nạp xong 1.7 B tham số. `resolve_mixed_precision` trong
`vton/trainer.py` giải quyết trước lúc dựng `Stage1Config`: `auto` lấy bf16 nếu có, fp16 nếu
không, và `bf16` tường minh trên card không hỗ trợ thì báo lỗi ngay ở dòng lệnh. Đây là
kiểu bug "phần cứng không hỗ trợ" mà chỉ mất vài giây để bắt nếu kiểm đúng chỗ, và vài phút
nếu để autocast tự phát hiện.

T4 16 GB chạy đủ hai cổng chặn (mask, overfit 8 mẫu) với `--batch-size 4`, hữu ích để soát
đường ống mà không đốt quota A100.

## 2.8 Bốn chỗ bản nháp sai, đã sửa khi hiện thực hoá

**(a) Không được cache `ip_tokens`.** Bản nháp cache đầu ra của `image_proj_model`, nhưng
module này *nằm trong nhóm trainable* — cache sẽ hỏng ngay sau step đầu tiên. Code cache
`clip_image_embeds` (đầu ra CLIP, đóng băng) và chạy projection trong vòng lặp.

**(b) `GradScaler` vô nghĩa với bf16.** Bản nháp dùng `GradScaler` cùng `autocast(bf16)`.
bf16 có dải mũ như fp32 nên không cần scale; `trainer.py` chỉ bật scaler khi
`mixed_precision == "fp16"`.

**(c) Không cần sửa attention processor, và 4 token là của nhánh IP.** Bản nháp đề xuất nâng
`num_tokens` 4→256 cho quần áo. Sai chỗ: giới hạn 4 token thuộc nhánh **IP (ảnh người)**.
Quần áo đi vào nhánh prompt, nơi cross-attention không có positional embedding phía K/V nên
nhận độ dài tuỳ ý — 257 token DINOv2 vào thẳng, `num_tokens` giữ nguyên 4.

**(d) Tiền xử lý ảnh quần áo cắt mất cổ và gấu.** Processor mặc định của DINOv2 resize cạnh
ngắn về 256 rồi **center-crop 224**. Ảnh áo VITON-HD là 768×1024, nên 34% chiều cao bị vứt —
kèm theo đường cổ, gấu áo và hai đầu tay. Đó đúng là những đặc trưng cấu trúc mà try-on phải
tái tạo, và không encoder nào suy ra được thứ nó chưa từng nhìn thấy.

Đã sửa: `pad_to_square` letterbox nền trắng (khớp backdrop VITON-HD) rồi tắt center-crop.
Đánh đổi là độ phân giải hữu dụng trên áo giảm ~25% vì có thêm viền, đổi lấy việc giữ trọn
món đồ — rẻ hơn nhiều so với mất hẳn thông tin.

Cùng lúc đó, `DataConfig.garment_resolution` là **nút chết**: `cache_specs` tính số token từ
nó nhưng processor vẫn giữ 224 của riêng mình, nên đổi config chỉ làm cache lệch shape.
`image_processor(resolution=...)` giờ nhận tham số này.

### Độ phân giải nhánh quần áo — nút đáng vặn nhất

Ở 224 px với patch 14, lưới chỉ **16×16**: mỗi token phủ ~24 px của ảnh áo rộng 384, nên một
logo ngực chiếm 1–2 token. Nút thắt nằm ở **patch embedding**, không nằm ở trọng số.

| `garment_resolution` | Lưới | Token | Cache fp16 | 1 patch phủ |
|---|---|---|---|---|
| 224 (mặc định) | 16×16 | 257 | 6.1 GB | 24 px |
| 336 | 24×24 | 577 | 13.8 GB | 16 px |
| 448 | 32×32 | 1025 | 24.4 GB | 12 px |

DINOv2 nội suy positional embedding nên mọi bội của 14 đều chạy. Giá phải trả: độ dài K/V
nhánh prompt tăng theo, và cache phình. Ở 448 nên dùng `--skip-garment-features` và chạy
backbone (vẫn đóng băng) trong vòng lặp thay vì giữ file 24 GB.

**Đừng mở đóng băng DINOv2 (302 M) ở Stage 1.** Nó phá đúng tiền đề của mục 2.2 — cache
`garment_features` hỏng ngay sau step đầu, y hệt cái bẫy `ip_tokens` ở mục (a) nhưng cao hơn
một tầng. Thêm nữa: 302 M tham số học từ 11,647 ảnh, với gradient chỉ tới được qua
cross-attention của G, là công thức phá hỏng đặc trưng pretrain. Chi tiết nhỏ nhưng nói lên
nhiều: `proj[-1]` khởi tạo bằng 0, nên gradient chảy ngược vào backbone **đúng bằng 0** ở
step 0 và bị bóp nghẹt một thời gian sau đó.

## 2.9 Checklist xác minh trước khi chạy dài

Bốn mục đầu đã được tự động hoá thành test (`pytest`, chạy CPU, không cần checkpoint):

| # | Kiểm tra | Cách chạy |
|---|---|---|
| 1 | Key `state_dict` khớp `ip_adapter.bin` | `pytest tests/test_checkpoint_compatibility.py` |
| 2 | Zero-init `conv_in` thực sự trung tính | `pytest tests/test_conv_in_expansion.py` |
| 3 | Mask dựng đúng vùng | `pytest tests/test_masking.py` |
| 4 | ARaM chạy được latent 64×48 | `pytest tests/test_mask_controller.py` |
| 5 | Mask thật trông đúng | lưới ảnh ở mục 2.4 ★ |
| 6 | Chỉ ~33 M được mở khoá | dòng log `trainable parameters` lúc khởi động |
| 7 | G chưa bị đụng sau training | `python scripts/dissect_checkpoints.py --compare-generator` |
| 8 | Overfit 8 mẫu về loss ~0 | `--max-steps 500` trên cache 8 mẫu ★ |
| 9 | `conv_in.weight[:, 4:]` khác 0 sau training | nếu vẫn bằng 0 thì kênh inpainting không nhận gradient |

Hai mốc ★ là cổng chặn: đừng đi tiếp nếu chưa qua.

```bash
# Cổng chặn #8: cache nhỏ rồi overfit
python scripts/build_vton_cache.py --limit 8 --output outputs/smoke_cache
python scripts/train_vton_stage1.py --cache outputs/smoke_cache \
       --batch-size 4 --gradient-accumulation-steps 1 --max-steps 500 --log-every 50
```

## 2.10 Việc còn lại cho Stage 2

- Mở F_θ (865.91 M), thêm `L_perceptual` (DISTS) + `L_regu` kiểu SDS theo Eq. 8 của paper.
- Bật `DataConfig.horizontal_flip` và cache cả hai chiều.
- Ablation bỏ F_θ (mục 1.7): thay `inverted_noise` bằng nhiễu Gauss, đo lại. Nếu chất lượng
  tụt không đáng kể thì bỏ F_θ và có VTON một-lượt-UNet, nhanh gấp đôi.
