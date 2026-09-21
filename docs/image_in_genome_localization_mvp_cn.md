# 基因组内图像检测、定位与恢复 MVP

## 1. 本阶段解决什么

本模块严格对应导师提出的第一阶段：在一段长度固定、位置未知的自然 DNA 中，判断是否连续插入了一张图像；若存在，给出图像 DNA 的精确起止位置，并按固定格式还原图像。

本阶段故意不加入压缩、标识符、纠错码、替换、插入或删除。它先回答最基础的问题：在没有序列扰动时，模型能否同时完成高准确率二分类和精确定位。格式也不是未知的，而是预先固定；分辨率、通道、位宽的推断属于下一阶段。

固定数据契约：

| 字段 | MVP 值 |
|---|---:|
| 图像 | 灰度、单通道、8 bit、128 x 128 |
| 预处理 | 保持比例缩放后中心裁剪，不拉伸、不补黑边 |
| 像素顺序 | row-major 原始像素，不压缩 |
| DNA 映射 | 每字节 4 个碱基，每碱基 2 bit；A=00、C=01、G=10、T=11 |
| 图像字节数 | 16,384 bytes |
| 图像 DNA 长度 | 65,536 bases |
| 观察窗口 | 131,072 bases |
| 插入方式 | 正样本中连续插入，起点均匀随机；负样本保持自然序列 |

模型采用分层长序列结构：四级 stride-4 CNN 将 131,072 个碱基压缩成 512 个 256-base bin，再用膨胀卷积或可选 Transformer 建模上下文。定位头分别预测起始 bin 和 bin 内 0-255 偏移，因此最终仍是单碱基精度。另有存在性头和粗分割头。

## 2. 已实现的代码

- `genome_mining/image_payload.py`：固定灰度图预处理、像素和 DNA 双向无损映射、像素指标。
- `genome_mining/image_dataset.py`：真实 FASTA 背景、图像随机插入、NumPy 内存映射数据集和泄漏审计。
- `genome_mining/image_torchdata.py`：按行读取长序列，不把全部样本载入内存。
- `genome_mining/models/image_locator.py`：分层定位器、联合损失和精确坐标解码。
- `genome_mining/image_training.py`：训练、最佳 checkpoint、测试预测、图像恢复和指标输出。

统一命令：

```bash
python -m genome_mining build-image-dataset --help
python -m genome_mining audit-image-dataset --help
python -m genome_mining train-image-locator --help
python -m genome_mining evaluate-image-locator --help
```

## 3. 服务器需要的文件

代码至少上传以下内容，并保持目录结构：

```text
genome_mining/
requirements-genome-v12.txt
environment-genome-v12-a800.yml
```

真实数据需要：

1. 多个不同个体或供体的未压缩 FASTA。训练、验证、测试供体必须互斥。
2. 足量且不重复的图片。正样本一张图只使用一次；系统还会用 SHA-256 拦截改名后的重复图片。
3. 一个 genome manifest CSV。GFF/GTF 和 `.fai` 在本 MVP 中不是必需文件。

单个 CHM13 只能用于数据构建冒烟测试，不能同时充当 train/val/test。否则模型会记住相同装配背景，测试结果没有说服力。正式实验应使用不同个体的高质量人类装配；训练集、验证集和测试集按供体划分，而不是把同一 FASTA 切片后随机划分。

## 4. A800 GPU3 环境

在项目根目录执行：

```bash
conda env create -f environment-genome-v12-a800.yml
conda activate genome-mining-v12
python -m pip install -r requirements-genome-v12.txt
```

环境已存在时只需补齐本次新增的 Pillow：

```bash
conda activate genome-mining-v12
conda install -c conda-forge pillow -y
```

验证环境并固定物理 GPU3：

```bash
export CUDA_VISIBLE_DEVICES=3

python - <<'PY'
import numpy as np
import torch
from PIL import Image
print('numpy:', np.__version__)
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('visible GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('logical cuda:0 =', torch.cuda.get_device_name(0))
PY
```

设置 `CUDA_VISIBLE_DEVICES=3` 后，程序中的 `cuda` 或 `cuda:0` 就是物理 GPU3，不要再写 `cuda:3`。

## 5. 准备供体 manifest

例如保存为 `data/manifests/t2t_donors.csv`：

```csv
donor_id,assembly_id,fasta,split
HG001,HG001_hap1,/data/t2t/HG001_hap1.fna,train
HG002,HG002_hap1,/data/t2t/HG002_hap1.fna,train
HG003,HG003_hap1,/data/t2t/HG003_hap1.fna,train
HG004,HG004_hap1,/data/t2t/HG004_hap1.fna,val
HG005,HG005_hap1,/data/t2t/HG005_hap1.fna,test
```

如果同一个人的 hap1 和 hap2 都使用，它们必须放在同一个 split，并使用相同 `donor_id`。输入必须是未压缩 `.fna/.fa/.fasta`；只有 `.gz` 时先解压：

```bash
gzip -dk /data/t2t/sample.fna.gz
```

构建器从每个装配的全体非重叠有效窗口中做可复现 reservoir 抽样，不会只取 chr1 开头。窗口中的 A/C/G/T 比例默认至少 95%。

## 6. 准备图片

最简单的方式是直接给图片根目录，程序会递归寻找 PNG/JPEG/BMP/TIFF/WebP，并自动做无重复 split 分配：

```text
/data/imagenet/
  n01440764/*.JPEG
  n01443537/*.JPEG
  ...
```

父目录名会保存为 `class_id`，后续可评估恢复图像对 ImageNet 分类的影响。若要显式控制图片 split，使用 CSV：

```csv
image_id,path,class_id,split
img000001,/data/imagenet/n01440764/a.JPEG,n01440764,train
img000002,/data/imagenet/n01443537/b.JPEG,n01443537,val
img000003,/data/imagenet/n01514668/c.JPEG,n01514668,test
```

正样本比例为 0.5 时，10,000 个总样本至少需要 5,000 张内容不同的图片。

## 7. 单个 CHM13 冒烟测试

你已有：

```text
/public/home/lifex/jiangjr/biolearning/database/data/raw/ncbi/human_t2t_chm13v2/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna
```

先只构建 train 数据，验证数据格式和磁盘写入，不训练、不报告泛化指标：

```bash
mkdir -p data/manifests artifacts/image_locator

cat > data/manifests/chm13_smoke.csv <<'CSV'
donor_id,assembly_id,fasta,split
CHM13,T2T-CHM13v2.0,/public/home/lifex/jiangjr/biolearning/database/data/raw/ncbi/human_t2t_chm13v2/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna,train
CSV

python -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/chm13_smoke.csv \
  --image-root /data/imagenet \
  --out artifacts/image_locator/chm13_smoke \
  --n-train 20 \
  --n-val 0 \
  --n-test 0 \
  --positive-fraction 0.5 \
  --image-size 128 \
  --window-length 131072 \
  --seed 42

python -m genome_mining audit-image-dataset \
  --dataset artifacts/image_locator/chm13_smoke \
  --out artifacts/image_locator/chm13_smoke/audit.json
```

审计命令返回码为 0 且 `passed=true` 才能继续。该 smoke 数据没有 val/test，不能用于 `train-image-locator`。

## 8. 正式构建、训练与评估

```bash
python -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/t2t_donors.csv \
  --image-root /data/imagenet \
  --out artifacts/image_locator/t2t_mvp \
  --n-train 6000 \
  --n-val 1500 \
  --n-test 2500 \
  --positive-fraction 0.5 \
  --image-size 128 \
  --window-length 131072 \
  --min-acgt-fraction 0.95 \
  --seed 42

python -m genome_mining audit-image-dataset \
  --dataset artifacts/image_locator/t2t_mvp \
  --out artifacts/image_locator/t2t_mvp/audit.json
```

10,000 个样本的 `sequences.npy` 约 1.31 GB，因为每个碱基以一个 `uint8` 保存。数据构建在 CPU 上运行；训练使用 GPU3：

```bash
export CUDA_VISIBLE_DEVICES=3

python -m genome_mining train-image-locator \
  --dataset artifacts/image_locator/t2t_mvp \
  --out artifacts/image_locator/t2t_mvp_dilated \
  --epochs 20 \
  --batch-size 16 \
  --lr 2e-4 \
  --d-model 128 \
  --downsample-stride 256 \
  --context dilated_cnn \
  --context-layers 6 \
  --workers 4 \
  --device cuda \
  --amp \
  --seed 42
```

如果显存充足可把 batch size 提到 32；出现 OOM 时降到 8 或 4。训练结束会自动在 test split 上评估并保存最多 25 组参考/恢复图像。也可单独重跑：

```bash
python -m genome_mining evaluate-image-locator \
  --dataset artifacts/image_locator/t2t_mvp \
  --model artifacts/image_locator/t2t_mvp_dilated/model.pt \
  --out artifacts/image_locator/t2t_mvp_dilated/test_recheck \
  --split test \
  --batch-size 16 \
  --workers 4 \
  --device cuda \
  --save-images \
  --max-saved-images 100
```

## 9. 输出与成功标准

数据目录：

```text
sequences.npy
metadata.csv
dataset_manifest.json
audit.json
```

训练目录：

```text
model.pt
train_log.json
run_summary.json
test_evaluation/metrics.json
test_evaluation/predictions.csv
test_evaluation/image_recovery.csv
test_evaluation/recovered_images/
```

重点指标：

- `presence_detection.auroc/aupr/f1`：有无图像的二分类。
- `localization.start_mae_bases` 和 `start_exact_rate`：起点误差和单碱基精确率。
- `base_phase_accuracy`：起点是否保持正确的 4-base 字节相位。
- `mean_fixed_span_iou`：固定长度预测区间与真值区间的 IoU。
- `joint_detect_and_exact_start_rate`：同时检出且起点完全正确。
- `image_recovery.exact_image_rate`：恢复的 16,384 个像素字节是否全部一致。
- `pixel_mae/PSNR/ssim_global`：定位略有偏差时的带噪恢复质量。

第一阶段不应只报二分类准确率。建议最低验收是 test donor 完全独立、图像完全独立，二分类 AUROC 很高，并且 `joint_detect_and_exact_start_rate` 与 `exact_image_rate` 达到可重复的高水平。

## 10. 下一阶段顺序

1. 固定格式下先加入低比例单碱基替换，定位指标与图像噪声指标分开报告。
2. 在载荷层加入 identifier、长度、通道、位宽、分辨率和编码方式字段，再训练字段解析头。
3. 加入短 indel，并使用同步标记、分块重同步与 `genome_survival_sim` 已有 Reed-Solomon/fountain code。
4. 从连续载荷扩展到分块、多位点和重组情形。
5. 将恢复图像送入 ImageNet 分类器，测量 top-1/top-5 label retention。

注意：随机 DNA “纠错网络”不是本路线的核心。模型负责存在性和位置/格式推断；精确抗干扰应主要由可解释的编码层、校验、同步和 ECC 完成。

