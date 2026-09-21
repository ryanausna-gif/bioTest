# T2T/HPRC 图像定位实验：Linux A800 完整执行手册

## 0. 实验边界

当前 MVP 固定为灰度、单通道、8 bit、128 x 128、原始像素逐行存储。每个像素字节编码成 4 个 DNA 碱基，因此图像载荷固定为 65,536 bp；观察窗口固定为 131,072 bp。正样本连续插入一次，负样本不插入。本阶段不包含替换、indel、压缩、identifier 或 ECC。

不要把同一个 CHM13 FASTA 切成 train/val/test 后报告泛化性能。正式实验按供体划分。HPRC Release 2 提供高质量、分相、near-T2T 装配，部分染色体已经达到完整 T2T，但不应把每个 R2 文件都表述为严格 gapless T2T。

官方入口：

- [HPRC Data Explorer](https://data.humanpangenome.org/assemblies)
- [HPRC Release 2 下载说明](https://github.com/human-pangenomics/hprc_intermediate_assembly/blob/main/data_tables/README.md)
- [HPRC Release 2 assembly index](https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/refs/heads/main/data_tables/assemblies_release2_v1.0.index.csv)
- [ImageNet 官方下载页](https://image-net.org/download.php)
- [ImageNet-1k Kaggle 官方入口](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
- [Oxford-IIIT Pet 官方页](https://www.robots.ox.ac.uk/~vgg/data/pets/)，适合下载链路和小规模实验

## 1. 推荐的三级实验

| 级别 | 基因组 | 图片 | 样本数 | 目的 |
|---|---|---|---:|---|
| 管线 smoke | 现有 CHM13，只建 train | Oxford Pets | 20 | 检查编码、磁盘和审计，不训练 |
| 模型 pilot | 12 个 HPRC 供体，8/2/2 | Oxford Pets 或 ImageNet | 2,000/400/600 | 跑通训练并调参 |
| 正式实验 | 100 个 HPRC 供体，70/15/15 | ImageNet-1k | 6,000/1,500/2,500 | 报告最终独立供体结果 |

正式方案如果保留 ImageNet 压缩包、解压文件、100 个压缩装配及解压 FASTA，建议准备至少 1 TB 磁盘。只下载每位供体的 haplotype 1，可把基因组数据量减半；第一阶段足够使用。

## 2. 上传代码并进入项目

本地至少上传：

```text
genome_mining/
scripts/prepare_hprc_image_experiment.py
environment-genome-v12-a800.yml
requirements-genome-v12.txt
```

服务器执行：

```bash
export PROJECT="$HOME/dpproject/projects/coding_recognition/cod"
cd "$PROJECT"
test -f environment-genome-v12-a800.yml
test -f scripts/prepare_hprc_image_experiment.py
python -m genome_mining --help
```

## 3. 创建 Conda 环境

```bash
source "$HOME/software/miniconda3/etc/profile.d/conda.sh"
cd "$PROJECT"

conda env create -f environment-genome-v12-a800.yml
conda activate genome-mining-v12
hash -r

which python
python -m pip -V
python -m pip install -r requirements-genome-v12.txt
conda install -c conda-forge pigz unzip -y
```

`which python` 和 `python -m pip -V` 必须都指向：

```text
$HOME/software/miniconda3/envs/genome-mining-v12/
```

验证 A800 GPU3：

```bash
export CUDA_VISIBLE_DEVICES=3
nvidia-smi -i 3

python - <<'PY'
import numpy as np
import torch
from PIL import Image
print('numpy:', np.__version__)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('visible gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('logical cuda:0:', torch.cuda.get_device_name(0))
PY
```

设置 `CUDA_VISIBLE_DEVICES=3` 后，程序里的 `cuda`/`cuda:0` 就是物理 GPU3，不要写 `cuda:3`。

## 4A. 快速图片集：Oxford-IIIT Pet

它约有 7,400 张图片，足够做 6,000 个左右总样本、50% 正样本的 pilot。正式论文实验仍推荐 ImageNet。

```bash
export DATA_ROOT="$HOME/biolearning/database/data/raw"
export PET_ROOT="$DATA_ROOT/oxford_pet"
mkdir -p "$PET_ROOT"
cd "$PET_ROOT"

wget -c https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
tar -xzf images.tar.gz

export IMAGE_ROOT="$PET_ROOT/images"
find "$IMAGE_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```

检查图片可读性：

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
root = Path.home() / 'biolearning/database/data/raw/oxford_pet/images'
bad = []
for path in root.iterdir():
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        continue
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        bad.append((str(path), str(exc)))
print('bad_images:', len(bad))
print(*bad[:10], sep='\n')
PY
```

## 4B. 正式图片集：ImageNet-1k

ImageNet 官方说明 ILSVRC 子集包含 1,281,167 张训练图片和 50,000 张验证图片。先在 ImageNet/Kaggle 页面登录、接受条款并加入 competition，然后在 Kaggle 设置页创建 API token。不要使用绕过条款的第三方直链。

```bash
conda activate genome-mining-v12
python -m pip install kaggle
mkdir -p "$HOME/.kaggle"
```

把 Kaggle 下载的 `kaggle.json` 放到 `$HOME/.kaggle/kaggle.json` 后：

```bash
chmod 600 "$HOME/.kaggle/kaggle.json"
kaggle competitions files -c imagenet-object-localization-challenge | head

export IMAGENET_ROOT="$DATA_ROOT/imagenet_ilsvrc2012"
mkdir -p "$IMAGENET_ROOT"
cd "$IMAGENET_ROOT"

kaggle competitions download -c imagenet-object-localization-challenge
unzip -n imagenet-object-localization-challenge.zip
tar -xzf imagenet_object_localization_patched2019.tar.gz

export IMAGE_ROOT="$IMAGENET_ROOT/ILSVRC/Data/CLS-LOC/train"
test -d "$IMAGE_ROOT"
find "$IMAGE_ROOT" -type f -iname '*.JPEG' | wc -l
```

最后一个数量应接近 1,281,167。当前任务只需要训练图片目录；程序会递归读取 synset 子目录，并保存父目录名作为 `class_id`。

## 5. 用现有 CHM13 做数据管线 smoke

```bash
cd "$PROJECT"
mkdir -p data/manifests artifacts/image_locator

export CHM13="$HOME/biolearning/database/data/raw/ncbi/human_t2t_chm13v2/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
test -s "$CHM13"

cat > data/manifests/chm13_smoke.csv <<CSV
donor_id,assembly_id,fasta,split
CHM13,T2T-CHM13v2.0,$CHM13,train
CSV

python -u -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/chm13_smoke.csv \
  --image-root "$IMAGE_ROOT" \
  --out artifacts/image_locator/chm13_smoke \
  --n-train 20 --n-val 0 --n-test 0 \
  --positive-fraction 0.5 \
  --image-size 128 \
  --window-length 131072 \
  --seed 42 2>&1 | tee artifacts/image_locator/chm13_smoke.build.log

python -m genome_mining audit-image-dataset \
  --dataset artifacts/image_locator/chm13_smoke \
  --out artifacts/image_locator/chm13_smoke/audit.json

python -m json.tool artifacts/image_locator/chm13_smoke/audit.json
```

必须看到 `passed: true`。这个数据没有 val/test，不能训练，也不能报告准确率。

## 6. 下载 12 个 HPRC 供体

如果服务器不能访问外网，不要运行本节的 `wget` 和 Linux 下载脚本。改用
[`hprc_r2_offline_windows_cn.md`](hprc_r2_offline_windows_cn.md) 中的 Windows 下载、服务器离线导入流程。

```bash
cd "$PROJECT"
export HPRC_ROOT="$DATA_ROOT/hprc_release2_image_pilot"
mkdir -p "$HPRC_ROOT/index" "$HPRC_ROOT/fasta" data/manifests/hprc_pilot

wget -c \
  https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/refs/heads/main/data_tables/assemblies_release2_v1.0.index.csv \
  -O "$HPRC_ROOT/index/assemblies_release2_v1.0.index.csv"

python scripts/prepare_hprc_image_experiment.py \
  --index "$HPRC_ROOT/index/assemblies_release2_v1.0.index.csv" \
  --out data/manifests/hprc_pilot \
  --fasta-dir "$HPRC_ROOT/fasta" \
  --n-train-donors 8 \
  --n-val-donors 2 \
  --n-test-donors 2 \
  --haplotypes hap1 \
  --seed 42

column -t -s $'\t' data/manifests/hprc_pilot/hprc_download_plan.tsv | head -20
chmod +x data/manifests/hprc_pilot/download_and_unpack_hprc.sh
bash data/manifests/hprc_pilot/download_and_unpack_hprc.sh \
  2>&1 | tee data/manifests/hprc_pilot/download.log
```

下载脚本把 HPRC index 中的 `s3://` 地址转换为匿名 HTTPS，使用 `curl -C -` 断点续传，逐个执行 gzip 完整性检查，并保留 `.fa.gz` 后解压出 `.fa`，不需要 AWS CLI。检查 manifest 指向的每个文件：

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path('data/manifests/hprc_pilot/genome_manifest.csv')
rows = list(csv.DictReader(p.open()))
missing = [r['fasta'] for r in rows if not Path(r['fasta']).is_file()]
print('assemblies:', len(rows))
print('donors:', len({r['donor_id'] for r in rows}))
print('split_donors:', {s: len({r['donor_id'] for r in rows if r['split']==s}) for s in ('train','val','test')})
print('missing:', missing)
PY
```

预期是 12 个 assembly、12 个 donor、`8/2/2`、`missing=[]`。

## 7. 构建 12 供体 pilot 数据集

先用较小规模：

```bash
cd "$PROJECT"
python -u -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/hprc_pilot/genome_manifest.csv \
  --image-root "$IMAGE_ROOT" \
  --out artifacts/image_locator/hprc12_pilot \
  --n-train 2000 \
  --n-val 400 \
  --n-test 600 \
  --positive-fraction 0.5 \
  --image-size 128 \
  --window-length 131072 \
  --min-acgt-fraction 0.95 \
  --seed 42 2>&1 | tee artifacts/image_locator/hprc12_pilot.build.log

python -m genome_mining audit-image-dataset \
  --dataset artifacts/image_locator/hprc12_pilot \
  --out artifacts/image_locator/hprc12_pilot/audit.json

python -m json.tool artifacts/image_locator/hprc12_pilot/audit.json
```

不要在 `passed=false` 时继续训练。数据目录应包含：

```text
sequences.npy
metadata.csv
dataset_manifest.json
audit.json
```

## 8. 先做一轮 GPU smoke 训练

```bash
export CUDA_VISIBLE_DEVICES=3
cd "$PROJECT"

python -u -m genome_mining train-image-locator \
  --dataset artifacts/image_locator/hprc12_pilot \
  --out artifacts/image_locator/hprc12_smoke_model \
  --epochs 1 \
  --batch-size 4 \
  --d-model 64 \
  --context dilated_cnn \
  --context-layers 2 \
  --workers 2 \
  --device cuda \
  --amp \
  --seed 42 2>&1 | tee artifacts/image_locator/hprc12_smoke_model.train.log
```

确认生成 `model.pt`、`train_log.json` 和 `test_evaluation/metrics.json` 后，再跑正式 pilot：

```bash
python -u -m genome_mining train-image-locator \
  --dataset artifacts/image_locator/hprc12_pilot \
  --out artifacts/image_locator/hprc12_dilated \
  --epochs 20 \
  --batch-size 16 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --d-model 128 \
  --downsample-stride 256 \
  --context dilated_cnn \
  --context-layers 6 \
  --workers 4 \
  --device cuda \
  --amp \
  --seed 42 2>&1 | tee artifacts/image_locator/hprc12_dilated.train.log
```

另开一个终端监控：

```bash
watch -n 2 nvidia-smi -i 3
tail -f "$PROJECT/artifacts/image_locator/hprc12_dilated.train.log"
```

如果 OOM，把 batch size 依次降为 8、4、2。不要修改 `CUDA_VISIBLE_DEVICES=3` 后再写 `--device cuda:3`。

## 9. 独立评估与结果检查

```bash
python -u -m genome_mining evaluate-image-locator \
  --dataset artifacts/image_locator/hprc12_pilot \
  --model artifacts/image_locator/hprc12_dilated/model.pt \
  --out artifacts/image_locator/hprc12_dilated/test_recheck \
  --split test \
  --batch-size 16 \
  --workers 4 \
  --device cuda \
  --save-images \
  --max-saved-images 100

python -m json.tool artifacts/image_locator/hprc12_dilated/test_recheck/metrics.json
```

优先查看：

```text
presence_detection.auroc
presence_detection.aupr
presence_detection.f1
localization.start_mae_bases
localization.start_exact_rate
localization.base_phase_accuracy
localization.joint_detect_and_exact_start_rate
image_recovery.exact_image_rate
image_recovery.mean_pixel_mae
image_recovery.mean_psnr
image_recovery.mean_ssim_global
```

二分类好但 `start_exact_rate` 很低，表示模型只学会了“有图”，还不能解码；不能把这种结果算作第一阶段完成。

## 10. 扩展到 100 个供体

重新生成独立目录，避免覆盖 pilot：

```bash
export HPRC100_ROOT="$DATA_ROOT/hprc_release2_image_100"
mkdir -p "$HPRC100_ROOT/fasta" data/manifests/hprc100

python scripts/prepare_hprc_image_experiment.py \
  --index "$HPRC_ROOT/index/assemblies_release2_v1.0.index.csv" \
  --out data/manifests/hprc100 \
  --fasta-dir "$HPRC100_ROOT/fasta" \
  --n-train-donors 70 \
  --n-val-donors 15 \
  --n-test-donors 15 \
  --haplotypes hap1 \
  --seed 2026

chmod +x data/manifests/hprc100/download_and_unpack_hprc.sh
bash data/manifests/hprc100/download_and_unpack_hprc.sh \
  2>&1 | tee data/manifests/hprc100/download.log
```

然后重复第 7–9 步，把 manifest 换为 `data/manifests/hprc100/genome_manifest.csv`，图片换为 ImageNet，样本数换为 `6000/1500/2500`。必须重新构建数据集，不能把 12 供体 pilot 的窗口混入正式测试集。

## 11. 常见错误

### pip 仍指向 `/opt/conda`

```bash
source "$HOME/software/miniconda3/etc/profile.d/conda.sh"
conda activate genome-mining-v12
hash -r
which python
python -m pip -V
```

安装时始终使用 `python -m pip`。

### `No module named genome_mining`

```bash
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
python -m genome_mining --help
```

### ImageNet 下载返回 403

先在 Kaggle 网页加入 competition、接受规则，再重新生成 API token；检查 `chmod 600 ~/.kaggle/kaggle.json`。

### 数据审计失败

不要跳过。重点查看 `donor_leakage`、`reused_image_hashes`、`cross_split_sequence_duplicates` 和 `overlapping_source_intervals`。修正 manifest 或图片源后重新构建到一个新输出目录。

### 构建阶段长时间没有 GPU 利用率

正常。`build-image-dataset` 是 CPU/磁盘任务，会扫描各装配；只有 `train-image-locator` 和 `evaluate-image-locator` 使用 GPU。

## 12. 结果归档

至少保留：

```text
HPRC assembly index 原文件
hprc_download_plan.tsv
genome_manifest.csv
dataset_manifest.json
audit.json
训练命令和日志
model.pt
train_log.json
test metrics.json
predictions.csv
image_recovery.csv
恢复图片样例
```

周报中同时记录代码版本、随机种子、供体 ID、图片来源、样本数、模型配置和 GPU 型号。只有 donor-disjoint、image-disjoint 的 test 结果才作为正式结果。
