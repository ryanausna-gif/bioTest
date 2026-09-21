# HPRC + Oxford Pets 最小隐写闭环实验

> Linux 服务器不能访问外网时，不要执行本文中的服务器下载命令，请改用
> [Windows 下载、Linux 完全离线流程](minimal_hprc_stego_offline_windows_cn.md)。

## 0. 本轮目标

本轮只完成下面的可复现闭环，不涉及生存仿真：

```text
HPRC donor-disjoint FASTA + Oxford-IIIT Pet 不重复图片
-> 灰度 128x128 原始像素
-> L0/L1/L2/L3 编码
-> 随机连续插入 131,072 bp 自然基因组窗口
-> 数据泄漏审计
-> oracle 边界精确解码
-> 隐写自然性统计
-> CNN 检测、边界定位和图片恢复
-> 可选 HyenaDNA 对照
```

最小实验只用于证明数据、codec、定位和恢复管线能够工作，不用于发表人群泛化结论。

## 1. 官方下载地址

Oxford-IIIT Pet 官方页面：

```text
https://www.robots.ox.ac.uk/~vgg/data/pets/
```

图片和标注直链：

```text
https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
```

HPRC Release 2 assembly index：

```text
https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/refs/heads/main/data_tables/assemblies_release2_v1.0.index.csv
```

HyenaDNA medium-160k 模型页：

```text
https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf
```

## 2. 服务器目录

```bash
export PROJECT=/public/home/lifex/jiangjr/dpproject/projects/coding_recognition/cod
export DATA_ROOT=/public/home/lifex/jiangjr/biolearning/database/data/raw
export PET_ROOT="$DATA_ROOT/oxford_pet"
export HPRC_ROOT="$DATA_ROOT/hprc_r2"
export HPRC_FASTA="$HPRC_ROOT/fasta"
export HPRC_PREP="$HPRC_ROOT/prepared_seed2026"
export MANIFEST_ROOT="$PROJECT/data/manifests/hprc_stego_v1"
export KEY_FILE="$PROJECT/data/keys/image_stego_v1.key"
export ARTIFACT_ROOT="$PROJECT/artifacts/minimal_hprc_stego_v1"

mkdir -p "$PET_ROOT" "$HPRC_FASTA" "$HPRC_PREP" "$MANIFEST_ROOT"
mkdir -p "$PROJECT/data/keys" "$ARTIFACT_ROOT"
cd "$PROJECT"
```

本地至少上传这些更新后的项目内容：

```text
genome_mining/
scripts/prepare_hprc_image_experiment.py
scripts/prepare_oxford_pet_manifest.py
scripts/run_minimal_stego_experiment.sh
environment-hyenadna-a800.yml
requirements-hyenadna-stego.txt
```

## 3. Conda 环境

```bash
source /public/home/lifex/jiangjr/software/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT"

# 仅在环境尚不存在时执行
conda env create -f environment-hyenadna-a800.yml

conda activate genome-hyenadna
hash -r
which python
python -m pip -V
```

`which python` 和 `python -m pip -V` 必须同时指向：

```text
/public/home/lifex/jiangjr/software/miniconda3/envs/genome-hyenadna/
```

不要直接执行裸 `pip`。缺少可选包时使用当前解释器：

```bash
python -m pip install -r requirements-hyenadna-stego.txt
```

验证 GPU3：

```bash
export CUDA_VISIBLE_DEVICES=3
nvidia-smi -i 3

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("logical cuda:0:", torch.cuda.get_device_name(0))
PY
```

设置 `CUDA_VISIBLE_DEVICES=3` 后，程序里的 `cuda` 是物理 GPU3，不要写 `cuda:3`。

## 4. 下载 Oxford-IIIT Pet

```bash
cd "$PET_ROOT"

curl -L --fail --retry 8 --retry-delay 5 -C - \
  -o images.tar.gz \
  https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz

curl -L --fail --retry 8 --retry-delay 5 -C - \
  -o annotations.tar.gz \
  https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

tar -xzf images.tar.gz
tar -xzf annotations.tar.gz

test -s "$PET_ROOT/annotations/list.txt"
find "$PET_ROOT/images" -type f -iname '*.jpg' | wc -l
```

官方数据约有 37 类、约 7,400 张图片。最后一条命令应大于 7,000。

## 5. 生成图片 manifest

脚本根据官方 `annotations/list.txt` 读取 breed/class，计算原文件 SHA256，以 `seed=2026` 做确定性类别分层划分：

```bash
cd "$PROJECT"

python scripts/prepare_oxford_pet_manifest.py \
  --images-dir "$PET_ROOT/images" \
  --annotations "$PET_ROOT/annotations/list.txt" \
  --out "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  --n-train 3000 \
  --n-val 750 \
  --n-test 1250 \
  --seed 2026 \
  --verify-images

python -m json.tool "$MANIFEST_ROOT/oxford_pet_image_manifest.summary.json" | head -n 60
wc -l "$MANIFEST_ROOT/oxford_pet_image_manifest.csv"
```

预期 `selected_images=5000`，CSV 为 5,001 行（含表头），三个 split 之间没有重复图片。

## 6. 准备 HPRC donor-disjoint FASTA

如果已保存此前使用的 index CSV，继续使用该文件，避免上游索引更新后改变供体集合。首次下载 index：

```bash
cd "$HPRC_ROOT"

curl -L --fail --retry 8 \
  -o assemblies_release2_v1.0.index.csv \
  https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/refs/heads/main/data_tables/assemblies_release2_v1.0.index.csv

sha256sum assemblies_release2_v1.0.index.csv > assemblies_release2_v1.0.index.csv.sha256
```

选择 8/2/2 个 donor，只取 hap1：

```bash
cd "$PROJECT"

python scripts/prepare_hprc_image_experiment.py \
  --index "$HPRC_ROOT/assemblies_release2_v1.0.index.csv" \
  --out "$HPRC_PREP" \
  --fasta-dir "$HPRC_FASTA" \
  --n-train-donors 8 \
  --n-val-donors 2 \
  --n-test-donors 2 \
  --haplotypes hap1 \
  --seed 2026

cat "$HPRC_PREP/donor_split.csv"
head -n 3 "$HPRC_PREP/hprc_download_urls.txt"
grep -n "curl -L" "$HPRC_PREP/download_and_unpack_hprc.sh" | head
```

更新后的脚本不会调用 AWS CLI。它把 `s3://` 转为匿名公开 HTTPS，使用 `curl -C -`，并在下载前检查 gzip 完整性。中断后直接重跑同一个脚本即可续传：

```bash
chmod +x "$HPRC_PREP/download_and_unpack_hprc.sh"
bash "$HPRC_PREP/download_and_unpack_hprc.sh" \
  2>&1 | tee "$HPRC_PREP/download.log"
```

如果 12 个 `.fa.gz` 已经从 Windows 上传到 `$HPRC_FASTA`，改用离线校验和解压：

```bash
chmod +x "$HPRC_PREP/verify_and_unpack_hprc_offline.sh"
bash "$HPRC_PREP/verify_and_unpack_hprc_offline.sh"
```

验证 manifest 中每个未压缩 FASTA 都存在：

```bash
tail -n +2 "$HPRC_PREP/genome_manifest.csv" | cut -d, -f3 | while read -r fasta; do
  test -s "$fasta" || { echo "missing: $fasta"; exit 1; }
done

wc -l "$HPRC_PREP/genome_manifest.csv"
```

hap1 模式应为 13 行：1 行表头和 12 个 assembly。

## 7. 生成隐写密钥

L1/L2/L3 使用 ChaCha20-Poly1305。密钥只保存在服务器，不写入 dataset manifest：

```bash
umask 077
if [[ ! -s "$KEY_FILE" ]]; then
  openssl rand -out "$KEY_FILE" 32
fi
chmod 600 "$KEY_FILE"
sha256sum "$KEY_FILE"
```

不要在构建完成后更换或覆盖该密钥，否则授权恢复会因 fingerprint 不一致而停止。

## 8. 先运行代码测试

```bash
cd "$PROJECT"

python -m compileall genome_mining scripts
python -m unittest \
  tests.test_prepare_hprc_image_experiment \
  tests.test_prepare_oxford_pet_manifest \
  tests.test_image_genome_mvp \
  tests.test_stego_hyenadna -v

python -m genome_mining list-stego-codecs
python -m genome_mining --help | grep -E 'build-image|verify-stego|train-image'
```

## 9. 数据层极小闭环，不训练模型

先用 80 个样本检查四种 codec。该步骤成功后再花 GPU 时间：

```bash
cd "$PROJECT"

N_TRAIN=40 N_VAL=20 N_TEST=20 \
EPOCHS=1 RUN_CNN=0 RUN_HYENA=0 \
PHYSICAL_GPU=3 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/data_gate" \
  "$KEY_FILE"
```

检查：

```bash
python -m json.tool "$ARTIFACT_ROOT/data_gate/dataset_audit.json"
python -m json.tool "$ARTIFACT_ROOT/data_gate/oracle_roundtrip/roundtrip_report.json"
cat "$ARTIFACT_ROOT/data_gate/stego_summary/stego_report.csv"
```

继续的必要条件：

```text
dataset_audit.json: passed = true
roundtrip_report.json: passed = true
四个 codec 的 decode_success_rate = 1.0
四个 codec 的 exact_image_rate = 1.0
```

这里使用真实边界，只验证数据制作、编码和恢复；不代表模型已经会定位。

## 10. 最小 CNN 隐写检测实验

默认最小配置是 400/100/200，共 700 个序列窗口、350 个正样本。四种 codec 轮流分配：

```bash
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=3

N_TRAIN=400 N_VAL=100 N_TEST=200 \
EPOCHS=3 RUN_CNN=1 RUN_HYENA=0 \
PHYSICAL_GPU=3 WORKERS=4 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/cnn_minimal" \
  "$KEY_FILE"
```

`build-image-dataset` 是 CPU/磁盘任务，构建期间 GPU 利用率接近零是正常的。L3 要为每个加密字节选择匹配 cover/k-mer 的码字，因此明显慢于 L0/L1。

训练结束后自动生成：

```text
cnn_minimal/cnn/model.pt
cnn_minimal/cnn/train_log.json
cnn_minimal/cnn/test_evaluation/metrics.json
cnn_minimal/cnn/test_evaluation/predictions.csv
cnn_minimal/cnn/test_evaluation/image_recovery.csv
```

查看主要结果：

```bash
python -m json.tool "$ARTIFACT_ROOT/cnn_minimal/cnn/test_evaluation/metrics.json" | less
```

重点字段：

```text
presence_detection.auroc / auprc / fpr95
localization.start_mae_bases / end_mae_bases
localization.mean_span_iou / exact_span_rate
image_recovery.decode_success_rate / exact_image_rate
by_codec
```

若训练中断但 dataset 已完整生成，保留同一个输出目录并设置：

```bash
REUSE_DATASET=1
```

脚本会跳过数据重建，重新做审计后继续训练。

## 11. 可选 HyenaDNA 最小对照

CNN 跑通后再下载 checkpoint：

```bash
export HF_HOME=/public/home/lifex/jiangjr/.cache/huggingface
export HYENA_MODEL=/public/home/lifex/jiangjr/models/hyenadna-medium-160k-hf
mkdir -p "$HYENA_MODEL"

hf download LongSafari/hyenadna-medium-160k-seqlen-hf \
  --local-dir "$HYENA_MODEL"
```

复用已经构建的 `cnn_minimal/dataset_l0_l3`，只训练冻结的 HyenaDNA-RC：

```bash
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=3

REUSE_DATASET=1 RUN_CNN=0 RUN_HYENA=1 \
HYENA_MODEL="$HYENA_MODEL" EPOCHS=3 \
PHYSICAL_GPU=3 WORKERS=4 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/cnn_minimal" \
  "$KEY_FILE"
```

输出在：

```text
cnn_minimal/hyenadna_rc/test_evaluation/metrics.json
```

`batch-size=1`、`gradient-accumulation=8` 已写入脚本。正反互补分支约为单向两倍计算量；若显存不足，先手工增加 `--hyena-no-reverse-complement` 做单向基线。

## 12. 扩大到 pilot

最小 700 样本只用于确认隐写检测能力。跑通后用新输出目录扩大到 2,000/400/600 和 5-10 epochs：

```bash
N_TRAIN=2000 N_VAL=400 N_TEST=600 \
EPOCHS=10 RUN_CNN=1 RUN_HYENA=0 \
PHYSICAL_GPU=3 WORKERS=4 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/cnn_pilot" \
  "$KEY_FILE"
```

不要覆盖 `cnn_minimal`，也不要根据 test 指标反复改变超参数。超参数选择只看 validation，test 只做冻结后的最终评估。

## 13. 当前最小实验可以回答什么

可以回答：

```text
L0-L3 数据是否能被无损编码和授权恢复
CNN/HyenaDNA 是否能在未见 donor 上判断窗口中是否存在载荷
模型是否能预测可变长度载荷的起止边界
不同 codec 的检测、定位、恢复和统计自然性有何差异
```

不能回答：

```text
自然人类基因组中是否真的存在人工编码信息
该隐写序列是否适合写入活体
载荷能否经历多代遗传
indel、测序误差和未知 codec 下是否仍可恢复
```

这些问题应在当前无噪声最小闭环稳定后单独增加实验变量。
