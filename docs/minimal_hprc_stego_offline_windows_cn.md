# Windows 下载、Linux 服务器完全离线的最小隐写实验

## 0. 离线边界

本流程假定：

```text
Windows 可以访问互联网
Linux A800 服务器不能访问互联网
Windows 可以通过 WinSCP、SFTP 或 scp 向服务器上传文件
服务器已有可用的 genome-hyenadna Conda 环境
```

服务器端不会执行 `curl`、`wget`、`hf download`、`pip install`、`conda install` 或 AWS CLI。
最小目标只训练 CNN，不需要 HyenaDNA checkpoint。

Windows Conda 环境不能复制到 Linux。若服务器还没有依赖环境，必须在另一台同架构 Linux
机器上制作 `conda-pack`，或者使用服务器可访问的内部 Conda 镜像；不能把 Windows 环境直接上传。

## 1. Windows 离线下载目录

在 PowerShell 中建立目录。下面以 `D:\stego_offline` 为例：

```powershell
$OFFLINE = "D:\stego_offline"
$PET = Join-Path $OFFLINE "oxford_pet"
$HPRC = Join-Path $OFFLINE "hprc_r2"
$HPRC_PLAN = Join-Path $HPRC "prepared_seed2026"
$HPRC_FASTA = Join-Path $HPRC "fasta"

New-Item -ItemType Directory -Force -Path $PET | Out-Null
New-Item -ItemType Directory -Force -Path $HPRC_PLAN | Out-Null
New-Item -ItemType Directory -Force -Path $HPRC_FASTA | Out-Null
```

## 2. Windows 手工下载 Oxford-IIIT Pet

在浏览器中打开并保存到 `$PET`：

```text
https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
```

最终路径应为：

```text
D:\stego_offline\oxford_pet\images.tar.gz
D:\stego_offline\oxford_pet\annotations.tar.gz
```

检查文件并生成跨机器 SHA256 清单：

```powershell
Get-Item "$PET\images.tar.gz", "$PET\annotations.tar.gz" |
    Select-Object Name,Length

$petFiles = @("images.tar.gz", "annotations.tar.gz")
$petHashes = foreach ($name in $petFiles) {
    $hash = (Get-FileHash -LiteralPath (Join-Path $PET $name) -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $name"
}
$petHashes | Set-Content -LiteralPath "$PET\windows_sha256s.txt" -Encoding Ascii
Get-Content "$PET\windows_sha256s.txt"
```

不要在 Windows 生成最终 `image_manifest.csv`。它必须在 Linux 解压后生成，才能保存 Linux 路径。

## 3. Windows 下载 HPRC index

在浏览器打开：

```text
https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/refs/heads/main/data_tables/assemblies_release2_v1.0.index.csv
```

使用“另存为”保存到：

```text
D:\stego_offline\hprc_r2\assemblies_release2_v1.0.index.csv
```

检查第一行和计算 SHA256：

```powershell
Get-Content "$HPRC\assemblies_release2_v1.0.index.csv" -TotalCount 2
Get-FileHash "$HPRC\assemblies_release2_v1.0.index.csv" -Algorithm SHA256 |
    Format-List
```

第一行必须包含：

```text
sample_id,haplotype,...,assembly_name,...,assembly_md5,...,assembly
```

## 4. Windows 生成固定 HPRC 供体计划

进入 Windows 上的项目目录。该目录必须包含更新后的
`scripts\prepare_hprc_image_experiment.py`：

```powershell
Set-Location "D:\学习\生信\灵序\codex"

python scripts\prepare_hprc_image_experiment.py `
  --index "$HPRC\assemblies_release2_v1.0.index.csv" `
  --out "$HPRC_PLAN" `
  --fasta-dir "/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/fasta" `
  --n-train-donors 8 `
  --n-val-donors 2 `
  --n-test-donors 2 `
  --haplotypes hap1 `
  --seed 2026

Get-Content "$HPRC_PLAN\donor_split.csv"
Get-Content "$HPRC_PLAN\hprc_download_urls.txt" -TotalCount 3
```

预期供体为：

```text
train: HG00423 HG01109 NA18970 NA20752 HG00290 NA19776 HG02391 HG03784
val:   HG00609 HG02074
test:  HG00329 NA19338
```

如果结果不同，先确认使用的是同一份 index 文件及其 SHA256，不要手工修改 split。

## 5. Windows 下载 12 个 HPRC FASTA

### 推荐方式：校验式 PowerShell 下载器

脚本只使用 Windows 自带的 `curl.exe`，不需要 AWS CLI。它会：

```text
下载官方 assembly MD5
验证已有文件
对半截文件执行 curl -C - 续传
下载后再次验证 MD5
生成供 Linux 验证的 windows_sha256s.txt
```

执行：

```powershell
curl.exe --version
Set-ExecutionPolicy -Scope Process Bypass

& "$HPRC_PLAN\download_hprc_windows.ps1" -OutputDir "$HPRC_FASTA"
```

中断后重复同一条命令。MD5 已通过的文件会跳过，未完成文件会继续下载。

检查：

```powershell
(Get-ChildItem $HPRC_FASTA -Filter "*.fa.gz").Count
Get-ChildItem $HPRC_FASTA -Filter "*.fa.gz" | Select-Object Name,Length
Get-Content "$HPRC_FASTA\windows_sha256s.txt"
```

应有 12 个 `.fa.gz`。

### 完全手工方式

`$HPRC_PLAN\hprc_download_urls.txt` 中有 12 个匿名 HTTPS 地址。可逐个复制到浏览器下载，
文件名必须保持不变，并全部放入 `$HPRC_FASTA`。下载完仍建议运行一次
`download_hprc_windows.ps1`，让它完成官方 MD5 验证和 SHA256 清单生成；已完整文件不会重下。

## 6. 上传到 Linux 服务器

使用 WinSCP、SFTP 或 scp，按以下映射上传，不要把 12 个大 FASTA 打成单一 zip。

| Windows 文件 | Linux 目标 |
|---|---|
| `oxford_pet/images.tar.gz` | `/public/home/lifex/jiangjr/biolearning/database/data/raw/oxford_pet/` |
| `oxford_pet/annotations.tar.gz` | 同上 |
| `oxford_pet/windows_sha256s.txt` | 同上 |
| `hprc_r2/fasta/*.fa.gz` | `/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/fasta/` |
| `hprc_r2/fasta/windows_sha256s.txt` | 同上 |
| `hprc_r2/prepared_seed2026/*` | `/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/prepared_seed2026/` |
| 更新后的项目代码 | `/public/home/lifex/jiangjr/dpproject/projects/coding_recognition/cod/` |

项目代码至少包括：

```text
genome_mining/
scripts/prepare_hprc_image_experiment.py
scripts/prepare_oxford_pet_manifest.py
scripts/run_minimal_stego_experiment.sh
environment-hyenadna-a800.yml
requirements-hyenadna-stego.txt
```

WinSCP 建议启用二进制传输，不要以文本模式上传 `.gz`、`.npy`、模型或密钥。

## 7. Linux 设置目录，全程不联网

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

## 8. Linux 验证 Windows 上传文件

Oxford Pets：

```bash
cd "$PET_ROOT"
sha256sum -c windows_sha256s.txt
```

两行都必须显示 `OK`。

HPRC：

```bash
cd "$HPRC_FASTA"
sha256sum -c windows_sha256s.txt
```

12 行都必须显示 `OK`。任何 `FAILED` 都应重新上传对应文件，不能继续解压。

## 9. Linux 离线解压 Oxford Pets

```bash
cd "$PET_ROOT"
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz

test -s "$PET_ROOT/annotations/list.txt"
find "$PET_ROOT/images" -type f -iname '*.jpg' | wc -l
```

图片数量应大于 7,000。

## 10. Linux 离线生成图片 manifest

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

预期为 `selected_images=5000`，CSV 共 5,001 行。

## 11. Linux 离线验证并解压 HPRC

```bash
chmod +x "$HPRC_PREP/verify_and_unpack_hprc_offline.sh"
bash "$HPRC_PREP/verify_and_unpack_hprc_offline.sh" \
  2>&1 | tee "$HPRC_PREP/offline_import.log"
```

然后验证 manifest：

```bash
cat "$HPRC_PREP/donor_split.csv"
sed -n '1,15p' "$HPRC_PREP/genome_manifest.csv"

tail -n +2 "$HPRC_PREP/genome_manifest.csv" | cut -d, -f3 | while read -r fasta; do
  test -s "$fasta" || { echo "missing: $fasta"; exit 1; }
done
```

必须得到 12 个未压缩 `.fa`，且 manifest 中不存在缺失路径。

## 12. Linux 检查现有 Conda 环境

服务器离线时不要重新执行 `conda env create` 或 `pip install`。使用已有环境：

```bash
source /public/home/lifex/jiangjr/software/miniconda3/etc/profile.d/conda.sh
conda activate genome-hyenadna
hash -r

which python
python -m pip -V
export CUDA_VISIBLE_DEVICES=3

python - <<'PY'
import cryptography
import numpy
import PIL
import reedsolo
import sklearn
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("logical cuda:0:", torch.cuda.get_device_name(0))
PY
```

这里任何 import 失败都表示离线环境仍不完整。Windows 下载的 wheel 通常不能用于 Linux，必须准备
Linux `manylinux` wheel 或 Linux `conda-pack`。

## 13. Linux 离线生成密钥

```bash
umask 077
if [[ ! -s "$KEY_FILE" ]]; then
  openssl rand -out "$KEY_FILE" 32
fi
chmod 600 "$KEY_FILE"
sha256sum "$KEY_FILE"
```

密钥由服务器本地生成，不需要上传，也不能在数据集构建后覆盖。

## 14. Linux 代码测试

```bash
cd "$PROJECT"

python -m compileall genome_mining scripts
python -m unittest \
  tests.test_prepare_hprc_image_experiment \
  tests.test_prepare_oxford_pet_manifest \
  tests.test_image_genome_mvp \
  tests.test_stego_hyenadna -v

python -m genome_mining list-stego-codecs
python -m genome_mining verify-stego-roundtrip --help
```

## 15. Linux 数据层 80 样本闸门

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

只有以下条件全部满足才训练模型：

```text
dataset audit passed = true
roundtrip passed = true
L0/L1/L2/L3 decode_success_rate = 1.0
L0/L1/L2/L3 exact_image_rate = 1.0
```

## 16. Linux 最小 CNN 隐写检测

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

查看结果：

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

## 17. 可选离线 HyenaDNA

最小 CNN 闭环不需要 HyenaDNA。后续需要时，在 Windows 打开：

```text
https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf/tree/main
```

推荐在 Windows 使用 `hf download` 下载完整 snapshot，再把整个目录上传到：

```text
/public/home/lifex/jiangjr/models/hyenadna-medium-160k-hf/
```

训练时必须传本地目录和 `--hyena-local-files-only`，防止程序尝试联网。不要只下载
`config.json` 和权重；该 checkpoint 使用自定义模型代码，必须保留 snapshot 中全部配置和 Python 文件。
