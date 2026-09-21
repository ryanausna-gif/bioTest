# HPRC2 论文数据纯手工下载与离线实验

## 1. 版本说明

论文 `10.64898/2026.07.21.739710` 的数据集名称是 HPRC Release 2
（HPRC2），论文报告 460 个 haplotype。它和持续更新的 HPRC Release 3
生产仓库不是同一数据冻结版本。

本实验固定从 HPRC2 assembly index v1.0 中按 `seed=2026` 选择 8/2/2 个
donor，每个 donor 只使用 haplotype 1。训练、验证、测试 donor 不重叠。

## 2. Windows 手工下载目录

建立以下目录：

```text
D:\stego_offline\hprc2_paper\fasta
D:\stego_offline\oxford_pet
D:\stego_offline\hyenadna-medium-160k-hf
```

HPRC2 的 12 个浏览器直链已保存于：

```text
artifacts\hprc2_paper_manual_8_2_2\hprc_download_urls.txt
```

逐行复制到浏览器地址栏，下载后保持原始文件名，全部保存到
`D:\stego_offline\hprc2_paper\fasta`。12 个压缩 FASTA 合计约 9.81 GiB。

固定划分为：

```text
train: HG00423 HG01109 NA18970 NA20752 HG00290 NA19776 HG02391 HG03784
val:   HG00609 HG02074
test:  HG00329 NA19338
```

## 3. Windows 离线校验

在 PowerShell 中运行：

```powershell
$FASTA = "D:\stego_offline\hprc2_paper\fasta"
$EXPECTED = "D:\学习\生信\灵序\codex\artifacts\hprc2_paper_manual_8_2_2\official_md5s.txt"

$actual = Get-ChildItem -LiteralPath $FASTA -Filter "*.fa.gz" |
  ForEach-Object {
    "{0}  {1}" -f ((Get-FileHash -LiteralPath $_.FullName -Algorithm MD5).Hash.ToLowerInvariant()), $_.Name
  } | Sort-Object
$expected = Get-Content -LiteralPath $EXPECTED | Sort-Object
$difference = Compare-Object $expected $actual
if ($difference) {
  $difference | Format-Table
  throw "HPRC2 FASTA count, filename, or MD5 mismatch."
}
"All 12 HPRC2 FASTA files passed official MD5 verification."

$actual | Set-Content -LiteralPath "$FASTA\official_md5s.txt" -Encoding Ascii
$sha256 = Get-ChildItem -LiteralPath $FASTA -Filter "*.fa.gz" |
  ForEach-Object {
    "{0}  {1}" -f ((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()), $_.Name
  }
$sha256 | Set-Content -LiteralPath "$FASTA\windows_sha256s.txt" -Encoding Ascii
```

## 4. 其他 Windows 手工下载文件

Oxford-IIIT Pet：

```text
https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
```

可选 HyenaDNA checkpoint：打开下列模型仓库的 `Files and versions`，把
`config.json`、`configuration_hyena.py`、`model.safetensors`、
`modeling_hyena.py`、`special_tokens_map.json`、`tokenization_hyena.py`、
`tokenizer_config.json` 和 `README.md` 保存到同一个目录：

```text
https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf/tree/main
```

## 5. 上传映射

使用 WinSCP/SFTP 二进制模式上传：

```text
D:\stego_offline\hprc2_paper\fasta\*
  -> /public/home/lifex/jiangjr/biolearning/database/data/raw/hprc2_paper/fasta/

D:\stego_offline\oxford_pet\*
  -> /public/home/lifex/jiangjr/biolearning/database/data/raw/oxford_pet/

D:\stego_offline\hyenadna-medium-160k-hf\*
  -> /public/home/lifex/jiangjr/models/hyenadna-medium-160k-hf/

artifacts\hprc2_paper_manual_8_2_2\*
  -> /public/home/lifex/jiangjr/biolearning/database/data/raw/hprc2_paper/prepared_seed2026/
```

## 6. Linux 离线导入

```bash
export PROJECT=/public/home/lifex/jiangjr/dpproject/projects/coding_recognition/cod
export DATA_ROOT=/public/home/lifex/jiangjr/biolearning/database/data/raw
export HPRC_ROOT="$DATA_ROOT/hprc2_paper"
export HPRC_FASTA="$HPRC_ROOT/fasta"
export HPRC_PREP="$HPRC_ROOT/prepared_seed2026"
export PET_ROOT="$DATA_ROOT/oxford_pet"
export MANIFEST_ROOT="$PROJECT/data/manifests/hprc2_stego_v1"
export KEY_FILE="$PROJECT/data/keys/image_stego_v1.key"
export ARTIFACT_ROOT="$PROJECT/artifacts/minimal_hprc2_stego_v1"

cd "$HPRC_FASTA"
sha256sum -c windows_sha256s.txt

chmod +x "$HPRC_PREP/verify_and_unpack_hprc_offline.sh"
bash "$HPRC_PREP/verify_and_unpack_hprc_offline.sh"

cd "$PET_ROOT"
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

解压后 HPRC 需要约 36 GiB，建议为 HPRC 压缩包、解压 FASTA 和实验输出
合计预留至少 70 GiB。

## 7. 图片 manifest 与密钥

```bash
cd "$PROJECT"
mkdir -p "$MANIFEST_ROOT" "$PROJECT/data/keys" "$ARTIFACT_ROOT"

python scripts/prepare_oxford_pet_manifest.py \
  --images-dir "$PET_ROOT/images" \
  --annotations "$PET_ROOT/annotations/list.txt" \
  --out "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  --n-train 3000 --n-val 750 --n-test 1250 \
  --seed 2026 --verify-images

umask 077
test -s "$KEY_FILE" || openssl rand -out "$KEY_FILE" 32
chmod 600 "$KEY_FILE"
```

## 8. 数据闸门与模型实验

先验证数据、四种 codec 和 oracle 恢复：

```bash
N_TRAIN=40 N_VAL=20 N_TEST=20 \
EPOCHS=1 RUN_CNN=0 RUN_HYENA=0 \
PHYSICAL_GPU=3 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/data_gate" \
  "$KEY_FILE"
```

通过后运行 CNN：

```bash
N_TRAIN=400 N_VAL=100 N_TEST=200 \
EPOCHS=3 RUN_CNN=1 RUN_HYENA=0 \
PHYSICAL_GPU=3 WORKERS=4 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/cnn_hyena_minimal" \
  "$KEY_FILE"
```

复用同一数据集运行 HyenaDNA：

```bash
export HYENA_MODEL=/public/home/lifex/jiangjr/models/hyenadna-medium-160k-hf

N_TRAIN=400 N_VAL=100 N_TEST=200 \
EPOCHS=3 RUN_CNN=0 RUN_HYENA=1 REUSE_DATASET=1 \
HYENA_MODEL="$HYENA_MODEL" PHYSICAL_GPU=3 WORKERS=4 SEED=2026 \
bash scripts/run_minimal_stego_experiment.sh \
  "$HPRC_PREP/genome_manifest.csv" \
  "$MANIFEST_ROOT/oxford_pet_image_manifest.csv" \
  "$ARTIFACT_ROOT/cnn_hyena_minimal" \
  "$KEY_FILE"
```

数据闸门必须满足 `dataset_audit.json` 与 `roundtrip_report.json` 的
`passed=true`，再解释 CNN/HyenaDNA 的 presence、IoU、边界误差和恢复指标。
