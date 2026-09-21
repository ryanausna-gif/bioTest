# HPRC Release 2 Windows 下载与 Linux 离线导入

这套流程用于 Linux 服务器不能访问外网、但 Windows 可以联网的情况。供体选择必须先由
`prepare_hprc_image_experiment.py` 固定完成，再下载对应 assembly，不能手工任选供体。

## 1. 生成离线计划

在能访问 HPRC 索引的机器上运行：

```bash
python scripts/prepare_hprc_image_experiment.py \
  --index assemblies_release2_v1.0.index.csv \
  --out hprc_r2_offline_8_2_2 \
  --fasta-dir /public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/fasta \
  --n-train-donors 8 \
  --n-val-donors 2 \
  --n-test-donors 2 \
  --haplotypes hap1 \
  --seed 2026
```

输出文件：

```text
donor_split.csv                    donor 与 train/val/test 划分
hprc_download_plan.tsv             assembly、URI、文件名和 MD5 URI
hprc_download_urls.txt             纯下载地址
download_hprc_windows.ps1          Windows 批量下载器
download_and_unpack_hprc.sh        Linux 能联网时的下载器
verify_and_unpack_hprc_offline.sh  Linux 离线验收与解压脚本
genome_manifest.csv                后续数据集构建入口
```

同一个索引、参数与 seed 会得到相同划分。跨机器复现时还应保存索引文件及其 SHA-256。

## 2. Windows 下载

打开 PowerShell 并进入离线计划目录。脚本使用 Windows 自带的 `curl.exe`，不需要安装 AWS CLI：

```powershell
curl.exe --version
Set-ExecutionPolicy -Scope Process Bypass
.\download_hprc_windows.ps1
```

脚本默认把文件保存到同级 `fasta` 目录。它会下载 index 中记录的官方 MD5，只有校验通过
才跳过已有文件；半截文件会通过公开 HTTPS 和 `curl -C -` 续传。完成后还会生成
`windows_sha256s.txt`，用于在 Linux 上验证上传是否损坏。

检查：

```powershell
(Get-ChildItem .\fasta -Filter "*.fa.gz").Count
Get-ChildItem .\fasta -Filter "*.fa.gz" | Select-Object Name,Length
Get-Content .\fasta\windows_sha256s.txt
```

hap1、8/2/2 配置应得到 12 个压缩 FASTA。不要在 Windows 解压。

## 3. 上传服务器

把 `fasta/*.fa.gz` 上传到：

```text
/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/fasta/
```

同时上传 `fasta/windows_sha256s.txt`，并在 Linux 解压前执行：

```bash
cd /public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/fasta
sha256sum -c windows_sha256s.txt
```

把其余计划文件上传到：

```text
/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/prepared_8_2_2/
```

## 4. Linux 离线验收和解压

```bash
export HPRC=/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2
export HPRC_FASTA=$HPRC/fasta
export HPRC_PREP=$HPRC/prepared_8_2_2

chmod +x "$HPRC_PREP/verify_and_unpack_hprc_offline.sh"
bash "$HPRC_PREP/verify_and_unpack_hprc_offline.sh" 2>&1 \
  | tee "$HPRC_PREP/offline_import.log"
```

脚本先要求 12 个指定文件全部存在，再逐个运行 `pigz -t` 或 `gzip -t`，最后解压为 `.fa`。
任何压缩包缺失、为空、损坏或解压失败都会返回非零状态。

## 5. 最终检查

```bash
cat "$HPRC_PREP/donor_split.csv"
sed -n '1,15p' "$HPRC_PREP/genome_manifest.csv"

python - <<'PY'
import csv
from pathlib import Path

manifest = Path("/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc_r2/prepared_8_2_2/genome_manifest.csv")
rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
missing = [row["fasta"] for row in rows if not Path(row["fasta"]).is_file()]
counts = {
    split: len({row["donor_id"] for row in rows if row["split"] == split})
    for split in ("train", "val", "test")
}
print("assemblies:", len(rows))
print("donor split:", counts)
print("missing:", missing)
assert len(rows) == 12
assert counts == {"train": 8, "val": 2, "test": 2}
assert not missing
PY
```

只有最终检查通过后，才把 `genome_manifest.csv` 交给 `build-image-dataset`。
