# genome_mining V12 迁移说明

本次迁移把旧项目 `dna_semantic_carrier_v12` 中最适合先接入主线的部分合并进了 `genome_mining`：

```text
V12 人工语义载荷/困难阴性/开放集 benchmark
  -> genome_mining.carrier_controls
  -> python -m genome_mining build-controls
```

这一步的目的不是直接证明真实基因组中存在人工编码信息，而是为后续模型训练和验证提供一套标准化“阳性控制 + 困难阴性 + 未知编码族”数据。

## 已迁移能力

### 1. V8 基础 carrier codec

```text
PlainCodec
EncryptedCodec
ECCCodec
WatermarkCodec
ProtocolCodec
StegoCodec
CodonStegoCodec
KmerMatchedCodec
DistributedCodec
AdversarialMimicCodec
```

### 2. V10 literature-inspired codec

```text
ConstrainedStorageCodec
IndexedPayloadCodec
FountainLikeCodec
SignatureWatermarkCodec
TraceabilityTagCodec
DNACryptoCodec
AdaptiveStegoCodec
ErrorRobustCodec
```

### 3. V11/V12 realistic codec

```text
RealisticStorageCodec
RealisticSparseWatermarkCodec
RealisticTraceabilityCodec
RealisticAdaptiveStegoCodec
```

### 4. 困难自然阴性

```text
natural
base_shuffle
kmer_shuffle
low_complexity
microsatellite
gc_patch
at_patch
periodic_decoy
```

这些阴性样本很重要，因为真实基因组中的 repeat、微卫星、低复杂度和 GC 异常区域很容易被模型误判为“人工编码痕迹”。

## 快速生成一个小 benchmark

```bash
python -m genome_mining build-controls \
  --out artifacts/genome_mining/v12_controls_smoke.csv \
  --generator-kind realistic \
  --length 1000 \
  --n-train 100 \
  --n-val 30 \
  --n-test 50 \
  --negative-rate 0.35 \
  --holdout-codecs CodonStegoCodec RealisticAdaptiveStegoCodec \
  --seed 42
```

输出：

```text
artifacts/genome_mining/v12_controls_smoke.csv
artifacts/genome_mining/v12_controls_smoke.manifest.json
```

## 使用真实 FASTA 作为背景

如果已经下载了真实基因组，例如 E. coli 或 T2T 子集，可以把真实窗口作为背景，再插入合成 carrier：

```bash
python -m genome_mining build-controls \
  --background-fasta data/raw/ncbi/ecoli_k12_mg1655/GCF_000005845.2_ASM584v2_genomic.fna \
  --out artifacts/genome_mining/ecoli_v12_controls.csv \
  --generator-kind realistic \
  --length 1000 \
  --n-train 1000 \
  --n-val 300 \
  --n-test 500 \
  --negative-rate 0.35 \
  --holdout-codecs CodonStegoCodec RealisticAdaptiveStegoCodec DNACryptoCodec \
  --seed 42
```

## CSV 关键字段

```text
split               train / val / test
sequence            DNA 序列
y                   0=自然/困难自然阴性, 1=人工 carrier 阳性
codec_family        编码族
codec_rule          具体规则
is_unknown          是否为 holdout 未知编码族
open_target         natural / known_special / unknown_anomaly
payload_start       载荷起点
payload_end         载荷终点
embedding_type      contiguous / sparse_watermark / protocol / adaptive_stego 等
detectability       easy / medium / hard / adversarial
metadata            JSON 元信息
```

## 和真实 genome_mining 扫描如何衔接

当前推荐流程：

```text
1. 用真实 FASTA 跑 genome_mining scan，得到 candidates.csv / candidates.bed
2. 用同一 FASTA 跑 build-controls，得到人工 carrier benchmark
3. 用 controls 训练/校准检测模型
4. 用训练好的模型重排真实 candidates
5. 对高分候选做注释过滤、解码尝试和 genome_survival_sim 生存验证
```

这意味着：

- `scan` 负责真实基因组候选发现。
- `build-controls` 负责构建训练和评估所需的正负控制。
- 后续的 baseline / OOD / deep model 训练模块会以 `build-controls` 的 CSV 为输入。

## 已继续迁移的模型层

在控制组层之后，V12 的核心模型层也已并入 `genome_mining`：

```text
genome_mining/models/pe_nac_plus.py
genome_mining/models/pe_nac_v11.py
genome_mining/models/deep_carrier_model.py
genome_mining/models/deep_encoders.py
genome_mining/model_metrics.py
genome_mining/model_training.py
genome_mining/evidence_fusion.py
genome_mining/ood_baselines.py
genome_mining/localization.py
genome_mining/torchdata.py
```

这些模块保留 V12 的核心输出语义：

```text
predictions.csv
metrics.json
train_log.json
model.pt
embeddings.npz
```

## 可选模型依赖

基础扫描和 `build-controls` 不依赖 PyTorch；模型训练需要额外环境：

```bash
python -m pip install -r requirements-genome-v12.txt
```

服务器上已有 PyTorch/CUDA 环境时，直接激活该环境即可。

## 训练 PE-NAC++

先跑传统 k-mer baseline：

```bash
python -m genome_mining train-baseline \
  --csv artifacts/genome_mining/ecoli_v12_controls.csv \
  --out artifacts/genome_mining/ecoli_logreg \
  --model logreg \
  --k 4
```

```bash
python -m genome_mining train-pe-nac-plus \
  --csv artifacts/genome_mining/ecoli_v12_controls.csv \
  --out artifacts/genome_mining/ecoli_pe_nac_plus \
  --epochs 10 \
  --batch-size 64 \
  --device cuda
```

## 训练 EF-PE-NAC++

```bash
python -m genome_mining train-ef-pe-nac-plus \
  --csv artifacts/genome_mining/ecoli_v12_controls.csv \
  --out artifacts/genome_mining/ecoli_ef_pe_nac_plus \
  --epochs 10 \
  --batch-size 64 \
  --device cuda
```

## 训练 V12 DeepOpenCarrier

```bash
python -m genome_mining train-deep-carrier \
  --csv artifacts/genome_mining/ecoli_v12_controls.csv \
  --out artifacts/genome_mining/ecoli_v12_hybrid \
  --encoder hybrid \
  --epochs 10 \
  --batch-size 64 \
  --device cuda
```

可选 encoder：

```text
cnn
dilated_cnn
transformer
hybrid
```

## 评估 OpenOOD 指标

```bash
python -m genome_mining evaluate-openood \
  --pred artifacts/genome_mining/ecoli_ef_pe_nac_plus/predictions.csv \
  --out artifacts/genome_mining/ecoli_ef_pe_nac_plus/openood_unknown.json \
  --target unknown
```

运行 V12 OOD baseline suite：

```bash
python -m genome_mining run-ood-suite \
  --pred artifacts/genome_mining/ecoli_v12_hybrid/predictions.csv \
  --emb artifacts/genome_mining/ecoli_v12_hybrid/embeddings.npz \
  --out artifacts/genome_mining/ecoli_v12_hybrid/ood_suite_unknown.json \
  --target unknown
```

## 训练 evidence fusion

```bash
python -m genome_mining train-evidence-fusion \
  --calibration-pred artifacts/genome_mining/ecoli_ef_pe_nac_plus/validation_predictions.csv \
  --pred artifacts/genome_mining/ecoli_ef_pe_nac_plus/predictions.csv \
  --out artifacts/genome_mining/ecoli_ef_pe_nac_plus/predictions_calibrated.csv \
  --model-out artifacts/genome_mining/ecoli_ef_pe_nac_plus/evidence_fusion.joblib \
  --target unknown
```

## 当前状态

上述候选提取、真实候选评分、baseline、独立验证集校准、OOD suite、定位评估、embedding 可视化、失败分析、leaderboard、编码器比较、消融和 leave-one-codec 矩阵现已迁移完成。完整命令和迁移边界见 `docs/genome_mining_v12_complete_migration_cn.md`。
