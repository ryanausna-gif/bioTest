# genome_mining V12 完整功能迁移

## 定位

`genome_mining` 是三层系统的发现层主入口，旧 `dna_semantic_carrier_v12` 是其模型和实验方法来源：

```text
真实 FASTA 扫描
  -> 候选坐标和序列
  -> V10 PE-NAC++ / V11 EF-PE-NAC++ / V12 DeepOpenCarrier
  -> 校准、OOD、定位、失败分析和候选排序
  -> decoder/recovery
  -> genome_survival_sim
```

本次按“能力”迁移，不保留 `ospd_ml`、`dna_carrier_v8` 等历史包名。旧压缩包中引用缺失脚本的实验链已经改成统一的 `python -m genome_mining` 命令。

## 已迁移能力

模型与数据：

- V8-V12 carrier codec、困难阴性和真实 FASTA 背景。
- V10 PE-NAC++、mixture prototypes、energy/prototype unknown score。
- V11 EF-PE-NAC++、dense/sparse localization、evidence fusion。
- V12 CNN、DilatedCNN、Transformer、Hybrid encoder、层级对比学习和 episodic pseudo-unknown。
- k-mer + logreg/RF/GB baseline。
- 外部 foundation embedding 的轻量分类头接口。

评估与实验：

- 独立 calibration codec 和最终 holdout codec。
- validation/test 分离的预测与 embedding 输出。
- unknown score calibration 和 evidence fusion。
- OpenOOD 指标、MSP、Energy、Mahalanobis、KNN、feature norm。
- region IoU、token F1、boundary error 和按 embedding type 分组定位。
- benchmark audit、benchmark card、failure analysis。
- PCA、t-SNE、UMAP embedding 图。
- leaderboard、paired bootstrap、encoder comparison、ablation、leave-one-codec matrix。
- 一键 V10/V11/V12 完整实验套件。

真实候选桥：

- 从 BED/CSV + FASTA 提取候选序列。
- 提取时只保留当前 FASTA contig，避免一次性加载整个人类基因组。
- 支持正负链并将局部定位坐标换算回基因组坐标。
- 支持 `.pt` 深度模型和 `.joblib` baseline。
- 支持把 unknown calibrator 或 evidence-fusion 模型应用到真实候选。

## 修复的旧 V12 问题

旧版本验证集没有 unknown 样本，却在验证集拟合 unknown calibrator；部分脚本还引用了压缩包内不存在的 `make_leaderboard.py`、`summarize_ablations.py`、`audit_dataset.py` 和 `train_carrier_baseline.py`。

当前 benchmark 将 codec 分成三组：

```text
train_codecs        模型训练的已知族
calibration_codecs  仅用于验证集校准的伪未知族
holdout_codecs      仅用于最终测试的未知族
```

训练命令会输出：

```text
model.pt
metrics.json
train_log.json
predictions.csv
embeddings.npz
validation_predictions.csv
validation_embeddings.npz
```

OOD 的 KNN 和 Mahalanobis 可以用 validation embedding 作为独立 reference，避免测试集自参考。

## A800 GPU3 一键实验

首次创建环境：

```bash
conda env create -f environment-genome-v12-a800.yml
conda activate genome-mining-v12
python -m pip install -r requirements-genome-v12.txt
```

`requirements-genome-v12.txt` 不再通过 pip 安装或覆盖 PyTorch；CUDA 版 PyTorch 由 Conda 环境文件负责。

在服务器项目根目录执行：

```bash
conda activate genome-mining-v12
export CUDA_VISIBLE_DEVICES=3

python -m genome_mining run-v12-suite \
  --out-root artifacts/t2t_v12_suite \
  --background-fasta /public/home/lifex/jiangjr/biolearning/database/data/raw/ncbi/human_t2t_chm13v2/GCF_009914755.1_T2T-CHM13v2.0_first5Mb.fna \
  --n-train 6000 \
  --n-val 1500 \
  --n-test 2500 \
  --length 1000 \
  --epochs 10 \
  --batch-size 64 \
  --encoders cnn dilated_cnn transformer hybrid \
  --device cuda \
  --seed 42
```

先检查命令链但不运行训练：

```bash
python -m genome_mining run-v12-suite \
  --out-root artifacts/t2t_v12_suite_dry \
  --background-fasta /path/to/first5Mb.fna \
  --device cuda \
  --dry-run
```

## 单独校准与 OOD

```bash
RUN=artifacts/t2t_v12_suite/deep_encoders/hybrid

python -m genome_mining calibrate-unknown \
  --calibration-pred "$RUN/validation_predictions.csv" \
  --pred "$RUN/predictions.csv" \
  --out "$RUN/predictions_calibrated.csv" \
  --model-out "$RUN/unknown_calibrator.joblib"

python -m genome_mining run-ood-suite \
  --pred "$RUN/predictions_calibrated.csv" \
  --emb "$RUN/embeddings.npz" \
  --reference-pred "$RUN/validation_predictions.csv" \
  --reference-emb "$RUN/validation_embeddings.npz" \
  --out "$RUN/ood_suite_unknown.json" \
  --target unknown
```

## 真实 T2T 候选评分

```bash
python -m genome_mining extract-candidates \
  --fasta /path/to/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna \
  --candidates artifacts/t2t_scan/candidates.bed \
  --target-length 1000 \
  --out artifacts/t2t_scan/candidate_sequences.csv \
  --out-fasta artifacts/t2t_scan/candidate_sequences.fa

python -m genome_mining score-candidates \
  --candidates artifacts/t2t_scan/candidate_sequences.csv \
  --model artifacts/t2t_v12_suite/deep_encoders/hybrid/model.pt \
  --unknown-calibrator artifacts/t2t_v12_suite/deep_encoders/hybrid/unknown_calibrator.joblib \
  --out artifacts/t2t_scan/candidate_v12_scores.csv \
  --batch-size 128 \
  --device cuda
```

真实基因组没有 `y/is_unknown` 真值，因此不能直接在真实候选上报告 AUROC。真实候选阶段应报告排序、跨模型一致性、repeat/低复杂度注释、跨物种保守性和后续解码证据。

## 未作为运行时核心复制的内容

旧项目的 LaTeX 论文生成器和方法示意图脚本没有并入核心包。它们不参与训练、推理或科学指标计算，而且旧版本包含硬编码路径和断链引用。论文图和文稿应在实验结果稳定后单独生成，避免与发现层运行时耦合。
