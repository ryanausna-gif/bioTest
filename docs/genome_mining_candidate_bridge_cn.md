# genome_mining 候选桥接层

这份文档说明 `genome_mining` 如何从“真实基因组扫描”进入“V12 模型打分”，并为后续 decoder/recovery 和 `genome_survival_sim` 做准备。

## 新增命令

```text
extract-candidates
score-candidates
```

`extract-candidates` 负责：

```text
FASTA + candidates.bed/candidates.csv
-> candidate_sequences.csv
-> candidate_sequences.fa
```

`score-candidates` 负责：

```text
candidate_sequences.csv + V12 model.pt / baseline model.joblib
-> candidate_v12_scores.csv
```

## 典型流程

第一步，扫描真实基因组：

```bash
python -m genome_mining scan \
  --fasta data/raw/ncbi/ecoli_k12_mg1655/GCF_000005845.2_ASM584v2_genomic.fna \
  --window 1000 \
  --step 500 \
  --threshold 2.5 \
  --top-n 200 \
  --out artifacts/genome_mining/ecoli_scan
```

第二步，提取候选序列。`--target-length` 建议与训练模型的 `length` 一致：

```bash
python -m genome_mining extract-candidates \
  --fasta data/raw/ncbi/ecoli_k12_mg1655/GCF_000005845.2_ASM584v2_genomic.fna \
  --candidates artifacts/genome_mining/ecoli_scan/candidates.bed \
  --target-length 1000 \
  --out artifacts/genome_mining/ecoli_scan/candidate_sequences.csv \
  --out-fasta artifacts/genome_mining/ecoli_scan/candidate_sequences.fa
```

第三步，用训练好的 V12 模型打分：

```bash
python -m genome_mining score-candidates \
  --candidates artifacts/genome_mining/ecoli_scan/candidate_sequences.csv \
  --model artifacts/genome_mining/ecoli_ef_pe_nac_plus/model.pt \
  --out artifacts/genome_mining/ecoli_scan/candidate_v12_scores.csv \
  --device cuda
```

如果已经用验证集训练了 unknown calibrator，可直接应用到真实候选：

```bash
python -m genome_mining score-candidates \
  --candidates artifacts/genome_mining/ecoli_scan/candidate_sequences.csv \
  --model artifacts/genome_mining/ecoli_v12_hybrid/model.pt \
  --unknown-calibrator artifacts/genome_mining/ecoli_v12_hybrid/unknown_calibrator.joblib \
  --out artifacts/genome_mining/ecoli_scan/candidate_v12_scores.csv \
  --device cuda
```

提取器按 FASTA contig 流式处理，不再一次性载入整个人类参考基因组，并支持负链定位坐标换算。

## 输出字段

`candidate_sequences.csv` 主要字段：

```text
candidate_id
chrom
start
end
strand
extract_start
extract_end
left_pad
right_pad
sequence_length
sequence
```

`candidate_v12_scores.csv` 主要字段：

```text
special_prob
naturalness_prob
known_conf
unknown_score
prototype_unknown_score
mixture_prototype_unknown_score
energy_unknown_score
pred_family
pred_region_start_local
pred_region_end_local
pred_region_start
pred_region_end
localization_max_score
sparse_site_max_score
```

其中：

```text
pred_region_start_local / pred_region_end_local
```

是模型在提取序列中的局部坐标；

```text
pred_region_start / pred_region_end
```

是换算回参考基因组后的坐标。

## 和三层架构的关系

```text
发现层 genome_mining:
  scan
  extract-candidates
  score-candidates

还原层 decoder/recovery:
  输入 candidate_sequences.fa + candidate_v12_scores.csv
  优先尝试 unknown_score 高、localization 明确的候选

仿真层 genome_survival_sim:
  输入候选坐标、疑似载荷区间、候选长度和可能的 carrier 类型
  评估突变/重组/生殖过程中的长期存活可能性
```

## 后续还需要继续搬/补的内容

1. `annotation bridge`：候选与 GFF/GTF/repeat/CpG/low-complexity 注释相交。
2. `decoder bridge`：把候选序列和 V12 打分结果送入 decoder/recovery。
3. `survival bridge`：把候选坐标、疑似载荷区间送入 `genome_survival_sim`。

leaderboard/report 和 `.joblib` baseline 候选评分已经完成，不再属于待迁移项。
