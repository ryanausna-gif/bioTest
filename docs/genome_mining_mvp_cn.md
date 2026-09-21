# genome_mining MVP 使用说明

`genome_mining` 是当前项目里新增的真实基因组挖掘层。它的第一版目标不是证明某段 DNA 存在人工来源，而是从 FASTA 中做坐标感知滑窗扫描，提取确定性统计特征，输出可复查的异常候选区域。

## 当前能力

- 读取普通 FASTA。
- 按 `chrom/start/end/strand` 生成固定长度滑窗。
- 计算 GC、熵、二/三核苷酸熵、压缩率、最长同聚物、k-mer 重复、周期性等特征。
- 使用鲁棒中位数/MAD 建立背景基线。
- 输出异常分数、主要证据、候选 BED/CSV、全窗口特征 CSV、Markdown 报告和 JSON 摘要。

## 快速运行

```bash
python -m genome_mining scan \
  --fasta reference.fa \
  --window 1000 \
  --step 500 \
  --threshold 2.5 \
  --top-n 50 \
  --out artifacts/genome_mining/reference_scan
```

调试小数据时可以限制窗口数：

```bash
python -m genome_mining scan \
  --fasta reference.fa \
  --window 1000 \
  --step 500 \
  --max-windows 10000 \
  --out artifacts/genome_mining/debug_scan
```

默认不会把每个窗口的原始序列写入 CSV，避免全基因组输出过大。若确实需要：

```bash
python -m genome_mining scan \
  --fasta reference.fa \
  --include-sequence \
  --out artifacts/genome_mining/with_sequence
```

## 输出文件

```text
windows.features.csv   # 所有窗口的特征和异常分数
candidates.csv         # 超过阈值后的候选窗口
candidates.bed         # 可直接与注释 BED/GFF 工具相交的候选区域
summary.json           # 本次扫描参数和背景基线
mining_report.md       # 人可读报告
```

## 当前边界

这还是第一层筛选器。候选区域必须继续做：

1. 与 repeat、转座子、低复杂度、CpG island、gene/intron/pseudogene 注释相交。
2. 多窗口尺度复扫，保留跨尺度稳定候选。
3. 与 shuffled / chromosome-held-out 背景比较，控制 false positive。
4. 送入后续 decoder 和 `genome_survival_sim` 做解码尝试与遗传生存验证。
