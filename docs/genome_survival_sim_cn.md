# 基因组编码数据生存率仿真器设计

本模块用于把“编码 DNA 在生殖和自然突变中能否长期可解码”转化为可运行仿真。第一版聚焦单个后代谱系，不模拟完整群体释放场景，也不提供任何湿实验操作流程。

## MVP 范围

- 物种预设：人类 T2T 尺度、小鼠 GRCm39 尺度、果蝇 dm6 尺度、toy genome。
- 世代数：默认几十代。
- 目标：随机单个后代个体是否仍可完整解码。
- 插入位置：默认随机非编码候选区标签；可用 `--candidate-bed` 输入候选 BED。
- 失败原因拆分：未遗传、标识符失效、block 丢失/突变、认证失败、payload mismatch。

## 编码方案

默认 payload 结构：

```text
[SYNC_MARKER][HEADER][CIPHERTEXT][AUTH_TAG]
```

启用 Reed-Solomon 后，`SYNC_MARKER` 仍保持在最外层用于定位；marker 后面的 `HEADER + CIPHERTEXT + AUTH_TAG` 会整体进入 RS 编码：

```text
[SYNC_MARKER][RS_ENCODED(HEADER + CIPHERTEXT + AUTH_TAG)]
```

启用 XOR parity 擦除恢复后，每组数据 block 会额外生成 1 个 parity block：

```text
data blocks:   B0, B1, ..., B7
parity block:  P = B0 xor B1 xor ... xor B7
```

启用 fountain 擦除恢复后，每组数据 block 会生成多个 GF(256) repair block：

```text
data blocks:    B0, B1, ..., B7
repair blocks:  R0, R1, R2, R3 ...
```

启用 chunked resync 后，单个 fragment 的 RS 编码结果会被切成多个带内部同步标记的 chunk：

```text
[SYNC_MARKER]
  [CHUNK_MARKER][CHUNK_HEADER][CHUNK_BYTES]
  [CHUNK_MARKER][CHUNK_HEADER][CHUNK_BYTES]
  ...
```

其中：

- `SYNC_MARKER` 由密钥和协议版本派生，模拟不易被直接识别的标识符；payload id 放在 header 中。
- `HEADER` 包含版本魔数、payload id、block id、总 block 数、copy id、nonce、明文长度和 payload 总长度。
- `CIPHERTEXT` 支持两种模式：`hmac_stream` 和 `chacha20_poly1305`。
- `AUTH_TAG` 在 `hmac_stream` 中使用 HMAC 截断标签；在 `chacha20_poly1305` 中使用 AEAD 自带 16 字节认证标签。
- `RS_ENCODED` 由 `--ecc-mode reed_solomon --ecc-symbols N` 开启，使用可选依赖 `reedsolo` 做字节级纠错。
- `xor_parity` 由 `--erasure-mode xor_parity --erasure-group-size N` 开启，不需要额外依赖；每组最多恢复 1 个完整缺失的数据 block。
- `fountain` 由 `--erasure-mode fountain --erasure-repair-blocks N` 开启，不需要额外依赖；每组可恢复多个完整缺失的数据 block，前提是 repair block 数量和线性秩足够。
- `chunked resync` 由 `--resync-mode chunked --resync-chunk-bytes N` 开启，用内部 chunk marker 在局部 indel 后重新找回后续 chunk。

`hmac_stream` 是无额外依赖的仿真模式，用于本地测试和对照。`chacha20_poly1305` 是真实 AEAD 模式，需要安装 `cryptography`，更适合正式评估突变后认证失败/篡改检测行为。
`reed_solomon` 能纠正少量碱基替换映射成的字节错误；配合 `chunked resync` 时，发生 indel 的 chunk 会被标记为 erasure，再由 RS 尝试恢复。
`xor_parity` 主要处理 block 级擦除，例如某个 block 的所有拷贝都丢失或认证失败；它不能处理同一组内两个及以上数据 block 同时丢失的情况。
`fountain` 是更推荐的 block 级擦除恢复层，适合对比同组丢失 2 个以上 block 的场景，但会增加额外 repair block 的存储开销。

## 遗传模型

每个个体是双倍体。每一代只跟踪目标后代谱系：

```text
当前个体 -> 产生一个带重组的配子 -> 与野生型配子结合 -> 下一代目标个体
```

这能回答“某个单个后代是否仍能解码”的问题。若以后要回答“群体中至少有一个个体能否解码”，应接入 SLiM 或 simuPOP。

## 运行示例

```bash
python -m genome_survival_sim simulate \
  --payload input.png \
  --species human_t2t \
  --generations 30 \
  --replicates 1000 \
  --strategy single_heterozygous_copy multi_locus_redundant \
  --mutation-rate-multiplier 1 10 100 \
  --out artifacts/human_payload_survival
```

输出：

- `all_generation_metrics.csv`
- `all_summary.json`
- `recommendations.md`
- 每个策略/突变率组合的独立子目录

## 真实注释数据接入

可以用 `make-candidates` 从 `.fai/chrom.sizes` 和 GFF/GTF 生成候选 BED：

```bash
python -m genome_survival_sim make-candidates \
  --chrom-sizes reference.fa.fai \
  --annotation annotation.gff3 \
  --mode intergenic \
  --min-length 1000 \
  --flank 1000 \
  --out data/candidates/intergenic.bed
```

支持的候选模式：

- `intergenic`：排除 gene-like feature 及其 flank 后取互补区。
- `intronic`：gene 区间减去 exon/CDS/UTR。
- `pseudogene`：提取 pseudogene-like feature。

生成的 BED 可继续传给仿真：

```bash
python -m genome_survival_sim simulate \
  --payload input.png \
  --species human_t2t \
  --chrom-sizes reference.fa.fai \
  --candidate-bed data/candidates/intergenic.bed \
  --generations 30 \
  --replicates 1000 \
  --strategy multi_locus_redundant \
  --copies-per-block 16 \
  --encryption-mode chacha20_poly1305 \
  --ecc-mode reed_solomon \
  --ecc-symbols 16 \
  --erasure-mode fountain \
  --erasure-group-size 8 \
  --erasure-repair-blocks 4 \
  --resync-mode chunked \
  --resync-chunk-bytes 32 \
  --out artifacts/human_intergenic
```

## 批量实验

可以用 JSON 定义实验矩阵：

```bash
python -m genome_survival_sim batch \
  --config docs/example_batch_config_tiny.json \
  --out artifacts/batch_tiny
```

较大的配置可参考 `docs/example_batch_config.json`。

批量实验会额外生成成本-收益分析文件：

- `batch_design_tradeoffs.csv`：每个设计组合的编码总碱基数、单位 payload 开销、最终解码率、半衰代数和 Pareto 标记。
- `batch_tradeoff.svg`：编码总碱基数与最终解码率的散点图，黑色描边点表示 Pareto-efficient 组合。
- `batch_report.md`：包含最优组合表和成本-收益前沿摘要。

## SLiM 桥接

当前 Python backend 负责 payload 级别的精确解码分析。若需要群体级 allele survival 交叉验证，可以先生成 SLiM marker-tracking 脚本：

```bash
python -m genome_survival_sim generate-slim \
  --out artifacts/slim/payload_bridge.slim \
  --population-size 1000 \
  --generations 50 \
  --payload-copy-count 16 \
  --tree-seq-output payload_bridge.trees
```

如果服务器已安装 `slim`，可以直接运行并解析：

```bash
python -m genome_survival_sim run-slim \
  --script artifacts/slim/payload_bridge.slim \
  --out artifacts/slim_run \
  --population-size 1000 \
  --tree-sequence artifacts/slim/payload_bridge.trees
```

也可以解析已有 stdout：

```bash
python -m genome_survival_sim parse-slim \
  --stdout docs/example_slim_stdout.txt \
  --out artifacts/slim_parse_demo \
  --population-size 10
```

输出：

- `slim_stdout.txt`
- `slim_stderr.txt`
- `slim_marker_metrics.csv`
- `slim_summary.json`
- `slim_report.md`

如果已有 `.trees` 文件，可以单独分析：

```bash
python -m genome_survival_sim analyze-trees \
  --tree-sequence artifacts/slim/payload_bridge.trees \
  --out artifacts/tree_sequence_analysis
```

## msprime 中性基线

`msprime` 可用于快速生成中性 ancestry/mutation baseline，用来和 Python payload 仿真、SLiM marker 生存结果对照：

```bash
python -m genome_survival_sim run-msprime \
  --out artifacts/msprime_baseline \
  --samples 100 \
  --sequence-length 1000000 \
  --population-size 10000 \
  --recombination-rate 1e-8 \
  --mutation-rate 1e-8 \
  --payload-bp 1000
```

输出：

- `msprime_baseline.trees`
- `msprime_summary.json`
- `msprime_report.md`

## stdpopsim 物种 catalog

`stdpopsim` 用来减少手写物种参数。当前先提供 catalog 入口：

```bash
python -m genome_survival_sim list-stdpopsim \
  --out artifacts/stdpopsim_catalog
```

输出：

- `stdpopsim_species.csv`
- `stdpopsim_species.json`
- `stdpopsim_species_report.md`

推荐安装可选依赖：

```bash
conda install -c conda-forge cryptography slim tskit pyslim msprime stdpopsim -y
python -m pip install reedsolo
```

这个脚本只追踪 payload marker 在群体中的存活，不替代 Python 的 block/auth 解码分析。它的作用是给 Python payload 仿真提供群体遗传层面的交叉验证。

## 推荐实验

第一批建议比较：

```text
species: human_t2t, mouse, fly
generations: 10, 20, 30, 50
strategies: single_heterozygous_copy, homozygous_same_locus, multi_locus_redundant
mutation multipliers: 1, 10, 100
copies_per_block for multi_locus_redundant: 8, 16, 32
```

单杂合位点通常会快速因为孟德尔分离而丢失。多位点冗余可以区分“遗传丢失”和“突变破坏”两类风险，更适合后续设计优化。
