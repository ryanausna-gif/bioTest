# 基因组编码数据长期存活率仿真建议

## 当前结论

如果评价目标是“单个随机后代仍可解码”，第一风险不是突变，而是遗传分离。

单个杂合插入位点在每代与未携带者繁殖时，目标后代继承该位点的概率约为 `1/2`。因此第 `n` 代仍继承的概率近似为：

```text
(1/2)^n
```

几十代后，这个概率会非常低。仿真器输出中的 `not_inherited` 就是用于量化这一类失败。

## 推荐实验顺序

1. 先用 toy genome 跑通参数扫描。
2. 再分别跑 `human_t2t`、`mouse`、`fly` 三个物种预设。
3. 每个物种先比较：

```text
single_heterozygous_copy
homozygous_same_locus
multi_locus_redundant
```

4. 多位点冗余策略下扫描：

```text
copies_per_block = 4, 8, 16, 32, 64
mutation_rate_multiplier = 1, 10, 100
generations = 10, 20, 30, 50
```

## 对编码方案的建议

第一版采用：

```text
SYNC_MARKER + HEADER + CIPHERTEXT + AUTH_TAG
```

建议保留这个结构。当前已经支持两种加密认证模式、一个可选的 Reed-Solomon 纠错层、XOR parity/fountain 两种 block 级擦除恢复层，以及一个 chunked resync 同步恢复层：

- `hmac_stream`：无依赖仿真模式，适合 smoke test 和对照实验。
- `chacha20_poly1305`：真实 AEAD 模式，需要安装 `cryptography`，适合正式评估认证失败/篡改检测行为。
- `reed_solomon`：通过 `--ecc-mode reed_solomon --ecc-symbols N` 启用，需要安装 `reedsolo`，用于评估少量字节错误被纠正后的 block 生存率。
- `xor_parity`：通过 `--erasure-mode xor_parity --erasure-group-size N` 启用，不需要额外依赖，每组可恢复 1 个完整缺失的数据 block。
- `fountain`：通过 `--erasure-mode fountain --erasure-repair-blocks N` 启用，不需要额外依赖，使用 GF(256) 线性 repair block 支持同组多个 block 丢失。
- `chunked`：通过 `--resync-mode chunked --resync-chunk-bytes N` 启用，用内部 chunk marker 在局部 indel 后重新同步，并把坏 chunk 交给 RS 作为 erasure。

后续升级重点：

- 系统扫描 `erasure_group_size`、`erasure_repair_blocks` 和 copies-per-block 的开销/收益曲线。
- 后续可加入标准 Reed-Solomon erasure code 或 RaptorQ 类 fountain code 做对照。
- 继续量化 chunk 大小、marker 长度和 RS symbols 的组合，找到 indel 场景下的最优开销。
- 标识符要多拷贝，并允许少量错配扫描。
- payload 必须分块，不要把整张图片或大文件编码成一整段连续序列。
- 加密密钥不得写入基因组，基因组中只保留密文、认证标签和必要的公开参数。

## 对插入位置的建议

MVP 阶段不要过早追求复杂安全港数据库。建议：

- 第一阶段：使用随机候选 BED，过滤掉编码区和高保守区。
- 第二阶段：比较 intergenic、intronic、pseudogene-like 三类候选区域。
- 第三阶段：为人、小鼠、果蝇分别整理物种特异候选 BED。

注意：非功能区域不一定会被选择主动清除，但更可能累积突变；功能区域可能突变率低一些，却会带来适应度风险。因此插入位置选择必须同时评估“数据完整性”和“适应度代价”。

## 对项目路线的建议

短期：

- 先用当前 Python MVP 完成几十代、单后代谱系仿真。
- 输出 CSV 后用 R/Python 画出解码率曲线。
- 找出哪些失败由遗传分离造成，哪些由突变造成。

中期：

- 接入真实 `.fai` 染色体长度和非编码候选 BED。
- 引入真实重组图谱。
- 在 `chacha20_poly1305` 基础上加入真正的纠错码。

长期：

- 如果目标从“单个后代”变成“群体中至少一个个体仍可解码”，接入 SLiM 做群体级 forward simulation。
- 比较死体 DNA 存储与活体遗传存储的主要差异：活体方案的关键不是保存一次，而是跨世代可恢复。
