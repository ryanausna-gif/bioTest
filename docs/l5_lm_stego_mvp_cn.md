# L5 上下文条件生成式 DNA 隐写 MVP

## 1. 本次实现的边界

本次实现的是 `lm_arithmetic_v1` 协议原型，目标是先证明：

```text
任意 bytes
-> 压缩 / ChaCha20-Poly1305 / 可选 Reed-Solomon
-> 上下文模型 P(A/C/G/T | context)
-> 确定性整数 CDF
-> 可逆分块逆算术编码
-> DNA
-> 使用相同模型和左侧 context 完整恢复 bytes
```

这里的 `L5-prototype` 表示编码选择已经由上下文概率模型控制，不表示短 k-mer
模型已经达到 DNA foundation model 的自然性。当前可序列化模型是可复现的 k-mer
next-base baseline；HyenaDNA causal adapter 已提供，但必须先确认 checkpoint 暴露
next-token logits，才能用于正式 A800 实验。

## 2. 代码结构

```text
genome_mining/lm_stego/
  integer_distribution.py  # 概率舍入、uniform mix、整数 CDF
  model_adapter.py         # k-mer 模型、HF causal DNA 模型接口
  arithmetic_codec.py      # 精确、自定界的分块区间编码
  codec.py                 # AEAD / RS / frame / L5 codec
```

`lm_arithmetic_v1` 已接入：

```text
make_stego_codec
list-stego-codecs
build-image-dataset
evaluate-image-locator
```

## 3. 为什么 block 不需要显式 DNA 分隔符

一个 `k` bit block 对应 `[0, 1)` 中的一个二进制 cell。编码器选取该 cell
内部一个不落在二进制 CDF 边界上的有理点，然后根据上下文模型不断选择碱基、
缩小序列区间。当序列区间完全进入目标 cell 时，该 block 结束。

解码器使用相同模型重放区间更新，并在同一个条件成立时恢复该 cell 编号。因此
block 可以自定界，不需要在每个 block 之间写入固定 marker。

当前 packet 为：

```text
raw payload
-> zlib（仅在变小时启用）
-> ChaCha20-Poly1305
-> optional Reed-Solomon
-> L5A1 + protected_length + CRC32 + protected_payload + keyed padding
-> fixed-size bit blocks
-> model-conditioned DNA
```

## 4. 服务器环境

先使用已有 Conda 环境，并绑定物理 GPU3：

```bash
conda activate image-dna-hyena
export CUDA_VISIBLE_DEVICES=3

python -m pip install -r requirements-hyenadna-stego.txt
python -c "import cryptography, reedsolo, numpy, PIL; print('L5 dependencies OK')"
```

k-mer 协议 baseline 本身不需要 GPU。HyenaDNA 生成概率接入后才使用 GPU。

## 5. 从真实 FASTA 冻结上下文模型

第一轮只读取训练 donor 或独立参考基因组中的一部分碱基，不要使用 test donor
专门拟合上下文模型：

```bash
export PROJECT=/public/home/lifex/jiangjr/dpproject/projects/coding_recognition/cod
export T2T=/public/home/lifex/jiangjr/biolearning/database/data/raw/ncbi/human_t2t_chm13v2
cd "$PROJECT"

python -m genome_mining train-l5-context-model \
  --fasta "$T2T/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna" \
  --out artifacts/l5/context_model_order6.json \
  --order 6 \
  --max-bases 5000000
```

输出 JSON 包含：

```text
model_type
model_id
order
trained_bases
reverse-complement 设置
每个 context 的 A/C/G/T counts
```

解码时会重新计算模型指纹；文件被修改后会拒绝加载。

## 6. 任意文件的 L5 round-trip

生成实验密钥：

```bash
mkdir -p data/keys artifacts/l5/roundtrip
openssl rand -out data/keys/l5_image.key 32
chmod 600 data/keys/l5_image.key
```

准备一段左侧 context。可以从 FASTA 提取，也可以先使用一个小 FASTA 文件。然后：

```bash
python -m genome_mining test-l5-roundtrip \
  --input path/to/test_image.png \
  --model artifacts/l5/context_model_order6.json \
  --codec-key-file data/keys/l5_image.key \
  --context-file path/to/left_context.fa \
  --dna-out artifacts/l5/roundtrip/payload.dna.txt \
  --recovered-out artifacts/l5/roundtrip/recovered.png \
  --sample-id image_000001 \
  --block-bytes 8 \
  --max-symbols-per-block 256 \
  --uniform-mix 0.20
```

输出中的 `exact` 必须为 `true`。`effective_bits_per_base` 是这一次真实编码长度
计算出的有效容量，不是 codec catalog 中的固定常数。

## 7. 构建 64x64 L5 图像定位数据集

128x128 的保守长度上界可能超过 160k 模型窗口。第一轮使用 64x64：

```bash
python -m genome_mining build-image-dataset \
  --genome-manifest data/hprc/genome_manifest.csv \
  --image-root data/imagenette/train \
  --out artifacts/l5/image64_l5 \
  --n-train 1000 \
  --n-val 200 \
  --n-test 300 \
  --positive-fraction 0.5 \
  --image-size 64 \
  --window-length 159744 \
  --codecs lm \
  --codec-key-file data/keys/l5_image.key \
  --lm-model-file artifacts/l5/context_model_order6.json \
  --lm-block-bytes 8 \
  --lm-max-symbols-per-block 256 \
  --lm-context-bases 4096 \
  --lm-cdf-precision-bits 12 \
  --lm-uniform-mix 0.20 \
  --ecc-symbols 16 \
  --seed 2026
```

注意：k-mer 模型的有效 context 长度等于其 `order`。`--lm-context-bases 4096`
是上限，不会把 order-6 模型伪装成 4096 bp 模型。

数据审计：

```bash
python -m genome_mining audit-image-dataset \
  --dataset artifacts/l5/image64_l5 \
  --out artifacts/l5/image64_l5/audit.json

python -m genome_mining summarize-stego-dataset \
  --dataset artifacts/l5/image64_l5 \
  --out artifacts/l5/image64_l5/stego_summary
```

## 8. 训练和恢复

CNN smoke test：

```bash
python -m genome_mining train-image-locator \
  --dataset artifacts/l5/image64_l5 \
  --out artifacts/l5/image64_l5_cnn \
  --backbone cnn \
  --epochs 5 \
  --batch-size 4 \
  --device cuda \
  --amp \
  --codec-key-file data/keys/l5_image.key
```

评价：

```bash
python -m genome_mining evaluate-image-locator \
  --dataset artifacts/l5/image64_l5 \
  --model artifacts/l5/image64_l5_cnn/model.pt \
  --out artifacts/l5/image64_l5_cnn/test_recovery \
  --split test \
  --device cuda \
  --codec-key-file data/keys/l5_image.key \
  --save-images
```

恢复器会从预测起点向左提取与 manifest 一致的 context。预测起点错一位时，
context 和算术区间都会改变，AEAD/CRC 会明确拒绝错误结果。

## 9. 当前验收结果

本地已经验证：

```text
整数 CDF 确定性和总质量
不同长度与最后不足 block 的 payload round-trip
无 expected length 时的自定界 block 解码
k-mer 模型保存、加载和 SHA-256 指纹
ChaCha20-Poly1305 L5 packet round-trip
L5 图像数据集构建、manifest、审计和像素精确恢复
L0-L4 原有 codec 回归
```

## 10. 尚未完成

这版不能用于宣称“HyenaDNA L5 已经自然不可检测”。仍需完成：

```text
确认并固定能输出 next-base logits 的 HyenaDNA/Evo2 checkpoint
高效增量推理或 block 批量生成，避免每个碱基完整重跑模型
同一 checkpoint 在编码/解码设备间的整数 CDF 一致性测试
右侧 flank 重排序和独立 Caduceus/NT 自然性评价
Top-K 边界恢复
substitution 后的 block 隔离与 ECC
indel 同步/HEDGES
L5 与 L3/L4 的开放集检测对照
```

因此当前最准确的阶段名称是：

```text
L5 可逆协议 MVP + k-mer reproducibility baseline + HF causal model adapter
```

