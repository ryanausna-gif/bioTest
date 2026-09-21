# HyenaDNA 图像-DNA 隐写检测与定位

## 1. 已实现的研究问题

这一版把原有固定 2-bit 图像插入实验升级为统一的可逆隐写基准：

```text
真实供体 FASTA + 不重复图片
        -> 分级隐写 codec
        -> 可选替换/indel 信道
        -> donor/image/interval 无泄漏数据集
        -> CNN 或 HyenaDNA 检测与起止边界定位
        -> 授权解码、图片恢复、分 codec 和自然性评价
```

模型只负责检测和定位。压缩、加密、认证、约束映射和 Reed-Solomon 由可解释的编码层完成。

## 2. 隐写等级

| 等级 | CLI 名称 | 编码方式 | 128x128 灰度图最大 DNA 长度（无 RS） |
|---|---|---|---:|
| L0 | `direct` | `00=A, 01=T, 10=C, 11=G` | 65,536 |
| L1 | `encrypted` | 压缩 + ChaCha20-Poly1305 + 2-bit | 65,668 |
| L2 | `constrained` | 加密字节映射到 6-base GC/同聚物约束码字 | 98,502 |
| L3 | `kmer` | 在多个合法码字中选择更匹配局部 cover/k-mer 的码字 | 98,502 |
| L4 | `cover` | 每 7 个自然 cover 碱基嵌入 3 bit，每组至多修改 1 位 | 306,453 |
| L5 原型 | `lm` | 上下文模型整数 CDF + 自定界逆算术分块 | 可变，64x64 起步 |

L4 逐位置保持 GC 类别，理论修改比例不超过 `1/7`，但容量只有 `3/7 bit/base`。128x128 的 L4 超过 HyenaDNA 160k 上下文。要么把图片降到 64x64，要么使用 450k checkpoint 和更大的观察窗口。

L5 的完整实现、命令和当前边界见 [l5_lm_stego_mvp_cn.md](l5_lm_stego_mvp_cn.md)。当前 `lm` 使用可冻结的 k-mer 概率模型验证可逆协议；Hugging Face causal DNA 模型适配器已经提供，但正式 HyenaDNA 生成实验仍需确认 checkpoint 的 next-token logits 与增量推理能力。

`--ecc-symbols N` 可在 DNA 映射前增加 Reed-Solomon。它主要抵抗替换或字节擦除，不能解决碱基 indel 造成的帧移。indel 的重同步仍应使用 HEDGES、分块同步标识或独立的同步码。

## 3. A800 GPU3 环境

```bash
cd /path/to/codex
conda env create -f environment-hyenadna-a800.yml
conda activate genome-hyenadna

export CUDA_VISIBLE_DEVICES=3
python - <<'PY'
import torch, transformers, cryptography
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('cuda:', torch.cuda.is_available())
print('visible GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('logical cuda:0:', torch.cuda.get_device_name(0))
PY
```

`CUDA_VISIBLE_DEVICES=3` 之后，程序中的 `cuda:0` 就是物理 GPU3。已有环境也可运行 `python -m pip install -r requirements-hyenadna-stego.txt`。

## 4. 下载并缓存 HyenaDNA

官方模型页：<https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf>

```bash
export HF_HOME=/public/home/lifex/jiangjr/.cache/huggingface
hf download LongSafari/hyenadna-medium-160k-seqlen-hf \
  --local-dir /public/home/lifex/jiangjr/models/hyenadna-medium-160k-hf

HYENA_MODEL=/public/home/lifex/jiangjr/models/hyenadna-medium-160k-hf
```

该 checkpoint 使用 Hugging Face 自定义模型代码，程序需要 `trust_remote_code=True`。正式实验应固定下载快照或 `--hyena-revision`，并保留模型目录校验值。

## 5. 生成授权解码密钥

L1-L4 必须使用密钥，数据集只保存 SHA-256 指纹，不保存密钥内容：

```bash
mkdir -p data/keys
umask 077
openssl rand -out data/keys/image_stego.key 32
chmod 600 data/keys/image_stego.key
```

同一个数据集的构建和恢复必须传入同一个 key 文件。

## 6. 构建闭集 L0-L3 数据集

供体清单格式：

```csv
donor_id,assembly_id,fasta,split
HG001,HG001_h1,/data/hprc/HG001.h1.fa,train
HG002,HG002_h1,/data/hprc/HG002.h1.fa,train
HG003,HG003_h1,/data/hprc/HG003.h1.fa,val
HG004,HG004_h1,/data/hprc/HG004.h1.fa,test
```

同一 donor 的 hap1/hap2 必须在同一个 split。单个 CHM13 只适合构建 smoke 数据，不能同时作为 train/val/test。

```bash
python -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/t2t_donors.csv \
  --image-root /data/imagenet \
  --out artifacts/hyena_stego/l0_l3_closed \
  --n-train 6000 --n-val 1500 --n-test 2500 \
  --positive-fraction 0.5 \
  --image-size 128 --window-length 131072 \
  --codecs direct encrypted constrained kmer \
  --codec-key-file data/keys/image_stego.key \
  --kmer-order 3 --seed 42

python -m genome_mining audit-image-dataset \
  --dataset artifacts/hyena_stego/l0_l3_closed \
  --out artifacts/hyena_stego/l0_l3_closed/audit.json

python -m genome_mining summarize-stego-dataset \
  --dataset artifacts/hyena_stego/l0_l3_closed \
  --out artifacts/hyena_stego/l0_l3_closed/stego_summary
```

## 7. 构建未知 codec 测试

下面的模型训练时只看到 L0/L1，验证集看到 L2，测试正样本全部为从未参加训练的 L3：

```bash
python -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/t2t_donors.csv \
  --image-root /data/imagenet \
  --out artifacts/hyena_stego/open_l3 \
  --n-train 6000 --n-val 1500 --n-test 2500 \
  --image-size 128 --window-length 131072 \
  --train-codecs direct encrypted \
  --val-codecs constrained \
  --test-codecs kmer \
  --codec-key-file data/keys/image_stego.key \
  --seed 43
```

## 8. 训练冻结的 HyenaDNA

```bash
export CUDA_VISIBLE_DEVICES=3

python -m genome_mining train-image-locator \
  --dataset artifacts/hyena_stego/open_l3 \
  --out artifacts/hyena_stego/open_l3_hyena_rc \
  --backbone hyenadna \
  --hyena-model-name "$HYENA_MODEL" \
  --hyena-local-files-only \
  --hyena-fine-tune frozen \
  --epochs 20 --batch-size 1 --gradient-accumulation 8 \
  --lr 2e-4 --d-model 128 --downsample-stride 256 \
  --workers 4 --device cuda --amp \
  --codec-key-file data/keys/image_stego.key \
  --seed 42
```

冻结骨干稳定后，可尝试：

```bash
--hyena-fine-tune last_n --hyena-last-n-layers 2 --gradient-checkpointing --lr 2e-5
```

全量微调不应作为第一个实验。正反互补双路约为单向两倍计算量，OOM 时先保持 `batch-size=1`，再增加 gradient accumulation；必要时先加 `--hyena-no-reverse-complement` 做单向基线。

## 9. 四模型消融

```bash
python -m genome_mining run-hyenadna-comparison \
  --dataset artifacts/hyena_stego/open_l3 \
  --out-root artifacts/hyena_stego/open_l3_comparison \
  --epochs 20 --batch-size 1 --gradient-accumulation 8 \
  --hyena-model-name "$HYENA_MODEL" --hyena-local-files-only \
  --device cuda \
  --codec-key-file data/keys/image_stego.key
```

它依次运行 CNN、冻结随机 HyenaDNA、冻结预训练单向 HyenaDNA、冻结预训练 RC HyenaDNA。正式运行前可增加 `--dry-run` 检查实验矩阵。

## 10. 加噪声和 RS 对照

```bash
python -m genome_mining build-image-dataset \
  --genome-manifest data/manifests/t2t_donors.csv \
  --image-root /data/imagenet \
  --out artifacts/hyena_stego/l1_rs_substitution \
  --n-train 6000 --n-val 1500 --n-test 2500 \
  --image-size 128 --window-length 131072 \
  --codecs encrypted \
  --codec-key-file data/keys/image_stego.key \
  --ecc-symbols 16 \
  --substitution-rate 0.0001 \
  --seed 44
```

indel 压力测试可增加 `--indel-rate`。当前信道将随机删除与等量随机插入配对，净长度为零，但中间序列仍发生帧移。没有同步码时恢复失败是预期结果，该实验用于量化问题，不用于宣称已经解决 indel。

## 11. 单独评估和输出

```bash
python -m genome_mining evaluate-image-locator \
  --dataset artifacts/hyena_stego/open_l3 \
  --model artifacts/hyena_stego/open_l3_hyena_rc/model.pt \
  --out artifacts/hyena_stego/open_l3_hyena_rc/test_recheck \
  --split test --batch-size 1 --device cuda \
  --codec-key-file data/keys/image_stego.key \
  --save-images --max-saved-images 100
```

主要输出：

```text
model.pt
train_log.json
run_summary.json
test_evaluation/metrics.json
test_evaluation/predictions.csv
test_evaluation/embeddings.npy
test_evaluation/image_recovery.csv
test_evaluation/recovered_images/
```

`metrics.json` 同时报告总体和 `by_codec` 指标。重点查看 `AUROC/AUPRC/FPR95`、`detector_advantage`、起止边界 MAE、span IoU、exact span、decode success 和 exact image rate。隐写自然性则查看 `stego_report.csv` 的 k-mer JSD、GC/CpG 差异、cover change fraction 与有效容量。

## 12. 解释边界

- L0 的高准确率只证明数据和定位管线能工作。
- 加密隐藏内容，不自动隐藏信息存在。
- 固定 header、同步标记和大量 RS 冗余可能提高恢复率，同时提高可检测性。
- 只针对 HyenaDNA 优化 encoder 会产生模型特定对抗样本，不等于自然 DNA。
- 正式结论必须保持 donor、图片、区间、codec 家族和密钥层面的隔离，并使用 CNN、k-mer/V12、HyenaDNA 及其他 DNA foundation model 交叉验证。
- 当前实现是计算仿真与算法评估工具，不代表任何序列适合写入活体基因组。

## 13. 主要参考资料

- HyenaDNA: <https://arxiv.org/abs/2306.15794>
- HyenaDNA 官方代码: <https://github.com/HazyResearch/hyena-dna>
- Caduceus 反向互补等变模型: <https://openreview.net/forum?id=1wc7ZzLUbs>
- Yin-Yang DNA 编码: <https://www.nature.com/articles/s43588-022-00231-2>
- DNA Fountain: <https://pubmed.ncbi.nlm.nih.gov/28254941/>
- HEDGES indel/替换纠错: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7414044/>
