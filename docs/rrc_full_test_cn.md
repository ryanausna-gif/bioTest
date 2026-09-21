# RRC 连续 DNA 隐写：相对完整的离线服务器测试

## 1. 本轮到底测试什么

范围是短随机 bit 消息的连续 DNA 隐写，不做图像恢复、STC、散点编辑、minimap2 或生存仿真。

```text
现有 HPRC haplotype FASTA + 固定 donor split
-> 不重叠窗口抽样及审计
-> 同一 context 下三组等长片段
   natural：真实相邻序列
   ordinary：相同概率模型/CDF 的普通采样
   rrc：RRC 消息编码 + 普通采样填充
-> 每条消息精确恢复 + 独立进程重放
-> LR / CNN / 冻结 HyenaDNA 探针
-> 验证集定阈值，测试集评价
-> 固定检测器和阈值的跨载荷率/跨概率模型迁移测试
```

这能回答：协议能否正确恢复、模型生成与自然序列是否可区分、编码是否增加额外可检测性、检测器是否只适用于一种生成风格。不能据此宣称隐写不可检测或具备自然生物功能。

这里检测器看到的是完整候选片段，不是插入后的一长段基因组。因此本轮不是 presence + 精确边界定位实验，也没有验证接头是否暴露插入位置。

## 2. 上传文件与环境检查

从 Windows 上传当前整个 `genome_mining/`，以及：

```text
tests/test_rrc.py
tests/test_rrc_pipeline.py
tests/test_rrc_experiment.py
requirements-rrc.txt
docs/rrc_suite_kmer.example.json
docs/rrc_suite_hyena.example.json
docs/rrc_full_test_cn.md
docs/rrc_dna_experiment_cn.md
```

不要上传本机 `artifacts/rrc_test_env`，它是 Windows 环境，不能用于 Linux。数据继续使用你已经手动下载、校验过的 HPRC，不需要 AWS CLI，也不重新下载图集。

以下所有命令在服务器项目根目录执行。把环境名换成你已能运行 HyenaDNA 的环境：

```bash
conda activate YOUR_EXISTING_ENV
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1

python -m pip -V
python -m genome_mining.rrc_cli --help
python -m genome_mining.rrc_suite --help
python - <<'PY'
import sys, torch, transformers, numpy, pandas, sklearn, joblib
print('python:', sys.executable)
print('torch:', torch.__version__, 'transformers:', transformers.__version__)
print('cuda:', torch.cuda.is_available(), 'visible:', torch.cuda.device_count())
assert torch.cuda.is_available()
print('logical cuda:0:', torch.cuda.get_device_name(0))
PY
```

不指定固定 GPU。命令使用 `--device cuda` 时，当前程序默认使用进程可见的第一张 GPU，不会自动选择最空闲的卡，也不会自动使用全部 GPU。如调度器已设置 CUDA_VISIBLE_DEVICES，保留其设置，不要覆盖或清除。只有你自己此前手动设置了该变量、且不受调度器管理时，才可用 `unset CUDA_VISIBLE_DEVICES` 撤销限制。

这些 export 设置是当前 shell 的环境变量，供随后启动的 Python 子进程读取，不是 Python 导入语句。关闭终端后通常失效，不会自动写入 conda 环境配置。CUBLAS_WORKSPACE_CONFIG 为确定性 CUDA 运算配置工作区；两个 OFFLINE 变量禁止 Hugging Face 自动联网取文件；PYTHONNOUSERSITE 禁止加载用户目录的 site-packages，减少不同 Python 环境的依赖混用。它们不会安装依赖，也不会下载模型。

已有依赖就不重装。缺包时需在匹配服务器 Linux/Python/CUDA 的联网机器准备 wheelhouse，再上传离线安装；Windows wheel 不能直接用于 Linux。始终用当前环境的 `python -m pip`，避免此前 pip 路径错位。

```bash
# 仅在缺少这些基础检测依赖且已上传 Linux wheelhouse 时执行
python -m pip install --no-index --find-links /path/to/linux_wheelhouse -r requirements-rrc.txt
```

## 3. 使用现有 HPRC manifest

之前生成的 `genome_manifest.csv` 含 `donor_id,assembly_id,fasta,split`，可以直接读取，不必重新手写 12 条记录。额外的 assembly_id 列会被忽略。

```bash
export HPRC=/public/home/lifex/jiangjr/biolearning/database/data/raw/hprc2_paper
export MANIFEST="$HPRC/prepared_seed2026/genome_manifest.csv"
mkdir -p data/rrc artifacts

python - <<'PY'
import csv, os
from pathlib import Path
from collections import Counter
p = Path(os.environ['MANIFEST'])
with p.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
print('split:', Counter(r['split'] for r in rows))
for r in rows:
    fa = Path(r['fasta'])
    if not fa.is_absolute():
        fa = p.parent / fa
    print(r['donor_id'], r['split'], fa.exists(), fa)
    assert fa.is_file(), f'Missing FASTA: {fa}'
assert len({r['donor_id'] for r in rows}) == len(rows)
PY
```

如果实际目录不同，修改 MANIFEST 即可。若 manifest 写的是 `.fa` 而你只有 `.fa.gz`，先按原离线解包流程解压，或在 manifest 中改为实际 `.fa.gz` 路径。代码两者均支持。不要把同一个 CHM13 文件复制成 12 个 donor 名称。

8/2/2 是当前可用的 donor 划分，不是“严格无泄漏”的保证。不同人的同源区域很相似，本工具未进行近同源分组，也不知道预训练模型见过哪些基因组。

## 4. 制作 pilot 数据集

```bash
python -m genome_mining.rrc_cli prepare-contexts \
  --manifest "$MANIFEST" \
  --out data/rrc/contexts_pilot.csv \
  --per-donor 10 --context-bases 1024 --length 512 \
  --sampling reservoir --seed 2026

python -m genome_mining.rrc_cli fit-context-model \
  --contexts data/rrc/contexts_pilot.csv \
  --order 4 --out data/rrc/kmer4.json

python -m genome_mining.rrc_cli keygen --out data/rrc/secret.key
```

默认 reservoir 会完整扫描每个 donor FASTA 一次，随机抽取干净的、不重叠网格窗口。不是在所有可能起点上均匀抽样，也不是按染色体等额抽样；长染色体按可用网格窗口数自然占更大权重。扫描过程中会输出进度，压缩大文件可能耗时较久。

极快的接口检查可改 `--sampling first`，并另用 `contexts_smoke.csv` 文件名，但它只取最早的干净窗口，有位置偏差，不应作为正式结果。

每条记录包含 1024 bp context 和紧随其后的 512 bp natural。12 donor × 10 窗口 = 120 对照单位，每个载荷率生成 360 条检测序列。8/2/2 划分对应每类 train/val/test 各 80/20/20 条，足够排查管线，但不够做稳定的 1% FPR 结论。

输出 `contexts_pilot.csv.audit.json` 记录采样 seed、每个源文件位置/大小/修改时间、有效窗口数和 CSV SHA256。这里未重新计算整份 FASTA 的校验和；继续保留你原先的官方 MD5 校验记录。

审计检查 donor 跨 split、同 donor/chrom 区间重叠、完全相同/反向互补序列跨 split。遇到审计失败应调查数据，不能反复换 seed 挑一份看起来表现好的测试集。

## 5. 先跑 k-mer 协议基线

项目已提供可直接执行的配置，路径相对于配置文件目录解析：

```bash
python -m genome_mining.rrc_suite --config docs/rrc_suite_kmer.example.json
```

它依次运行 32、64、128 bit，每条输出固定为 512 bp，净率分别为 0.0625、0.125、0.25 bit/base。每个载荷率执行 build、独立 verify、LR 和 CNN 的三项二分类。k-mer 仅从 train 行拟合，不使用 val/test 拟合模型。

配置里的 device=cpu 是便于排查协议问题；可在启动前改成 cuda，使 CNN 用已选定的 GPU。k-mer 概率计算本身仍在 CPU。实验开始后不要更改配置再续跑。

终端按阶段报告，详细子进程进度写在：

```bash
tail -f artifacts/rrc_suite_kmer/logs/bits_32_build.log
```

中断后继续：

```bash
python -m genome_mining.rrc_suite --config docs/rrc_suite_kmer.example.json --resume
```

同一个输出目录只允许一个进程运行。完整实验再次 --resume 会复用生成检查点和已完成评价，但仍执行独立重放校验。

## 6. HyenaDNA 概率源

把已完整下载的可信 HF causal checkpoint 上传到项目的：

```text
models/hyenadna-tiny-1k-seqlen-hf/
```

轻量接口参考：[LongSafari/hyenadna-tiny-1k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/tree/main)。本机已经用这份真实权重验证接口。需要配置、tokenizer、自定义 Python 文件及权重，不能只有 safetensors。服务器全程本地加载，无须外网。

先做单条 GPU 协议检查：

```bash
python -m genome_mining.rrc_cli encode \
  --hf-model models/hyenadna-tiny-1k-seqlen-hf \
  --trust-remote-code --device cuda --context-bases 1024 \
  --key-file data/rrc/secret.key --bits 0010110101011010 --length 64 \
  --out artifacts/rrc_gpu_one

python -m genome_mining.rrc_cli decode \
  --hf-model models/hyenadna-tiny-1k-seqlen-hf \
  --trust-remote-code --device cuda --context-bases 1024 \
  --key-file data/rrc/secret.key \
  --dna artifacts/rrc_gpu_one/carrier.dna.txt \
  --metadata artifacts/rrc_gpu_one/receiver.json \
  --out artifacts/rrc_gpu_decoded

cat artifacts/rrc_gpu_decoded/message.bits
```

应恢复 `0010110101011010`。这个上下文仅为接口测试；数据集生成会用真实 FASTA context。`trust-remote-code` 只是同意执行下载目录中的自定义代码，不代表联网；仅对你审查/信任的模型使用。

通过后运行：

```bash
python -m genome_mining.rrc_suite --config docs/rrc_suite_hyena.example.json
# 如中断，用同一配置和同一模型/密钥续跑
python -m genome_mining.rrc_suite --config docs/rrc_suite_hyena.example.json --resume
```

默认三个检测器：

| 检测器 | 实际实现 | 能回答什么 |
|---|---|---|
| logreg | 4-mer + 统计特征 + 标准化 + LogisticRegression | 浅层统计差异是否明显 |
| cnn | 从头训练的多层 Conv1D，训练期反向互补增强 | 不依赖生成器权重的局部序列模式是否可检测 |
| hyena-probe | 冻结 HyenaDNA backbone，mean/max pooling + 线性分类器 | 预训练表示中是否有可分信号 |

Hyena probe 不是 HyenaDNA SFT 或全量微调。示例中它和生成器使用同一 checkpoint，属于共享预训练表示的探针，不应称为完全独立的 LM 检测器；CNN 是独立训练的对照。后续可以替换 detector_hf_model，但仍须兼容当前单碱基 causal HF 适配器。

编码、解码和普通采样都是 batch=1 全前缀推理，无增量缓存。A800 不等于可以立即生成几十万条长 DNA。先看 pilot 的 metrics.json 中 total_seconds，再估算规模；不要一开始就扩成整幅图片消息。

## 7. 断点、失败与私有数据

每个 build 输出：

```text
dataset/
  sequences.csv                  # 仅此作为检测数据；特征只读取 sequence
  metrics.json                   # 消息恢复、编码段长度、padding、时间与净率
  failures.json                  # 容量不足，不重抽消息
  run.json                       # 输入/模型/配置指纹、运行状态、失败率
  verification.json              # 独立重放结果
  receiver_private/
    plan.json                    # 认证后的运行计划
    pair_*.checkpoint.json       # 认证检查点，成功和失败均保存
    pair_*.json                  # 授权解码侧信息
```

receiver_private 及密钥都不能作为检测器输入。metadata 中的结束位置、nonce、context、CDF 摘要不是可公开喂给分类器的特征。

消息和不同用途的 nonce 从密钥、随机 run_id、sample_id 和用途标签派生。相同计划续跑沿用原值，不会重新抽取更好编码的消息。新目录会有新 run_id。跨机器复现还需要原计划、检查点、模型及一致的软件环境；只记 seed 不够。

协议长度不足会记录 budget_failure。多载荷入口会跳过该载荷率的检测评价，继续其他载荷率，最后非零退出并给出 capacity_failures.json。不要删除失败样本后声称隐写效果很好。若确实要研究“成功条件下”的分布，可单独 evaluate --allow-failures，但必须报告选择偏差，示例主线不启用它。

配置/密钥/模型/输入变化会拒绝续跑。旧版没有 plan.json 的数据集不能原地续跑，使用新的输出目录；旧单条 RRC packet 的协议接口未被替换。

## 8. 如何读结果

先看 `suite_status.json`、`capacity_failures.json`、各 dataset/verification.json；再看 suite 根目录 summary.csv。详细报告位于：

```text
bits_32/logreg_attempt_1/report.json
bits_32/cnn_attempt_1/report.json
bits_32/hyena-probe_attempt_1/report.json
```

| 任务 | 负例 / 正例 | 解释 |
|---|---|---|
| natural_vs_ordinary | 自然 / 普通生成 | 生成模型和真实自然分布之间的差异 |
| ordinary_vs_rrc | 普通生成 / 隐写生成 | RRC 对同一生成分布引入的额外信号 |
| natural_vs_rrc | 自然 / 隐写生成 | 候选片段相对天然 DNA 的整体可检测性 |

三个模型分别训练，不能互换解读。ordinary_vs_rrc 接近随机，但 natural_vs_rrc 很高，意味着编码可能未增加明显差异，而概率源本身仍不像天然 DNA。任何一个弱检测器接近随机，都不是安全性证明。

阈值只在验证负例上确定，预测使用 score > threshold。报告 AUROC、AP、实际验证 FPR、测试 FPR/TPR。CNN 仅按验证损失选择 checkpoint，未用 test 选择 epoch。报告还包含按 donor 重抽样的 95% 区间，保留三类配对结构；它不包含训练随机性、阈值估计的不确定性，只有两个测试 donor 时尤其不稳定。

diagnostics.json 是各 split 的 GC、最长同聚物、单碱基熵及聚合 4-mer JSD。它们是诊断统计，不是天然性证书。roc.csv 可作 ROC 图，predictions.csv 保留测试分数，training.json 保存 CNN 训练曲线数值；当前没有自动绘图。

## 9. 固定检测器的迁移测试

先训练完源实验，不再调参数。以下用“32 bit ordinary_vs_rrc 的 CNN”，测试 128 bit：

```bash
python -m genome_mining.rrc_cli evaluate-transfer \
  --detector-dir artifacts/rrc_suite_hyena/bits_32/cnn_attempt_1/ordinary_vs_rrc \
  --csv artifacts/rrc_suite_hyena/bits_128/dataset/sequences.csv \
  --device cuda --bootstrap 500 \
  --out artifacts/rrc_transfer_32_to_128
```

再测试概率源切换，以下使用 k-mer 实验训好的检测器，测试 HyenaDNA 生成数据：

```bash
python -m genome_mining.rrc_cli evaluate-transfer \
  --detector-dir artifacts/rrc_suite_kmer/bits_128/cnn_attempt_1/ordinary_vs_rrc \
  --csv artifacts/rrc_suite_hyena/bits_128/dataset/sequences.csv \
  --device cuda --bootstrap 500 \
  --out artifacts/rrc_transfer_kmer_to_hyena
```

两个实验必须用同一固定长度、同一 donor 划分。命令只读取目标 test，沿用源阈值，不在目标 val 或 test 重新训练/校准。它拒绝目标 test donor 和源 train/val donor 重叠，也检查 exact/RC 序列重复，但未检查近同源。自测试可把目标换为原实验，结果应与其原 report 一致。

要评价 natural_vs_rrc，换 detector-dir 最后一级即可。使用 hyena-probe 时加 --hf-model 和 --trust-remote-code，并保持原冻结 encoder 指纹。任务不同不能混用阈值。

这是跨载荷率/跨概率源迁移，还不是已经完成 L0-L5 开放 codec benchmark。现有 L0-L5 数据要先统一到同样的样本字段、长度、donor 划分和任务定义，不能直接拼接不同预处理的数据制造“泛化”。

## 10. 扩大实验及本轮验收

pilot 通过后可生成 `contexts_main.csv`，提高 per-donor（例如 100），并复制配置到新文件，修改 contexts、out 和对应 kmer model。新的数据用新的 kmer train-only 拟合；不要把 pilot 分数选出来的参数继续当成独立最终测试结果。

每 donor 100 窗口仍只有 200 个验证负例，1% FPR 只有约两个计数的量级。报告程序会提示低 FPR 估计不稳定。正式低 FPR 测量需要更多负例和更多独立 donor；重复 seed 不能替代新增 donor。

本轮最小验收：

1. 选定载荷率无未解释的编码失败，全部验证消息 exact=True。
2. 断点恢复不改变已完成样本，普通采样可独立重放。
3. 三组等长、分组审计通过，接收侧信息未进入特征。
4. 三项二分类完成，阈值仅来自 val，报告 FPR 和样本数。
5. 至少一个跨载荷率和一个跨概率源测试完成，保留源阈值。
6. 记录环境与模型指纹、数据校验记录，不把 pilot 的高/低准确率作为最终论文结论。

```bash
python -m pip freeze > artifacts/rrc_environment_freeze.txt
python -m unittest discover -s tests -p 'test_rrc*.py' -v
# 可选真实 HF 权重集成测试，默认 CPU，合成序列只测试软件接口
RRC_TEST_HF_MODEL="$PWD/models/hyenadna-tiny-1k-seqlen-hf" \
  python -m unittest discover -s tests -p 'test_rrc*.py' -v
```

仍未完成：Hyena 增量推理、噪声纠错/indel 同步、隐藏区间定位、无 sidecar 盲解码、正式 HPRC 检测结论及密码学安全性证明。固定长度及普通采样填充也可能留下停止位置相关信号，需要实测，不能直接继承原 RRC 论文的安全性结论。
