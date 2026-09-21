# 连续 DNA 的 Rotation Range Coding 实验

本模块是 ACL 2026 RRC 的 DNA 适配实验，不使用 STC、4-mer Codebook、Beam Search、参考编辑或 minimap2。输出是单条连续 ACGT 序列。目标是短随机消息隐写与分布评价，暂不研究图像或扰动恢复。

本轮已增加可续跑数据集、多载荷实验入口、独立验证、CNN 和冻结 HyenaDNA 探针。建议先按 [完整测试手册](rrc_full_test_cn.md) 执行；本文保留协议和单命令说明。

## 来源与实现边界

- 论文：https://aclanthology.org/2026.acl-long.39/
- 作者代码：https://github.com/ryehr/RRC_steganography
- 本次核对版本：`dae326259e4fca8bc4fcf460dafdbc0e88a0a71a`。
- 独立实现论文 Algorithms 3/4 的区间旋转、符号选择与反向旋转，参考作者的反向校验终止保护。没有复制/执行作者源文件。
- 差异：使用 Python Fraction 精确有理数、ACGT 固定顺序整数 CDF、128-bit HMAC 伪随机偏移、每条 fresh nonce。整数频率下限为 1；uniform_mix=0，但量化与频率下限仍改变模型分布。
- HF logits 只保留 A/C/G/T 后重新归一化，N 和特殊 token 不参与输出；普通采样对照也使用完全相同的条件化与量化分布。
- 这不是作者代码的逐位兼容版本，也不继承其理论安全性保证。协议有限随机性、量化、停止时间、认证侧信息和填充都需要独立分析。
- 支持 1..2048 bit，最多 16384 bp，适合协议试验。长消息的有理数运算、逐碱基模型推理可能很慢。
- 使用中点终止条件，并要求反向旋转恢复原消息才结束。接收端仍须知道有效载荷结束位置、消息长度、context、模型和参数。
- `receiver.json` 是授权接收者侧信息：含 nonce、有效结束位置、CDF 摘要和 HMAC 认证。它不存明文消息，但也不是自描述 DNA 协议。不能把它交给检测器。
- 不提供噪声纠错；任何变更由外部认证拒绝。认证通过并不等于 DNA 中嵌入了纠错码。
- 短消息版本未另加 AEAD。密钥用于旋转与认证，不能据此把本协议当成经过审计的加密系统。

## 上传到服务器

上传更新后的整个 `genome_mining/`、`tests/test_rrc*.py`、`requirements-rrc.txt` 和本文档。无需上传 `artifacts/rrc_test_env`。所有命令在项目根目录执行。

```bash
conda activate YOUR_EXISTING_ENV
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m pip -V
python -m genome_mining.rrc_cli --help
python -m unittest discover -s tests -p 'test_rrc*.py' -v
```

核心 k-mer/RRC 流程无新增第三方依赖。`evaluate` 需要 requirements-rrc.txt 中的四项；在现有环境已有时不必安装。离线服务器缺包时，在匹配 Linux/Python 的联网环境准备 wheelhouse 后上传，执行 `python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements-rrc.txt`；Windows 原生 wheel 不能直接用于 Linux。

## 准备真实数据

建立 CSV `data/rrc/donors.csv`，列为 `donor_id,split,fasta`。每个 donor 只用一条 haplotype、只出现在一个 split。示意如下，路径必须替换为服务器实际文件：

```csv
donor_id,split,fasta
DONOR_TRAIN,train,/absolute/path/to/train_hap1.fa.gz
DONOR_VAL,val,/absolute/path/to/val_hap1.fa.gz
DONOR_TEST,test,/absolute/path/to/test_hap1.fa.gz
```

已有 8/2/2 donor 计划就填入那 12 条真实记录，不重新随机挑选。路径也可相对于 manifest 目录。程序接受解压 FASTA 或 gzip，不下载数据。单独一份 CHM13 不能充当 donor-disjoint 实验。

```bash
python -m genome_mining.rrc_cli prepare-contexts \
  --manifest data/rrc/donors.csv --out data/rrc/contexts.csv \
  --per-donor 10 --context-bases 1024 --length 512

python -m genome_mining.rrc_cli fit-context-model \
  --contexts data/rrc/contexts.csv --order 4 --out data/rrc/kmer4.json

python -m genome_mining.rrc_cli keygen --out data/rrc/secret.key
```

prepare-contexts 默认完整流式读取 FASTA，用 seed 固定的 reservoir sampling 从干净、不重叠网格窗口中抽样，不再只选染色体起始窗口。快速冒烟可显式指定 `--sampling first`，但存在起始位置偏差。模型只使用 train 行。审计拒绝 donor 跨 split、同 donor/chrom 区间重叠和完全一致/反向互补序列跨 split；不检测近同源泄漏，不代表严格无泄漏。

正式实验可提供自行按染色体/同源区域分组采样的 contexts.csv，字段为 `sample_id,donor_id,split,chrom,start,context,natural`。start 是 context 起点、0-based；natural 紧随 context。保持全部片段及派生版本在同一 split，额外审计近同源、重叠、重复家族和预训练数据重合。

## 独立编码与解码

```bash
python -m genome_mining.rrc_cli encode \
  --model data/rrc/kmer4.json --key-file data/rrc/secret.key \
  --context ACGTACGT --bits 0010110101011010 --length 512 \
  --out artifacts/rrc_one

python -m genome_mining.rrc_cli decode \
  --model data/rrc/kmer4.json --key-file data/rrc/secret.key \
  --dna artifacts/rrc_one/carrier.dna.txt \
  --metadata artifacts/rrc_one/receiver.json --out artifacts/rrc_decoded
cat artifacts/rrc_decoded/message.bits
```

每次 encode 随机生成 nonce，自动执行恢复检查；所有输出目录必须是新目录。keygen 不覆盖旧密钥。保存密钥，解码仍需要它。API 调用者必须保证同一密钥下不重复 nonce。

## 制作三组数据与评价

```bash
python -m genome_mining.rrc_cli build \
  --contexts data/rrc/contexts.csv --model data/rrc/kmer4.json \
  --key-file data/rrc/secret.key --bits 128 --length 512 \
  --context-bases 1024 --out artifacts/rrc_kmer128

python -m genome_mining.rrc_cli evaluate \
  --csv artifacts/rrc_kmer128/sequences.csv \
  --fpr 0.01 --out artifacts/rrc_kmer128_eval
```

三组分别为 natural、ordinary、rrc，全部等长。RRC 完成消息后的剩余位置用同一整数 CDF 普通采样填充。所有三组都只含片段本身，没有把长 flanks 混进去。

- sequences.csv：检测数据。训练特征只由 sequence 计算，kind 是监督标签；sample_id/pair_id/donor_id/split 仅分组，不能作为特征。
- receiver_private/：解码侧信息，与 detector 输入隔离。
- metrics.json：每条 exact、编码 bp 数、固定长度净 bit/base、累计条件熵、耗时。
- failures.json：所有失败。不能重抽消息直到成功；失败时 build 保存成功部分但非零退出，正式评价必须检查失败比例。
- run.json：模型指纹、参数、输入 SHA256 和尝试数。v2 数据集使用随机 run_id，并按密钥/run_id/sample_id/用途派生消息与 nonce；`build --resume` 保持原消息，新的实验目录不承诺字节相同。receiver_private 中的认证检查点也保存失败，不能续跑时重抽。
- evaluate/report.json：三项独立任务 natural-vs-ordinary、ordinary-vs-rrc、natural-vs-rrc；可选 `--detector logreg/cnn/hyena-probe`，最后一个为冻结表示+线性分类器，不是完整微调；只用验证负例确定阈值，报告实际验证 FPR、测试 FPR/TPR、AUROC/AP 和 donor bootstrap 区间。

每个 split 的负例很少时，1% FPR 没有稳定统计解释。这里的 10 窗口/donor 只用于冒烟测试。代码已加入 CNN/冻结 HyenaDNA 探针，但正式评价仍需增加 donor/窗口数，并运行真实 HPRC 实验。特别是只有两个测试 donor 时，bootstrap 数字本身也很不稳定，不能作强泛化结论。

固定输出长度隐藏了长度差异，但消息相关停止和填充转换仍可能留下特征。entropy_utilization 使用编码段统计，不计接收者侧信息，不是严格信道容量；单条超过 1 不能当成突破信息论。ordinary-vs-rrc 接近随机不等于已证明安全。

## 接入本地 HyenaDNA causal checkpoint

先在 Windows 从官方模型入口选择并完整下载 checkpoint、tokenizer、配置和必要的自定义代码，再上传： https://github.com/HazyResearch/hyena-dna 。不要只上传权重。分类 checkpoint 或只有 hidden states 的 backbone 不适用。

已核对可用接口的轻量候选：https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/tree/main 。其 config 声明 HyenaDNAForCausalLM 和 AutoModelForCausalLM，最大长度 1026。手动下载 `config.json`、`configuration_hyena.py`、`modeling_hyena.py`、`tokenization_hyena.py`、`tokenizer_config.json`、`special_tokens_map.json`、`model.safetensors` 到同一目录。这是接口核对，A800 仍需实测。

```bash
python -m genome_mining.rrc_cli encode \
  --hf-model /absolute/path/to/local_causal_dna_checkpoint \
  --trust-remote-code --device cuda --context-bases 1024 \
  --key-file data/rrc/secret.key --bits 00101101 --length 64 \
  --out artifacts/rrc_hf_smoke
```

只有确认下载的自定义模型代码可信时使用 trust-remote-code；该选项不启用联网。加载器通过 AutoTokenizer 检查 ACGTN 各为独立 token，通过 AutoModelForCausalLM 检查逐位置 logits。部分官方原版 HyenaDNA checkpoint 需其官方 loader，未必支持此 HF API；本适配器会报错而不伪造概率。具体 checkpoint 必须先通过上述冒烟测试。

通过后，在 build 中用 `--hf-model ... --trust-remote-code --device cuda` 替换 `--model ...`。固定同一 checkpoint、context 上限和软件环境。接收端同样传 HF 模型及 context-bases。模型目录会流式计算内容指纹，首次加载可能较慢。

编码与解码都使用 batch=1、相同的全前缀窗口推理。尚未做增量缓存，长 DNA 很慢；不将常规 Transformer KV cache 假定适用于 Hyena。量化不能保证跨设备一致，CDF SHA256 检查会拒绝不一致；必须在 A800 实测。接口能运行不等于 foundation model 已改善隐蔽性。

## 当前验收边界

本地测试覆盖精确区间、短消息穷举、密钥/篡改拒绝、CDF 不一致、长度预算、抽样、断点续跑、多载荷套件、LR/CNN 和固定阈值迁移评价。真实 HyenaDNA tiny 权重在 CPU 已通过生成、独立解码和冻结探针训练/重载检查。完整 17 项 RRC 测试通过，详情见 rrc_validation_20260908.md。合成测试数据只验证软件；真实 HPRC 检测效果、A800 推理和跨设备一致性仍需服务器实验，不提供实测隐蔽率或生物学自然性结论。
