# RRC 本地验证记录（2026-09-08）

实现协议：dna_rrc_fraction_v1。作者代码核对 SHA：dae326259e4fca8bc4fcf460dafdbc0e88a0a71a。

本地隔离环境：artifacts/rrc_test_env；默认 Python 环境未安装新依赖。该 Windows venv 不能上传作为 Linux 环境。

## 首版已执行

- `python -m unittest discover -s tests -p 'test_rrc*.py' -v`：10 项测试通过。1..6 bit 全部 126 条消息穷举；不同长度、全零全一、概率改变、认证、预算与采样测试。
- 在装有 numpy/pandas/scikit-learn/joblib 的隔离环境执行同一测试：包括实际 LogisticRegression 三任务评价，10 项通过。
- `python -m unittest discover -s tests -p test_l5_stego.py`：7 项中 5 项通过、2 项缺可选依赖跳过。
- 使用实际 `LongSafari/hyenadna-tiny-1k-seqlen-hf` 权重，CPU、context limit 128，32-bit 消息通过 encode 内部恢复检查。
- 随后启动独立 decode 进程，读取 DNA、receiver.json 和密钥，恢复同一条 32-bit 消息；不是只复用编码进程内的模型状态。
- 该单条消息结果：编码段 17 bp，固定输出 128 bp，编码段 1.88235 bit/bp；固定输出净率 0.25 bit/bp。不能把这一个样本解释为容量或隐蔽率结论。

## 产物

- `artifacts/rrc_hf_verified/carrier.dna.txt`
- `artifacts/rrc_hf_verified/receiver.json`
- `artifacts/rrc_hf_test.key`（仅本地测试密钥，不用于正式实验）
- `artifacts/rrc_hyena_tiny/`（实际下载的轻量 checkpoint）

## 未验证

真实 HPRC 数据生成与检测结果、A800/CUDA 性能、CPU-GPU CDF 一致性、HyenaDNA 全量微调、统计不可检测性、突变与 indel 鲁棒性。已有认证仅拒绝损坏，不恢复损坏。

## 本轮完善后的复验

数据实验格式升级为 `rrc_paired_v2`，单条编码协议仍是 `dna_rrc_fraction_v1`。

在本地隔离环境设置 `RRC_TEST_HF_MODEL=artifacts/rrc_hyena_tiny` 后运行：

```text
python -m unittest discover -s tests -p 'test_rrc*.py' -v
Ran 17 tests in 45.778s
OK
```

新增检查内容：

- reservoir 抽样可复现、不是默认取开头窗口；区间重叠和配对审计。
- HMAC 检查点篡改拒绝、消息/nonce 用途隔离。
- 成功实验续跑的 sequences.csv 字节不变；参数变化拒绝续跑。
- 容量失败被保存，续跑仍保持同一失败，不重新抽消息。
- 实际训练 LR 和两轮 CNN，保存后重载打分与原测试预测一致。
- 固定阈值迁移评价与同数据原评价一致；源 train/val donor 与目标 test 重叠时拒绝。
- 两个载荷率的 suite CLI 跑通及续跑，不重复训练已完成的评价。
- 真实 HyenaDNA tiny 权重：六组小型合成 context 上生成 RRC、独立进程精确解码、重放普通采样；冻结 Hyena probe 三任务训练、保存、重载打分成功。

另独立检查真实权重的冻结特征接口：2 条 32 bp 序列输出 `(2, 256)` 特征。模型指纹 `11caa8f87ab80087beb9195347ff28e31f0c9688629d35f0684a9e2ebefed66d`，PyTorch `2.14.0+cpu`。以上是接口兼容性记录，不是推荐服务器重装为这个版本。

旧 L5 回归重新运行：7 项中 5 项通过，2 项因可选 cryptography 依赖缺失跳过。新增/修改 RRC 模块 compileall 通过。

测试使用合成序列验证工程正确性，没有把它们当成真实 donor 或正式 natural DNA，也未据此公布隐蔽率。测试临时目录自动清理，测试代码可重跑；原先的独立单条 packet 产物仍保留。完整服务器步骤见 `rrc_full_test_cn.md`。
