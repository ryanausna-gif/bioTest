# 深度学习模型设计：未知图片字节流格式识别与恢复

## 1. 任务定义

给定一段未知二进制流 `x`，其中可能包含一个完整图片 payload，但没有文件名、没有扩展名，也可能存在随机前缀或后缀噪声。目标是输出：

- `format`: 图片格式类别，例如 `png`、`jpeg`、`bmp`、`webp`、`gif`、`tiff`
- `start_offset`: 图片 payload 的起始位置
- `image`: 通过标准解码器恢复出的图片

本阶段不假设真实 DNA 数据，只处理通用二进制数据流。后续可把 DNA 序列先映射为字节流，或把 tokenizer 从 byte token 改为 A/C/G/T token。

## 2. 推荐模型

模型采用 byte-level 多任务结构：

```text
bytes -> byte embedding -> local CNN blocks -> Transformer encoder -> pooled feature
                                                        |-> format head
                                                        |-> start-offset head
```

### Byte Embedding

把每个字节 `0..255` 看作 token，额外使用 `256` 作为 padding token。这样模型可以直接学习文件头、chunk 标记、压缩流模式等局部结构。

### CNN 层

一维卷积擅长捕捉局部字节模式，例如：

- PNG: `89 50 4E 47 0D 0A 1A 0A`
- JPEG: `FF D8 FF`
- WebP: `RIFF....WEBP`
- BMP: `42 4D`

这些模式非常局部，CNN 比纯 Transformer 更省数据。

### Transformer 层

Transformer 用于整合更长距离的结构线索，例如 WebP 的 `RIFF` 与 `WEBP` 间隔、PNG chunk 顺序、JPEG marker 序列等。

### 多任务输出

训练目标包含两部分：

```text
loss = CE(format_pred, format_label)
     + lambda_start * CE(start_pred, start_offset_label)
```

如果后续加入损坏数据，可扩展第三个 head：

```text
corruption head -> 预测哪些字节可能被破坏
```

## 3. 数据合成策略

第一阶段可以完全合成训练数据：

1. 随机生成小图片，包含渐变、色块、线条、文字、噪声。
2. 使用 Pillow 分别编码成 PNG、JPEG、BMP、WebP、GIF、TIFF。
3. 在真实 payload 前后加入随机字节，模拟未知数据流。
4. 可选加入轻度字节替换噪声，但训练初期不要破坏 payload，否则标准解码器无法恢复。

样本形式：

```python
{
    "stream": bytes,
    "format": "png",
    "start_offset": 173,
    "payload": bytes,
}
```

项目也支持真实图片目录作为数据源。真实图片不会直接作为带扩展名的文件输入模型，而是先由 Pillow 读取为像素内容，再随机重新编码为 PNG/JPEG/BMP/WebP/GIF/TIFF 等目标格式，最后加入随机前缀/后缀字节。这样可以同时得到真实图像内容和均衡的格式标签：

```bash
python -m image_format_recovery.train \
  --data-dir /path/to/real_images \
  --epochs 20 \
  --steps-per-epoch 1000 \
  --batch-size 64 \
  --checkpoint artifacts/model.pt \
  --device cuda
```

如果没有 `--data-dir`，训练集就是在线合成数据；如果提供 `--data-dir`，训练集来自真实图片目录。

## 4. 恢复流程

实际恢复不是让模型直接输出像素，而是：

```text
未知字节流
  -> 模型预测 format + start_offset
  -> 根据 format 做结构化 payload 裁剪
  -> Pillow 解码
  -> 保存为 PNG
```

项目中的 `formats.py` 同时提供基于魔数的强基线。当魔数存在时，基线往往比模型更准；模型真正有价值的场景包括：

- 文件头不在第 0 字节，而是隐藏在大段未知数据中
- 文件头附近有轻度噪声，需要从整体结构判断格式
- 后续 DNA 映射引入同步偏移、片段顺序变化或弱格式线索

## 5. 评估指标

建议分三类指标：

- 格式准确率：`format_accuracy`
- 起点定位准确率：`start_offset_accuracy` 或 `abs_offset_error`
- 图片恢复率：能否被 Pillow 成功打开，并得到正确尺寸/像素

如果 JPEG 参与评估，像素不能要求完全一致，因为 JPEG 是有损压缩；可以比较尺寸和感知相似度。PNG/BMP/TIFF 可做严格像素一致性。

## 6. 后续接入 DNA 数据

当转向 DNA 场景时，可以沿用这套管线：

```text
DNA 序列 -> ACGT 解码/纠错 -> 字节流 -> 格式识别/边界定位 -> 图片恢复
```

或训练端到端 tokenizer：

```text
A/C/G/T tokens -> sequence model -> format/start/corruption heads
```

更鲁棒的 DNA 版本建议加入：

- Reed-Solomon / LDPC 等纠错码
- 同步 marker，处理插入和删除错误
- payload 分片编号和 hash 校验
- 多拷贝共识恢复
