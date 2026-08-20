# Qwen3.8-27B 结构特点分析

## 1. 模型定位

Qwen3.8-27B 是阿里 Qwen 团队 2026 年 8 月中旬开源的新一代 **27B 稠密（Dense）原生多模态大模型**，Apache 2.0 协议，可免费商用、微调与再分发。它是 2.4T 参数 MoE 旗舰 Qwen3.8-Max（约 95B 激活参数）的"可部署稠密兄弟"，与旗舰共享同一套混合注意力骨干，是当时开源 Qwen 家族中综合能力最强的稠密模型。

## 2. 基本信息

| 项目       | 数值                                                                                    |
| -------- | ------------------------------------------------------------------------------------- |
| 总参数量     | 27B（含视觉编码器约 **27.78B**，BF16）                                                          |
| 架构类      | `Qwen3_5ForConditionalGeneration`（model_type: `qwen3_5`）                              |
| 层数 / 隐藏维 | 64 层 / hidden_size 5,120                                                              |
| FFN 中间维  | 17,408（SwiGLU，silu）                                                                   |
| 词表       | 248,320（padded，不绑定词嵌入）                                                                |
| 全注意力层    | 24 Q 头 / 4 KV 头（GQA），head_dim 256，RoPE 维 64，`partial_rotary_factor=0.25`，输出带门控（swish） |
| 线性注意力层   | Gated DeltaNet：16 QK 头 + 48 V 头，head_dim 128，卷积核 dim 4，恒定大小循环状态                       |
| 上下文长度    | 原生 **262,144** tokens；YaRN 静态 RoPE 缩放可扩展至 **~1,010,000**                              |
| 模态       | 文本 + 图像 + 视频（原生多模态，预训练阶段即融合）                                                          |
| 许可证      | Apache 2.0                                                                            |

## 3. 核心结构特点

### 3.1 混合注意力（Hybrid Attention）

- 64 层中仅 **16 层为全注意力**（Gated Attention），其余 **48 层为 Gated DeltaNet 线性注意力**（恒定大小循环状态，无随序列增长的 KV）。
- 排列规律：`16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`，即每 4 层出现 1 个全注意力层（`full_attention_interval: 4`），3:1 混合。

```
层:  [GDN→FFN][GDN→FFN][GDN→FFN][GA→FFN] × 16 组  = 64 层
      ├─ 线性注意力（循环状态恒定，3/4 层）
      └─ 全注意力（KV 随序列增长，1/4 层）
```

- **收益**：长序列计算中 3/4 的层从 O(n²) 变为 O(n)；只有 16 层保留增长型 KV 缓存，缓存开销约为同配置全注意力 64 层稠密模型的 **1/4**，使 262K 超长上下文在消费级硬件上可部署。
- **代价**：线性注意力层的信息检索能力弱于全注意力，因此每隔 3 层插入一个全注意力层作"锚点"，兼顾全局信息交互与效率。这与课程中 DeepSeek V3（MLA 压缩 KV）、Kimi K3（纯线性注意力）形成一条"注意力轻量化"演进脉络，Qwen 走的是两者折中的混合路线。

### 3.2 原生多模态（Text + Image + Video）

- 预训练阶段即端到端融合视觉与文本，而非后期拼接，config 中带独立 `vision_config`（视觉塔：hidden 1,152、27 层、patch 16、时间维 patch 2、空间 merge 2，输出投影到 5,120）。
- 图像/视频位置编码使用 **mRoPE**（多模态 RoPE，`mrope_interleaved`，section [11,11,10]），与文本位置编码同一机制。
- 可统一理解文本、图像、视频（含长视频），27.78B 总参数含视觉塔。

### 3.3 超长上下文：262K 原生 → 1M 扩展

- 原生 262,144 tokens 即覆盖绝大多数 Agent/长文档场景；需要更长时通过 YaRN（factor 4.0, rope_theta 1e7）静态缩放至约 1M。
- 官方建议只在确实需要超长上下文时启用 YaRN，并提示静态缩放可能轻微影响短文本性能。

### 3.4 内置 MTP（多 Token 预测）推测解码

- 检查点自带 1 层 MTP 草稿头（`mtp_num_hidden_layers: 1`，共享词嵌入），无需单独草稿模型即可做推测解码，官方/主流框架（vLLM `method: mtp`，3 个草稿 token；Ascend 侧为 `qwen3_5_mtp`）直接支持，可显著提升解码吞吐。

### 3.5 思考机制

- 思考模式默认开启，输出 `<think>` 推理过程；`reasoning_effort`（xhigh / high / medium / low）可调推理深度；`preserve_thinking` 可保留历史轮次推理上下文（利于多轮 Agent 场景）；也可 `enable_thinking: false` 关闭得到直接回答。
