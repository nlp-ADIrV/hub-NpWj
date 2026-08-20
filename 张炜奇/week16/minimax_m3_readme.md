# MiniMax-M3 结构特点分析

## 1. 模型定位

MiniMax-M3 是 MiniMax 于 2026 年 6 月发布的**旗舰级原生多模态 MoE 模型**，6 月 7 日开放权重（Hugging Face / GitHub），6 月 11 日公开配套技术报告《MiniMax Sparse Attention》（arXiv:2606.13392）。它是**首个把"稀疏注意力"（MSA）做到生产级规模并开源**的模型，也是 MiniMax 首次开源旗舰模型。注意其许可证为 **MiniMax Community License**——非商用免费，商用需署名并通知官方，比 Apache/MIT 严格。

## 2. 基本信息

| 项目 | 数值 |
|---|---|
| 总参数量 | **~428B**，激活 ~23B（A23B；视觉编码器约 0.6B） |
| 架构 | MoE + MSA 块稀疏注意力，共 60 层 |
| 层布局 | 前 **3 层 Dense（MLP + 全注意力）**，后 **57 层 MoE + MSA** |
| 隐藏维 | 6,144（前代 M2 为 3,072，翻倍） |
| 专家配置 | **128 个路由专家 top-4 + 1 个共享专家** |
| 注意力 | GQA：64 Q 头 / 4 KV 头（每 GQA 组 16 个 query 头） |
| 路由 | sigmoid 打分 + 可学习偏置，aux-loss-free（DeepSeek-V3 风格，非 softmax） |
| 激活 / 归一化 | SwiGLU-OAI（clamped）/ Zero-Centered RMSNorm（fp32 归一化） |
| 上下文长度 | 原生 **~1M** tokens（官方 API 保障 512K） |
| 模态 | 文本 + 图像 + 视频，**预训练 Step 0 起联合训练**（~100 万亿交错 token） |
| 权重精度 | BF16 主版本 + MXFP8 量化版（社区另有 NVFP4/GGUF） |
| 许可证 | MiniMax Community License（非 Apache/MIT） |

## 3. 核心结构特点

### 3.1 MSA 块稀疏注意力

MSA（MiniMax Sparse Attention）是**基于 GQA 的 block 级稀疏注意力**，把一层标准 GQA 注意力拆成两条分支：

```
输入
 ├─ Index Branch（Lightning Indexer，索引分支）
 │    每个 GQA 组 1 个 index query 头，所有组共享 1 个 index key 头
 │    token 级打分 → max-pooling 聚合成 block 级分数（block_size=128）
 │    → 每个组独立选 Top-k 块（exp-free 选择，适配张量核）
 └─ Main Branch（主分支）
      只对选中的块做精确的 block 稀疏 softmax 注意力（KV-outer 稀疏）
```

- **每个 query 只 attend 约 2,048 个 token**（Top-k 块 × 128），注意力成本与序列长度解耦——这是能支撑 1M 上下文的关键。
- 当前 query 的**本地邻域块强制包含**，保证基本上下文不丢失。
- **训练技巧**（Top-k 不可导）：引入 **KL 对齐损失**让索引器模仿主分支的注意力分布，配合 Indexer Warmup（索引预热）和 stop-gradient。
- **收益**（论文数据，109B 实验模型，1M 上下文）：每 token 注意力计算量约为 GQA 的 **1/28.4**，H800 上 prefill 提速 14.2×、decode 提速 7.6×（墙钟）；官方称生产版 M3 相对前代 M2 每 token 端到端计算量约为 **1/20**。

> 注意区分：论文里的 109B 是验证 MSA 的实验模型，**不是** M3 本身；M3 是 428B/A23B 生产模型。

### 3.2 大规模 MoE：128 专家 top-4 + 共享专家

- 128 个路由专家 top-4 激活，外加 1 个**共享专家**（学通用知识，路由专家学专门知识），与 DeepSeek-V3 的共享专家设计一致。
- **sigmoid 打分 + 可学习偏置**路由（aux-loss-free），避免 softmax 路由导致的负载不均问题。
- 前 3 层刻意不用 MoE（Dense + 全注意力），让浅层稳定、无路由开销。
- 细节：clamped SwiGLU（SwiGLU-OAI）抑制激活值爆炸；Zero-Centered RMSNorm 初始化权重为 0、fp32 归一化，提升训练稳定性。

### 3.3 原生多模态

- 与 Qwen3.8-27B 这类"LLM + 视觉塔"融合方案不同，M3 **从预训练第一步就文本/图像/视频联合训练**（约 100 万亿交错 token），视觉理解深度融入主干。
- 视觉编码器：CLIP 风格 ViT，32 层，Conv3d patch embedding + 3D RoPE（带时间维的位置编码，天然支持视频），**PatchMerger** 把视觉 token 投影到 6,144 维与 LLM 隐藏维对齐。
- 支持 computer use（桌面操作），配合 1M 上下文做长程 Agent 任务。

### 3.4 思考模式

- Thinking / Non-Thinking 双模式可切换；官方称其推理版在 Artificial Analysis 智能指数暂列开源模型第一（厂商口径，未经完全独立验证）。
