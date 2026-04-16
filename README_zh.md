# CS336 Spring 2025 作业 1：大语言模型基础

本项目是 **Stanford CS336（2025年春季）课程的第一个作业**，目标是从零开始使用 PyTorch 实现一个完整的大型语言模型（LLM），深入理解 Transformer 架构的核心组件。

> **注意**：这是一个教学项目，要求学生**完全从零实现**，不使用 PyTorch 内置的 Transformer 模块。

---

## 📚 项目简介

在这个作业中，你将亲手实现现代大语言模型的各个核心组件，包括：

- **分词器（Tokenizer）**：实现 BPE（Byte Pair Encoding）算法，从头训练自己的分词器
- **神经网络基础组件**：Linear、Embedding、RMSNorm、SiLU、SwiGLU 等
- **注意力机制**：缩放点积注意力、多头自注意力、RoPE 旋转位置编码
- **Transformer 架构**：完整的 Transformer Block 和语言模型
- **训练基础设施**：AdamW 优化器、余弦学习率调度、梯度裁剪、检查点保存/加载

通过完成这个作业，你将深入理解 LLaMA、GPT 等现代大语言模型的内部工作原理。

---

## 🗂️ 项目结构

```
assignment1-basics/
├── cs336_basics/               # 主代码目录（需要在此实现各个模块）
│   ├── __init__.py
│   └── pretokenization_example.py  # 预分词示例
│
├── tests/                      # 单元测试
│   ├── adapters.py             # 接口适配器（连接你的实现与测试）
│   ├── test_model.py           # 模型相关测试
│   ├── test_nn_utils.py        # 神经网络工具测试
│   ├── test_optimizer.py       # 优化器测试
│   ├── test_tokenizer.py       # 分词器测试
│   ├── test_train_bpe.py       # BPE 训练测试
│   ├── test_data.py            # 数据加载测试
│   ├── test_serialization.py   # 序列化测试
│   ├── fixtures/               # 测试数据
│   └── _snapshots/             # 测试快照
│
├── cs336_assignment1_basics.pdf  # 作业说明书（PDF）
├── README.md                   # 英文说明
├── README_zh.md                # 本文件（中文说明）
├── pyproject.toml              # 项目配置与依赖
├── uv.lock                     # uv 锁文件
└── make_submission.sh          # 提交脚本
```

---

## 🚀 环境配置

本项目使用 [`uv`](https://github.com/astral-sh/uv) 作为包管理工具，确保环境的可复现性和便携性。

### 1. 安装 uv

```bash
# 推荐：使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip
pip install uv

# macOS 用户也可以使用 Homebrew
brew install uv
```

### 2. 安装依赖

项目依赖会自动管理，运行以下命令即可：

```bash
# 运行任意 Python 文件时会自动解决并激活环境
uv run <python_file_path>

# 例如运行测试
uv run pytest
```

### 主要依赖

- **PyTorch** (~2.11.0) - 深度学习框架
- **einops / einx** - 张量操作工具
- **jaxtyping** - 类型注解，明确张量形状
- **tiktoken** - OpenAI 的分词器（参考实现）
- **pytest** - 单元测试框架
- **wandb** - 实验追踪（可选）

---

## 🧪 如何运行测试

### 运行所有测试

```bash
uv run pytest
```

### 运行特定测试

```bash
# 运行模型相关测试
uv run pytest tests/test_model.py

# 运行分词器测试
uv run pytest tests/test_tokenizer.py

# 运行特定测试函数
uv run pytest tests/test_model.py::test_linear
```

### 初始状态

> ⚠️ **重要**：刚开始时，所有测试都会因为 `NotImplementedError` 而失败。
> 
> 你需要：
> 1. 在 `cs336_basics/` 目录下实现各个模块
> 2. 在 `tests/adapters.py` 中连接你的实现
> 3. 重新运行测试，直到全部通过

---

## 📊 数据准备

下载训练所需的数据集：

```bash
mkdir -p data
cd data

# TinyStories 数据集（简单故事，用于基础训练）
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText 样本（更大的真实语料）
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

---

## 📝 作业模块详解

### 模块 1：神经网络基础

| 组件 | 描述 | 对应函数 |
|------|------|----------|
| **Linear** | 线性变换层 | `run_linear()` |
| **Embedding** | 词嵌入层 | `run_embedding()` |
| **RMSNorm** | 均方根归一化（LLaMA 使用） | `run_rmsnorm()` |
| **SiLU** | Sigmoid Linear Unit 激活函数 | `run_silu()` |
| **SwiGLU** | Swish-Gated Linear Unit 前馈网络 | `run_swiglu()` |

### 模块 2：注意力机制

| 组件 | 描述 | 对应函数 |
|------|------|----------|
| **Scaled Dot-Product Attention** | 缩放点积注意力 | `run_scaled_dot_product_attention()` |
| **Multi-Head Self Attention** | 多头自注意力（不使用 RoPE） | `run_multihead_self_attention()` |
| **RoPE** | 旋转位置编码（Rotary Position Embedding） | `run_rope()` |
| **MHA with RoPE** | 带位置编码的多头注意力 | `run_multihead_self_attention_with_rope()` |

### 模块 3：Transformer 架构

| 组件 | 描述 | 对应函数 |
|------|------|----------|
| **Transformer Block** | 单个 Transformer 块（Pre-norm 结构） | `run_transformer_block()` |
| **Transformer LM** | 完整的 Transformer 语言模型 | `run_transformer_lm()` |

### 模块 4：训练基础设施

| 组件 | 描述 | 对应函数 |
|------|------|----------|
| **AdamW** | AdamW 优化器 | `get_adamw_cls()` |
| **Cosine LR Schedule** | 余弦学习率调度（含 warmup） | `run_get_lr_cosine_schedule()` |
| **Cross Entropy** | 交叉熵损失 | `run_cross_entropy()` |
| **Gradient Clipping** | 梯度裁剪 | `run_gradient_clipping()` |
| **Softmax** | Softmax 函数 | `run_softmax()` |

### 模块 5：数据与序列化

| 组件 | 描述 | 对应函数 |
|------|------|----------|
| **Get Batch** | 从数据集采样批次 | `run_get_batch()` |
| **Save Checkpoint** | 保存模型检查点 | `run_save_checkpoint()` |
| **Load Checkpoint** | 加载模型检查点 | `run_load_checkpoint()` |

### 模块 6：BPE 分词器

| 组件 | 描述 | 对应函数 |
|------|------|----------|
| **Train BPE** | 从头训练 BPE 分词器 | `run_train_bpe()` |
| **Tokenizer** | 使用训练好的分词器 | `get_tokenizer()` |

---

## 🎯 学习要点

完成这个作业后，你将掌握：

1. **Transformer 架构的完整实现细节** - 理解每个组件的作用和数学原理
2. **位置编码** - 深入理解 RoPE 旋转位置编码的工作原理
3. **现代优化技术** - AdamW、学习率调度、梯度裁剪等
4. **分词算法** - BPE 算法的完整实现
5. **张量操作** - 熟练使用 PyTorch 进行高效的张量运算
6. **模型训练流程** - 数据加载、训练循环、检查点管理

---

## 📖 参考资源

- 课程网站：[cs336.stanford.edu](https://cs336.stanford.edu)
- 作业说明书：`cs336_assignment1_basics.pdf`
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA 论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE 论文

---

## ⚠️ 学术诚信提示

根据课程要求：
- ✅ 允许使用 AI 工具进行**概念性问题**的询问和**底层编程**帮助
- ❌ **禁止**使用 AI 直接解决作业问题或生成实现代码
- ❌ **禁止**参考第三方实现（课程材料是独立的）

目标是**通过实践学习**，而不是观看 AI 生成解决方案。

---

## 🤝 问题反馈

如果你发现作业说明书或代码有任何问题，欢迎：
- 在 GitHub 上提交 Issue
- 提交 Pull Request 修复

---

**祝你学习愉快！** 🚀
