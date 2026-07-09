# 第十周作业：基于本地文件的问答系统

## 1. 项目说明

本项目实现了一个基于本地文件的简单问答系统。

程序会读取 `data/` 目录中的文本资料，将文档切分为若干文本块，然后使用 BM25 方法进行检索。用户输入问题后，系统先找到最相关的文本块，再从文本块中提取与问题最相关的句子作为回答。

整个程序不需要网页，也不依赖在线接口，可以直接在命令行中运行。

## 2. 目录结构

```text
week10_qa_system/
├── data/                       # 本地资料
│   ├── 01_RAG基础.md
│   ├── 02_向量检索.md
│   └── 03_大模型推理部署.md
├── evaluation/
│   ├── evaluate.py             # 简单检索效果测试
│   └── questions.json          # 测试问题
├── src/
│   ├── __init__.py
│   ├── bm25.py                 # BM25 检索
│   ├── document_loader.py      # 文档读取
│   ├── qa_system.py            # 问答主流程
│   ├── text_splitter.py        # 文本切分
│   └── tokenizer.py            # 分词
├── main.py                     # 程序入口
├── requirements.txt
└── README.md
```

## 3. 运行环境

建议使用 Python 3.9 及以上版本。

创建环境：

```bash
conda create -n week10 python=3.10 -y
conda activate week10
```

安装依赖：

```bash
pip install -r requirements.txt
```

说明：示例资料均为 Markdown 文件，因此即使没有安装 `pypdf` 也可以正常运行。安装 `pypdf` 后，程序还可以读取 PDF 文件。

## 4. 运行方式

### 4.1 交互式问答

在项目根目录运行：

```bash
python main.py
```

示例：

```text
问题：RAG 的基本流程包括哪些步骤？

回答：
RAG 的典型流程可以分为文档处理、检索和答案生成三个阶段。

参考文件：01_RAG基础.md
```

输入 `exit` 或 `quit` 可以结束程序。

### 4.2 单次提问

```bash
python main.py -q "向量数据库的作用是什么？"
```

### 4.3 指定资料目录

```bash
python main.py --data ./data
```

## 5. 替换成自己的文件

将自己的资料放入 `data/` 目录即可。目前支持：

- `.txt`
- `.md`
- `.pdf`

然后重新运行：

```bash
python main.py
```

程序启动时会重新读取并建立索引。

## 6. 简单测试

运行：

```bash
python evaluation/evaluate.py
```

程序会读取 `evaluation/questions.json` 中的问题，检查排名第一的检索结果是否来自正确资料，并输出 Hit@1。

## 7. 方法说明

本项目主要包含以下步骤：

1. 读取本地文件；
2. 按段落和句子切分文本；
3. 对文本建立 BM25 检索索引；
4. 根据用户问题检索最相关文本块；
5. 从最相关文本块中抽取相关句子作为回答。

这种方法实现简单、运行速度快，适合课程作业和小规模资料问答。对于更复杂的问题，可以继续增加向量检索或大模型生成模块。
