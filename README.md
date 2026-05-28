# 华智大学智能问答助手

基于 LangChain + Qwen + FAISS + Streamlit 的校园规章制度问答系统，支持多轮对话。

## 系统架构

```
用户提问 → 检索模块(FAISS) → 提示词装配 → Qwen大模型推理 → 返回回答
                ↑
         知识库构建模块（文档切分 + 向量化）
```

| 模块 | 文件 | 功能 |
|------|------|------|
| 知识库构建 | `knowledge_base.py` | 加载文档、切分文本、向量化存储 |
| 检索模块 | `retriever.py` | 语义相似度检索相关片段 |
| 提示词装配 | `prompt_builder.py` | 组合问题、上下文、历史为提示 |
| 大模型推理 | `llm_engine.py` | 调用 Qwen 生成回答 |
| 聊天界面 | `app.py` | Streamlit 多轮对话前端 |

## 环境准备

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装并启动 Ollama

本项目通过 [Ollama](https://ollama.com) 运行 Qwen 量化模型，这是最简便的本地部署方式。

```bash
# 安装 Ollama（Windows 访问 https://ollama.com/download 下载安装包）

# 拉取 Qwen 轻量化模型（约 1GB）
ollama pull qwen2.5:1.5b

# 验证模型已就绪
ollama list
```

Ollama 启动后会自动在 `http://localhost:11434` 提供 OpenAI 兼容 API。

> **可选：** 如果你的机器有更多资源，可使用更大的模型：
> - `qwen2.5:3b` — 3B 参数，效果更好
> - `qwen2.5:7b` — 7B 参数，需要 8GB+ 显存/内存
>
> 修改 `config.py` 中的 `LLM_MODEL_NAME` 即可切换。

### 3. 首次运行构建知识库

```bash
python knowledge_base.py
```

这将：
1. 加载 `docs/` 目录下的所有 `.txt` 文档
2. 按 500 字切分（重叠 100 字）
3. 使用 `text2vec-base-chinese` 模型向量化
4. 存入 FAISS 并保存至 `vector_store/` 目录

> 首次运行会自动下载 Embedding 模型（约 400MB），之后会使用缓存。

## 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，即可开始对话。

## 添加自定义文档

1. 将 `.txt` 文档放入 `docs/` 目录
2. 重新构建知识库：
   ```bash
   python knowledge_base.py
   ```
3. 重启 Streamlit 应用

## 项目结构

```
├── docs/                    # 校园文档（知识源）
│   ├── 奖学金管理办法.txt
│   ├── 选课指南.txt
│   └── 研究生培养管理.txt
├── vector_store/            # FAISS 向量数据库（自动生成）
├── config.py                # 项目配置
├── knowledge_base.py        # 模块一：知识库构建
├── retriever.py             # 模块二：检索模块
├── prompt_builder.py        # 模块三：提示词装配
├── llm_engine.py            # 模块四：大模型推理
├── app.py                   # 模块五：Streamlit 聊天界面
├── requirements.txt         # Python 依赖
└── README.md                # 本文件
```

## 配置说明

所有可调参数集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 文本切分长度 |
| `CHUNK_OVERLAP` | 100 | 相邻片段重叠长度 |
| `RETRIEVAL_TOP_K` | 4 | 检索返回的片段数量 |
| `EMBEDDING_MODEL_NAME` | `shibing624/text2vec-base-chinese` | 中文向量化模型 |
| `LLM_MODEL_NAME` | `qwen2.5:1.5b` | Qwen 模型版本 |
| `LLM_TEMPERATURE` | 0.3 | 生成温度（越低越确定） |
| `LLM_MAX_TOKENS` | 1024 | 最大生成长度 |

## 技术选型说明

- **LangChain**：提供文档加载、切分、检索链等核心抽象
- **Qwen 2.5 (1.5B)**：通义千问轻量化模型，中文效果优秀，资源消耗低
- **FAISS**：Meta 开源的高效向量相似度搜索库
- **Ollama**：零配置本地模型部署，提供 OpenAI 兼容 API
- **Streamlit**：快速构建交互式 Web 界面
- **text2vec-base-chinese**：中文语义向量模型，适合中文文档检索
