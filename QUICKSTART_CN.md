# JP-DocRAG-Eval 快速上手指南

本项目是一个针对日文工业/法规文档的 RAG（检索增强生成）评测流水线。
核心特点：**低算力需求**、**全流程可复现**、**自动化评测**。

## 1. 环境准备 (Setup)
确保已安装 Python 3.9+。

```bash
# 1. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
# (如果没有 requirements.txt，至少安装以下库)
pip install pdfplumber sentence-transformers faiss-cpu numpy tqdm pandas

# 3. 设置 PYTHONPATH (重要：防止模块导入错误)
export PYTHONPATH=$PYTHONPATH:.
```

## 2. 数据准备 (Ingestion)
**目的**: 将 PDF 从 `data/docs/` 读取并转换为标准化的文本格式。

```bash
# 读取 PDF -> 提取文本 -> 保存为 artifacts/pages.jsonl
python3 src/jprag/ingest.py
```

## 3. 数据清洗 (Cleaning)
**目的**: 去除页眉、页脚、无效噪点，优化日文格式（合并断行）。

```bash
# 输入: artifacts/pages.jsonl -> 输出: artifacts/pages_clean.jsonl
python3 src/jprag/clean.py
```

## 4. 切片 (Chunking)
**目的**: 将清洗后的页面按字符数切分，保留上下文重叠。
*参数建议: max_chars=400, overlap=50 (适合法规条文)*

```bash
# 输入: artifacts/pages_clean.jsonl -> 输出: artifacts/chunks.jsonl
python3 src/jprag/chunk.py --max_chars 400 --overlap_chars 50
```

## 5. 索引构建 (Indexing)
**目的**: 分别构建倒排索引 (BM25) 和 向量索引 (Dense) 以便检索。

```bash
# 5.1 构建 BM25 索引 (基于字符 N-gram) -> artifacts/bm25.pkl
python3 src/jprag/index_bm25.py

# 5.2 构建 Dense 索引 (FAISS + MiniLM) -> artifacts/dense/
python3 src/jprag/index_dense.py
```

## 6. 统一评测 (Unified Evaluation)
**目的**: 对三种检索模式 (BM25, Dense, Hybrid) 进行跑分，对比 Recall@k。
*注意: 混合检索 (Hybrid) code采用了加权 RRF (BM25权重=2.0)，确保基线稳定。*

```bash
# 运行评测 -> 输出 reports/results.csv 和 failure_cases.md
python3 src/jprag/eval_run.py --gold data/qa/generated_qa.jsonl
```

## 7. 参数扫描 (Ablation Sweep) (可选)
**目的**: 自动尝试不同的 Chunk Size 等参数组合，寻找最优解。

```bash
# 自动修改 chunk 参数 -> 重建索引 -> 评测 -> 汇总结果
python3 src/jprag/sweep.py --ablate_chunking
```
