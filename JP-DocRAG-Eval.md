下面给你一份**以“低算力、强复现、可评测”为核心**的实践指南版本（从零到一），目标是做出一个能被日本 AI/LLM 实习认可的开源项目：**日文科研/工业文档 RAG 的“解析 + 检索 + 评测闭环”**。  
重点：**先把检索评测闭环做扎实（CPU 就能跑），生成式回答/LLM judge 都是可选加分项**。

---

# JP-DocRAG-Eval 低算力实践指南（从头到尾）

## 0. 你要交付的“最小可行成果”

做完这几个，你的项目就已经很专业了：

1. 输入一批日文 PDF（文本型，不做 OCR）
    
2. 输出结构化 chunks（带 doc_id / page / chunk_id，可追溯）
    
3. 两种检索基线：**BM25 + Dense(FAISS)**
    
4. 一个小型“金标准”QA 集（80–200 问，带证据页码/chunk）
    
5. 自动评测脚本：**Recall@k / MRR / nDCG@k**
    
6. ablation：chunk_size / overlap / cleaning 开关（至少跑 6–12 组）
    
7. 生成报告：results.csv + results.md + failure_cases.md

---

# 1) 技术栈选择（低算力优先）

## 1.1 Python & 依赖（建议）

- Python 3.10+
    
- 文档解析：`pdfplumber`（比 pypdf 更稳一些）
    
- 检索：
    
    - BM25：`rank-bm25`
        
    - Dense：`sentence-transformers` + `faiss-cpu`
        
- 日文规范化（可选但推荐）：`jaconv`
    
- 数据存储：JSONL（第一版最方便调试）
    
- CLI：`typer`（可选）
    
- 测试：`pytest`
    
- 代码质量：`ruff`
    

安装（示例）：

```bash
pip install pdfplumber rank-bm25 sentence-transformers faiss-cpu numpy pandas tqdm pyyaml jaconv
pip install typer pytest ruff
```

> **为什么不用 LangChain/LlamaIndex？**  
> 第一版目标是“解析+检索+评测闭环”。引入编排框架会增加复杂度，但不会明显提升闭环质量。等你闭环稳了，再加也不迟。

---

# 2) 仓库结构（建议照抄，面试官一眼看懂）

```text
jp-docrag-eval/
  README.md
  LICENSE
  configs/
    default.yaml
    sweep/
      chunk_300.yaml
      chunk_600.yaml
      clean_on.yaml
      clean_off.yaml
  data/
    docs/                # 你本地放PDF（开源时建议提供下载脚本/清单，不强行分发PDF）
    qa/
      gold_qa.jsonl      # 你标注的评测集
  artifacts/
    corpus.jsonl         # chunk后的语料（可复现产物）
    bm25.pkl
    faiss.index
    embeddings.npy
    index_meta.json
  reports/
    results.csv
    results.md
    failure_cases.md
  src/
    jprag/
      ingest.py
      clean_jp.py
      chunk.py
      index_bm25.py
      index_dense.py
      retrieve.py
      eval_metrics.py
      eval_run.py
      sweep.py
      analyze_failures.py
      utils.py
  tests/
    test_metrics.py
    test_chunk.py
```

---

# 3) 文档解析（Ingestion）：PDF → page texts（可追溯）

## 3.1 只支持“文本型PDF”（第一版边界）

- 不做 OCR
    
- 不追求表格/公式完美还原（这属于第二阶段增强）
    
- 但必须保留 `doc_id + page`，保证引用可追溯
    

## 3.2 解析输出格式（pages.jsonl）

每行一页：

```json
{"doc_id":"doc001","source":"xxx.pdf","page":1,"text":"..."}
```

实现要点：

- 用 pdfplumber 提取每页 text
    
- 对空页/异常页记录日志，不要直接崩
    

---

# 4) 日文清洗（Cleaning）：决定检索上限的关键

做成“可开关的模块”，后面才能做 ablation。

## 4.1 推荐的清洗步骤（低成本高收益）

1. **全角半角统一**（数字/字母/符号）
    
2. **换行修复**（PDF 强制换行最影响 chunk）
    
3. **页眉页脚去除**（检索最大噪声源）
    
4. **去噪行过滤**（页码、目录点线、重复横线等）
    

### 页眉页脚去除（实用算法）

- 对每页取前 N 行 + 后 N 行（例如 N=2）
    
- 统计跨页出现频率
    
- 出现频率 > 阈值（例如 60% 页出现）则判定为页眉/页脚并删除
    

这套规则解释性强，也很适合写进报告。

---

# 5) 分块（Chunking）：从 page texts → chunks（带元数据）

第一版推荐：**page 内切分**（引用落页码更稳）。

## 5.1 chunk 数据结构（corpus.jsonl）

每行一个 chunk：

```json
{
  "chunk_id": "doc001:p001:c003",
  "doc_id": "doc001",
  "source": "xxx.pdf",
  "page": 1,
  "text": "....",
  "meta": {"section": null}
}
```

## 5.2 chunk 策略（可复现）

- 粗分句：用简单正则按 `。！？` 切
    
- 拼接到目标长度（用**字符数近似 token**即可）
    
    - `chunk_chars`: 800（约等价几百 token，按你文档密度调整）
        
    - `overlap_chars`: 100（10–20%）
        

> 不要一开始搞“标题感知 Markdown chunk”，那是第二阶段。先把闭环跑稳。

---

# 6) 建索引（Indexing）：BM25 + Dense 两条基线

## 6.1 BM25（强基线，CPU）

日文没有空格分词，第一版有两个低成本方案：

- **方案A（推荐）字符 2-gram/3-gram**：不需要 MeCab/Sudachi，效果通常不差
    
- 方案B：简单规则切分（标点/空格），效果可能一般但能跑通
    

建议你实现 A，并在 README 说明“无需分词器”。

BM25 建好后序列化成 `bm25.pkl`，并保存 `chunk_id` 列表用于反查。

## 6.2 Dense（Embedding + FAISS，CPU也可）

选一个小而快的多语 embedding 模型（例：`paraphrase-multilingual-MiniLM-L12-v2` 这种 384 维级别的），CPU 跑也可以接受。

流程：

1. 对每个 chunk 计算 embedding → `embeddings.npy`
    
2. 建 FAISS 索引（cosine 相似度通常做法是先 normalize，再用 inner product）
    
3. 保存 `faiss.index` + `index_meta.json`（记录模型名、维度、normalize 与否）
    

---

# 7) 检索（Retrieval）：支持三种模式（都很实用）

你应该支持：

1. `bm25`：只用 BM25
    
2. `dense`：只用 dense
    
3. `hybrid`：简单融合（推荐 **RRF** 或加权归一）
    

### RRF（Reciprocal Rank Fusion）推荐原因

- 不需要调复杂权重
    
- 工程上稳、效果经常不错
    
- 很好解释、很适合写报告
    

---

# 8) 金标准评测集（Gold QA）：让项目“可信”的核心证据

## 8.1 gold_qa.jsonl 格式（建议）

每行一个样本：

```json
{
  "qid": "q0001",
  "question": "PMUの配置最適化で用いられる評価指標は？",
  "gold": [
    {"doc_id":"doc001","page":3,"chunk_id":"doc001:p003:c002"}
  ],
  "notes": "答案在第3页xx段"
}
```

## 8.2 标注策略（省事但靠谱）

- 先选 5–10 份文档（总页数别太夸张）
    
- 每份文档写 10–20 个问题（混合：定义/步骤/参数/对比）
    
- 每题至少标 1 个 gold evidence（页码必填，chunk_id 尽量填）
    
- 加 10–20 个“困难问题”：同义词、跨段落、术语缩写，方便失败分析
    

> 80–200 问就够用。宁可小而高质量，也不要大而乱。

---

# 9) 评测（Evaluation）：只做“检索评测”也足够专业

你要输出三类指标（top-k 默认 k=1/3/5/10）：

- **Recall@k**：gold evidence 是否出现在 top-k
    
- **MRR**：第一个命中的倒数排名
    
- **nDCG@k**：如果你有多个 gold 或你以后想加“相关性分级”
    

输出：

- `reports/results.csv`：每个 config 一行（非常关键）
    
- `reports/results.md`：把 csv 转 markdown 表格，贴进 README
    
- `reports/failure_cases.md`：失败样本（问题 + top-3 chunk 摘要 + gold 页码）
    

---

# 10) Ablation（一定要做）：让它像研究而不是 demo

第一版至少跑这 12 组（你也可以少一点，但建议 >= 6）：

维度建议：

- chunk_chars：600 / 900 / 1200
    
- overlap_chars：0 / 100
    
- cleaning：on / off
    
- retriever：bm25 / dense / hybrid
    

把这些写成 YAML config，用 `sweep.py` 批量跑，最后汇总成一张表。

---

# 11) “一键复现”命令设计（README 最重要部分）

建议你提供这四条命令（哪怕内部是 python 模块）：

```bash
# 1) 解析+清洗+chunk -> artifacts/corpus.jsonl
python -m jprag.ingest --config configs/default.yaml

# 2) 建 BM25
python -m jprag.index_bm25 --config configs/default.yaml

# 3) 建 Dense + FAISS
python -m jprag.index_dense --config configs/default.yaml

# 4) 跑评测 -> reports/results.*
python -m jprag.eval_run --config configs/default.yaml
```

再提供一个：

```bash
# 5) 跑 ablation sweep
python -m jprag.sweep --dir configs/sweep/
```

---

# 12) 可选加分模块（不影响低算力主线）

这些都是“第二阶段增强”，你有空再加：

## 12.1 更强解析：Marker / layout-aware parsing（可插拔）

做成 `--parser simple|marker`。  
优势：双栏/表格/公式更好。  
风险：依赖更重、速度更慢、效果波动。  
**你可以用 ablation 给出“simple vs marker 对检索指标的提升”**，这比口头说强太多。

## 12.2 生成式回答 + 引用（可选）

只要做到：

- 输出回答时强制附引用 `[doc_id:page:chunk_id]`
    
- 自动检查引用 chunk 是否来自 top-k
    

不用追求“回答多漂亮”，重点是“引用可追溯”。

## 12.3 RAG 指标框架（Ragas 等）

放最后做，因为它会引入 LLM judge、成本、波动。  
你可以只在 20–50 问的小集合上跑，固定模型/温度/版本，作为加分附件。

---

# 13) 质量保障（少量但关键）

- `test_metrics.py`：用一个小例子验证 Recall/MRR 计算正确
    
- `test_chunk.py`：保证 chunk_id 唯一、chunk 不为空、page 不丢
    
- GitHub Actions：跑 ruff + pytest（不跑大评测）
    

---

# 14) 你做完后，简历怎么写才“硬”

你可以写成这种完全真实、且可验证的 bullet（把数字填进去）：

- Built a reproducible Japanese document retrieval evaluation pipeline (PDF ingestion → cleaning/chunking → BM25 & dense FAISS retrieval), with a gold QA set (**N=__**) and automated metrics (Recall@k, MRR, nDCG).
    
- Conducted ablation studies on chunk size/overlap and cleaning heuristics (header/footer removal, line-break repair), producing benchmark reports and failure-case analysis.
    

---

## 你照这个指南做，会不会需要高算力？

不会。核心闭环（解析+BM25+dense+评测）**CPU 即可**。  
如果你愿意进一步降低成本：

- Dense embedding 可先只做 1 次缓存（embeddings.npy）
    
- 文档数量控制在 5–10 份、页数别太夸张
    
- QA 集 80–150 问就能出很漂亮的 ablation 表
    

---

如果你希望我把它再“落到可直接开工”的级别：我可以下一步直接给你

1. `configs/default.yaml` 和 sweep 配置模板
    
2. `gold_qa.jsonl` 的标注模板 + 一个半自动标注流程（帮你省时间）
    
3. `results.md` / `failure_cases.md` 的固定输出格式（你跑出来就能贴 README）




## 推荐的数据源（电力系统领域、日文、适合做 RAG）

### A. 电力系统“规程/技术要件/并网”类（最适合做 RAG）

这类文档特点：**术语密集、条款明确、PDF 很多、工程师经常要查**。对检索评测也特别友好（问题 → 证据页码通常很明确）。

- **METI/資源エネルギー庁：系統連系技术要件指南（ガイドライン）**  
    这是“并网技术要件”核心文档之一，非常适合做问答与证据定位。 
    
- **OCCTO（电力广域机构）：グリッドコード/系統連系規程 相关资料**  
    OCCTO 会持续更新 grid code 讨论与反映资料，适合做“版本变更/条款查找”类问题。 
    
- **一般送配电事業者的系統連系技術要件（各区域电网公司）**  
    比如 TEPCO PG、关西送配电等都有公开 PDF，内容更贴近实际工程接入条件。 
    

**为什么这类最推荐**：

- 问题可以做得非常“工程化”：某个参数、某个条件、某个要求写在哪页
    
- gold evidence 标注很快（页码清楚）
    
- 检索的提升空间大（BM25/向量/混合检索差异明显）
    

---

### B. 市场/规则/交易规程类（也很适合）

- **JEPX（日本卸电力取引所）取引規程 / 業務規程**  
    很多岗位会涉及“制度/规则文本搜索”，这类 RAG 很常见。 
    

**适合的问题类型**：市场有哪些、交易单位、约束、流程、定义条款等。

---

### C. 政策与研究报告类（用来扩展“背景解释能力”）

- **エネルギー白書（资源能源厅）**：结构清晰、章节分明，适合做“背景解释+引用”型回答。 
    
- **NEDO 项目公开资料/报告**：经常包含“技术路线/问题定义/图表描述”，适合做“研究总结型”问答。 
    

---

## 选哪个做第一版语料库（低算力、最高成功率）

建议你第一版就控制在 **20–50 个 PDF**，总页数 **< 1500 页**，非常够用。

**最推荐的组合（从易到难）**：

1. 并网/技术要件：METI 指南 + 2 家送配电公司的“系統連系技術要件” 
    
2. OCCTO grid code 相关资料（挑 5–10 份） 
    
3. JEPX 规程（1–2 份就行） 
    
4. 白书/NEDO（作为扩展语料） 
    

> 开源注意：**不要直接把大量 PDF commit 进仓库**。更稳的做法是：
> 
> - repo 里放 `docs_list.csv`（标题+来源链接）
>     
> - 提供 `download_docs.py` 去官方下载  
>     这样更像工业界做法，也更少版权风险。
>     

---

## 最后“呈现给用户”应该长什么样？它是搜索引擎吗？

一句话：**它是“企业级技术文档搜索（检索）+ 可选的引用式问答（RAG）”**。  
你完全可以把它设计成两个模式：

### 1) Search 模式（必做）：像搜索引擎，但更适合 PDF 规程

用户输入问题/关键词后，系统返回：

- Top-k 命中 chunks 列表
    
- 每条包含：`doc_id / 文档名 / 页码 / chunk 摘要`
    
- 支持“点开原文页码”（或至少显示来源信息）
    

这一步已经非常有意义：

- 比传统全文搜索更稳（尤其你做了清洗、去页眉页脚、chunk、混合检索）
    
- 对工程师来说，“快速定位条款在哪页”就是生产力
    

### 2) Answer 模式（可选加分）：在 Search 结果上生成“带引用的总结”

在 top-k chunks 上，让模型给一个简短回答，并强制引用：

- 每段末尾标注 `[文档名:页码:chunk_id]`
    
- 同时把证据 chunks 展示出来（可展开）
    

它和“普通搜索引擎”的核心区别在于：

- **不仅返回链接/片段**，还会把多个证据融合成可读回答
    
- **必须可追溯**（引用到页码/段落），适合规程/工业文档这种“不能胡编”的场景
    

---

## 这个项目的“意义”到底是什么（从实习视角）

对你来说，它不是“做一个聊天机器人”，而是证明你具备工业界最想要的三件事：

1. **非结构化文档 ETL 能力**：PDF 清洗、chunk、元数据追溯
    
2. **检索系统能力**：BM25 / Dense / Hybrid + 指标评测（Recall@k, MRR）
    
3. **研究式严谨**：ablation + failure cases（这点很像博士训练）
    

而且电力领域文档天然“专业、复杂、术语多”，你用它做语料库会显得更可信、更有门槛。