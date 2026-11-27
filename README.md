# KiloKB

面向技术文档的版本感知检索。输入一个 ORM 迁移问题，KiloKB 分别检索 Pydantic v1.10.15 和 v2.7.4 的文档，每条结果都能打开固定 commit 中的原文。

## 快速开始

在仓库根目录使用 Python 3.10 或更新版本运行。示例语料随仓库提供，检索使用 Python 标准库，可离线运行。

```sh
python -m kilokb orm-demo --chunk-words 48
python -m kilokb orm-demo --chunk-words 112
python -m kilokb orm-demo --version v1.10.15 --query "ORM 对象怎么用 from_orm？"
```

默认问题用中文描述迁移需求，并附上四个英文 API 标识符。BM25 按版本独立排序，默认每版返回前 5 条。输出 JSON 中的 `url`、`start_line` 和 `end_line` 对应原文，`score` 可用于查看同版内的排名。

## 版本感知检索接口

同一个检索服务支持 CLI 和 HTTP。单版本请求可指定完整 tag 或 `v1`、`v2` 别名，也可在 query 中写版本。单版本请求未提供版本时默认使用 `v2.7.4`。比较模式解析两个不同版本，`top_k` 对每个版本单独生效。

```sh
python -m kilokb search --query "V1 orm_mode from_orm 与 V2 from_attributes model_validate" --mode compare --trace
python -m pip install -r requirements-api.txt
python -m uvicorn kilokb.api:app --host 127.0.0.1 --port 8000
```

在本机的 `http://127.0.0.1:8000/docs` 打开交互文档，向 `POST /v1/search` 提交以下请求。

```json
{
  "kb_id": "pydantic",
  "query": "V1 orm_mode from_orm 与 V2 from_attributes model_validate",
  "mode": "compare",
  "versions": ["v1.10.15", "v2.7.4"],
  "top_k": 5,
  "profile": "bm25"
}
```

响应按版本返回证据，并附上固定 commit 的原文链接。未知版本、参数与文本版本冲突、比较模式缺少版本等情况会返回 `422` 和错误代码。当全部、部分或没有版本返回证据时，状态分别为 `complete`、`partial` 或 `insufficient`。下文给出了可选阈值的使用方式和实测结果。

用 `GET /v1/traces/{trace_id}` 查看原始请求参数，以及各阶段的候选 ID、名次、耗时和返回数量。每个服务进程保留最近 128 条轨迹。默认 `bm25` 可离线运行。使用 `dense` 或 `hybrid_rrf` 前，按下文准备 BGE-M3 依赖和模型缓存。首次请求对选中的版本计算文档向量，后续请求在同一进程内复用。 [双版本 Hybrid 轨迹样例](reports/pydantic-search-trace.json)

## 拒答评测与阈值

[题集](evals/pydantic-confidence.json)覆盖有答案、库内无答案和版本冲突。50 道题按题族拆成开发集 30 道和留出集 20 道。无答案标注以仓库锁定的两份文档为范围。比较题按版本侧计分，版本冲突单独统计。

```sh
python -m kilokb confidence-eval --device cpu --local-files-only
python -m kilokb search --query "V2 怎样使用 from_attributes？" --profile hybrid_rrf --abstain --trace
```

阈值分数取 RRF Top 5 中能匹配 Dense 候选的最高余弦值，保留六位小数。开发集约束为有答案侧误拒率至多 10%，在满足约束的阈值中最小化无答案侧误返回，同分取更高阈值。选出的 `0.544103` 固定应用于留出集。HTTP 请求可指定 `"abstain": true` 显式启用，适用 `hybrid_rrf`、`top_k=5` 和已校准的语料快照。

| 留出集指标 | 候选直接返回 | 启用阈值 |
|---|---:|---:|
| 无答案侧误返回 | 10/10（100%） | 7/10（70%） |
| 有答案侧误拒 | 0/17（0%） | 2/17（11.8%） |
| 已返回的有答案侧缺少至少一个标注锚点 | 2/17 | 2/15 |
| 版本冲突拦截 | 3/3 | 3/3 |

开发集误拒为 2/25（8%），留出集为 2/17（11.8%）。阈值因此保留为显式选项。指标衡量检索证据的返回或拦截，报告同时保存每个版本侧的锚点覆盖情况。 [阈值清单](evals/pydantic-confidence-policy.json) · [实测与逐题结果](reports/pydantic-confidence-cuda.json)

两次留出集误拒都出现在同一道字段排序比较题（`confidence-h09`）。两个版本的 Top 5 都找齐了原文锚点，但分数低于阈值。同样，私有属性的加密问题（`confidence-h17`）在库内无答案，两版都因主题相近而得到高分。这两类案例说明单一相似度阈值难以同时处理专有术语和证据完整性。

## 换切块后还能找到原文吗？

同一个问题在两种按词切块设置下均覆盖以下四个原文行锚点。表中列出包含对应证据的最佳排名结果及其原文行区间。

| 版本与证据 | 48 词块 | 112 词块 |
|---|---|---|
| [V1 行 128–129 · `orm_mode` / `from_orm`](https://github.com/pydantic/pydantic/blob/5476a758c8ac59887dbfa3aa1c3481d0a0e20837/docs/usage/models.md#L128-L129) | 第 1 名 · 127–134 | 第 1 名 · 122–134 |
| [V2 行 241 · `from_attributes`](https://github.com/pydantic/pydantic/blob/1ab655a54c0defdc5a2a9b13bc90feb6fba78719/docs/concepts/models.md#L241) | 第 2 名 · 239–250 | 第 1 名 · 233–261 |
| [V2 行 285 · `model_validate`](https://github.com/pydantic/pydantic/blob/1ab655a54c0defdc5a2a9b13bc90feb6fba78719/docs/concepts/models.md#L285) | 第 4 名 · 263–286 | 第 2 名 · 250–291 |

回归测试用版本、commit、路径、原文行号和关键标识符核对证据。结果说明这道题的证据覆盖在两种切块设置下都成立。当前演示的匹配信号来自英文术语和 API 标识符。

```sh
python -m pip install pytest
python -m pytest -q
```

Pydantic 原文快照的出处和 SHA-256 见 [语料锁定清单](corpora/pydantic.lock.json)，上游许可见 [Pydantic MIT License](corpora/pydantic/LICENSE)。

## 纯中文检索与 BM25 对照

在 Python 3.11 或更新版本安装可选依赖。首次运行会从 Hugging Face 缓存约 2.3 GB 的固定版本权重。模型下载后，加 `--local-files-only` 可离线运行。

```sh
python -m pip install -r requirements-dense.txt
python -m kilokb orm-demo --retriever dense --version v2.7.4 --query "如何从对象关系映射实例的属性构建校验模型并读取对应的字段？"
python -m kilokb orm-compare --chunk-words 112 --top-k 5 --device cpu
```

对照使用三条[固定的纯中文问题](evals/pydantic-orm-zh.json)，共四个问题与版本组合。两路都在指定版本内检索相同的 Markdown 切块，并用相同的原文行号检查 Top 5 证据。Dense 使用 BGE-M3 的 CLS 向量，归一化后计算余弦相似度。BM25 保留上面的英文标识符演示，当前分析器只提取拉丁字母和数字词项。 [BGE-M3 官方模型](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)

| 检索器 | Top 5 证据齐全的组合 | Top 5 命中的证据锚点 |
|---|---:|---:|
| BM25 | 0/4 | 0/8 |
| BGE-M3 Dense | 3/4 | 7/8 |

针对 v1.10.15 和 v2.7.4 分别提问时，Dense 都找齐了配置和调用方法的证据。同一条通用问题查询两个版本时，v2.7.4 找到配置说明（第 1 名），调用示例（原文第 285 行）未进入 Top 5。这个小样本验证了由中文问题定位英文原文的能力。 [逐题结果、引用与运行环境](reports/pydantic-orm-zh-cpu.json)

下载权重后，在 PowerShell 中运行包含真实模型验收的完整测试。

```powershell
$env:KILOKB_RUN_MODEL_TESTS = '1'
python -m pytest -q
```

## 混合检索与 RRF

`hybrid` 按版本分别取 BM25 和 BGE-M3 Dense 的前 40 个候选。RRF 按名次累加 `1 / (60 + rank)`，其中名次从 1 开始。结果提供合并分数、两个召回分支的名次和原文引用。单路没有匹配时，融合保留另一条分支的排序。

```sh
python -m kilokb orm-demo --retriever hybrid --version v2.7.4 --query "from_attributes 和 model_validate 如何配合？"
python -m kilokb orm-compare --hybrid --chunk-words 112 --top-k 5 --device cpu
```

`--hybrid` 在原三道纯中文题上加入三道[含英文 API 名称的配对题](evals/pydantic-orm-api.json)。三种检索共用版本范围、切块参数和原文锚点。报告分别统计两组题，并列出每条命中的原文、排名和来源分支。 [完整 CPU 对照报告](reports/pydantic-orm-hybrid-cpu.json)

| 题组 | Top 5 指标 | BM25 | Dense | Hybrid + RRF |
|---|---|---:|---:|---:|
| 纯中文 | 证据齐全的版本组合 | 0/4 | 3/4 | 3/4 |
| 纯中文 | 命中的证据锚点 | 0/8 | 7/8 | 7/8 |
| 含 API 名称 | 证据齐全的版本组合 | 4/4 | 3/4 | 4/4 |
| 含 API 名称 | 命中的证据锚点 | 8/8 | 7/8 | 8/8 |

原纯中文跨版本题里，BM25 没有候选，Hybrid 与 Dense 的 Top 5 顺序相同。加入 API 名称后，V2 的调用示例在 BM25 排第 2、Dense 排第 11，融合后排第 3。这个结果展示了包含标识符时词法召回对融合的贡献。 [V2 调用示例原文第 285 行](https://github.com/pydantic/pydantic/blob/1ab655a54c0defdc5a2a9b13bc90feb6fba78719/docs/concepts/models.md#L285)

## 冻结 RRF 候选后的重排

原来的六道题共八个「题目 × 版本」组合。每个组合冻结 RRF 前 30，合计 240 条，记录输入名次、原文区间和文本 SHA-256。三组读取同一份候选文件，回放时校验题集、语料及候选哈希。候选文本统一使用标题路径和原文。 [冻结候选](reports/pydantic-orm-rrf-candidates.jsonl) · [校验清单](reports/pydantic-orm-rrf-candidates.manifest.json)

```sh
python -m pip install -r requirements-dense.txt
python -m kilokb orm-rerank-freeze --output reports/pydantic-orm-rrf-candidates.jsonl
python -m kilokb orm-rerank --candidates reports/pydantic-orm-rrf-candidates.jsonl --top-k 5 --warmup 10 --rounds 5 --device cpu
```

仓库已提供冻结文件，也可以直接执行回放。首次回放会额外缓存约 2.3 GB 的 Cross-Encoder 权重和 BGE-M3 的投影权重。所需文件入缓存后可加 `--local-files-only`。用兼容 CUDA 的 PyTorch 可把回放设备改为 `cuda`。

R0 直接取 RRF 前五。R1 让 BGE Cross-Encoder 联合读取每一对问题和候选。R2 用 BGE-M3 的多向量投影预计算文档向量，在线编码问题，对每个查询 token 找到最相似的文档 token，再取平均。当前小语料用内存精确打分。 [Cross-Encoder 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) · [BGE-M3 多向量公式](https://bge-model.com/bge/bge_m3.html#multi-vector)

| 方案 | 纯中文 Top 5 锚点 | 含 API 名称 Top 5 锚点 | 证据齐全的组合 |
|---|---:|---:|---:|
| RRF 保持原序 | 7/8 | 8/8 | 7/8 |
| BGE Cross-Encoder | 4/8 | 8/8 | 5/8 |
| BGE-M3 MaxSim | 8/8 | 8/8 | 8/8 |

原纯中文跨版本题的 V2 调用示例在冻结候选中排第 8。MaxSim 将它提升至第 2。Cross-Encoder 的前五未覆盖这段示例。这六道相关题用于复查具体失败和取舍。 [V2 原文](https://github.com/pydantic/pydantic/blob/1ab655a54c0defdc5a2a9b13bc90feb6fba78719/docs/concepts/models.md#L285) · [完整实测报告](reports/pydantic-orm-rerank-cuda.json)

| 方案 | 回放 P50 | 回放 P95 |
|---|---:|---:|
| RRF 保持原序 | <0.01 ms | <0.01 ms |
| BGE Cross-Encoder | 400.29 ms | 528.20 ms |
| BGE-M3 MaxSim | 11.09 ms | 14.49 ms |

实测环境为 RTX 4070 Ti、FP32、单并发和 PyTorch 2.6.0+cu124。每组预热 10 次，测量 5 轮共 40 次。耗时从冻结候选开始，覆盖在线查询处理、打分和排序。模型加载与文档预计算独立记录。MaxSim 为 85 个唯一候选片段预计算了约 75 MiB 的原始向量。报告同时给出了实际截断数和各方法的额外显存峰值。
