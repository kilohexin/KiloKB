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
