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
