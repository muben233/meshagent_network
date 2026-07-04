# MeshAgent MALT 部分复现实验报告

## 1. 复现目标

本文复现论文 *MeshAgent: Enabling Reliable Network Management with Large Language Models* 中与 MALT（Network Lifecycle Management）应用相关的核心实验流程。MALT 任务要求 LLM 根据自然语言网络管理查询生成 Python 代码，在网络图上完成节点查询、属性更新、容量统计、拓扑修改等操作，并将执行结果与预定义 ground truth 对比。

本次复现实验聚焦于 MALT 场景下的运行时机制，包括约束注入、query-specific constraint retrieval、CoT 分步代码生成、执行错误修复、约束检查、confidence-based abstention 等。由于本地环境未复现原文中的多模型设置、fine-tuning 训练和人工审核的动态约束演化流程，因此本文将实验定位为 MALT 运行时流程的近似复现，而不是对论文 Table 4 的完全等价复现。

## 2. 原文实验概述

原文在三个网络管理应用上评估 MeshAgent：Traffic Analysis、MALT 和 Cloud Resource Graph。主实验使用带 ground truth 的 benchmark queries，并在 GPT-4o、Gemini 和 DeepSeek-V3 等模型上比较 CoT、Few-shot/RAG、Fine-tuned、RL/ReAct、LATS 以及叠加 MeshAgent 后的效果。

原文的 MeshAgent 不是单独的提示词，而是一套完整流程：

1. 从少量样例和网络图结构中构建 constraints。
2. 对每个 query 检索相关 constraints。
3. 将 constraints 加入 prompt，引导 LLM 生成代码。
4. 在 sandbox 中执行生成代码。
5. 使用 validation tests 检查输出是否满足网络结构和应用约束。
6. 如果出现执行错误或约束违反，将错误上下文反馈给 LLM 进行 error reduction。
7. 根据输出一致性和 debug 次数计算 confidence。
8. 对低置信度结果 abstain，只返回通过可靠性筛选的答案。

因此原文主要报告 reliable accuracy，即系统实际回答的问题中的正确率。MALT 中，原文 Table 4 报告 GPT-4o 上 CoT+Few-shot 从 0.842 提升到 MeshAgent 后的 0.986，CoT+Fine-tuned 从 0.910 提升到 MeshAgent 后的 0.986。

## 3. 本地实现与修改

本次复现使用仓库中的 `app-malt` 作为实验入口。为了保证结果有效，首先修正了原 benchmark 中的图状态污染问题：每个 query 的模型执行图和 ground truth 执行图都重新加载，避免前一题对图的修改影响后一题。

本地新增和使用的主要脚本如下：

| 文件 | 作用 |
|---|---|
| `benchmark.py` | 基础执行框架，包含 Baseline、+Constraints 和简化 Full MeshAgent |
| `run_reproduction_benchmark.py` | 运行三组 raw accuracy benchmark |
| `run_malt_paper_reproduction.py` | 更接近原文流程的 MALT Full MeshAgent runner |
| `results_50_*.json` | 50 条三组 raw benchmark 结果 |
| `results_malt_paper50_full_r3_*.json` | Full MeshAgent 50 条、每题 3 次、带 abstention 的结果 |

`run_malt_paper_reproduction.py` 中对 Full MeshAgent 还原了以下流程：

1. 本地 hybrid RRF 检索 constraints。
2. 检索相关 tool 片段。
3. 使用 CoT 将 query 拆成 3 个步骤。
4. 每步生成 `process_graph(graph_data)` 代码。
5. 执行代码并捕获 execution error。
6. 对 graph 输出运行结构约束检查。
7. 出错后携带 query、step、constraints、tool 和 error trace 重新生成代码。
8. 最多执行 5 轮 error reduction。
9. 记录每次生成代码、ground truth code、返回预览、debug 次数、constraints、tools 和 steps。
10. 基于多次输出一致性和 debug 次数计算 confidence，并在低于阈值时 abstain。

## 4. 实验设置

实验环境为 Windows PowerShell，本地路径：

```text
F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt
```

执行命令如下。

三组 raw benchmark：

```powershell
cd F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt
..\venv\Scripts\python.exe run_reproduction_benchmark.py --limit 50 --timeout 120 --output-prefix results_50
```

更接近原文流程的 Full MeshAgent：

```powershell
cd F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt
..\venv\Scripts\python.exe run_malt_paper_reproduction.py --limit 50 --runs 3 --timeout 120 --groups "Full MeshAgent" --output-prefix results_malt_paper50_full_r3
```

前 50 条 query 的难度分布根据仓库原始 query 顺序和注释推断如下：

| 难度 | 数量 |
|---|---:|
| easy | 31 |
| medium | 12 |
| hard | 7 |

## 5. 实验一：三组 Raw Accuracy 对比

第一组实验比较 Baseline、+Constraints 和简化 Full MeshAgent。每条 query 执行一次，不使用 paper-style repeated runs，也不计算 reliable accuracy。

结果文件：

```text
results_50_Baseline.json
results_50_Constraints.json
results_50_Full_MeshAgent.json
results_50_summary.json
```

结果如下：

| 方法 | 正确数 | 总数 | Raw Accuracy |
|---|---:|---:|---:|
| Baseline | 16 | 50 | 32.00% |
| +Constraints | 40 | 50 | 80.00% |
| Full MeshAgent, simplified | 34 | 50 | 68.00% |

该实验说明显式约束对 MALT 任务非常有效。Baseline 在许多查询中无法正确理解图结构和容量计算规则，而加入全部 constraints 后，正确率从 32.00% 提升到 80.00%。不过，简化版 Full MeshAgent 没有超过 +Constraints，说明仅加入 CoT 和初步 debug 并不一定提升结果，错误的分步生成反而可能破坏原本可由约束直接解决的简单任务。

## 6. 实验二：Paper-style Full MeshAgent

第二组实验只运行 Full MeshAgent，每条 query 重复 3 次，并加入 confidence/abstention 机制。该实验更接近原文的 reliable accuracy 口径。

结果文件：

```text
results_malt_paper50_full_r3_Full_MeshAgent.json
results_malt_paper50_full_r3_summary.json
```

整体结果如下：

| 指标 | 数值 |
|---|---:|
| Queries | 50 |
| Runs per query | 3 |
| Total attempts | 150 |
| Raw correct | 127 |
| Answered | 127 |
| Abstained | 23 |
| Correct answered | 117 |
| Wrong answered | 10 |
| Abstained wrong | 13 |
| Abstained correct | 10 |
| Raw accuracy before abstention | 84.67% |
| Total accuracy | 78.00% |
| Reliable accuracy | 92.13% |
| Abstain rate | 15.33% |
| Abstain accuracy | 86.67% |
| Abstain precision | 56.52% |
| Abstain recall | 56.52% |

其中指标定义如下：

```text
Raw accuracy before abstention = 所有 attempts 中生成结果正确的比例
Total accuracy = 未拒答且正确的 attempts / 全部 attempts
Reliable accuracy = 未拒答且正确的 attempts / 未拒答 attempts
Abstain rate = 拒答 attempts / 全部 attempts
```

与实验一中的简化 Full MeshAgent 相比，paper-style Full MeshAgent 的 raw accuracy 从 68.00% 提升到 84.67%，说明 query-specific constraints、tool retrieval、多次运行和 error reduction 对 MALT 有明显帮助。加入 abstention 后，系统只对 127 个 attempts 给出答案，其中 117 个正确，reliable accuracy 达到 92.13%。

## 7. 按难度分析

Paper-style Full MeshAgent 的按难度结果如下：

| 难度 | Raw Accuracy | Reliable Accuracy | Abstain Rate |
|---|---:|---:|---:|
| easy | 89/93 = 95.70% | 81/83 = 97.60% | 10/93 = 10.75% |
| medium | 32/36 = 88.90% | 30/30 = 100.00% | 6/36 = 16.67% |
| hard | 6/21 = 28.60% | 6/14 = 42.90% | 7/21 = 33.33% |

结果显示，Full MeshAgent 对 easy 和 medium 查询效果较好。easy 查询 raw accuracy 达到 95.70%，medium 查询在 abstention 后 reliable accuracy 达到 100.00%。但 hard 查询仍然是主要短板，raw accuracy 只有 28.60%，reliable accuracy 也只有 42.90%。

错误主要集中在以下 query：

```text
0/3 correct: #02, #15, #16, #17, #19, #21
2/3 correct: #10, #11, #13, #22, #46
3/3 correct: 39 queries
```

其中 #15、#16、#17、#21 等 hard graph manipulation queries 的问题最明显。这类任务要求模型不只是返回结构合法的 graph，还要满足复杂的 query intent，例如正确删除交换机及其端口、平衡容量、选择最优位置或构造特定子图。当前 verifier 能检查图结构合法性，但不能充分判断输出图是否真正满足查询意图，因此部分错误 graph 会以较高 confidence 被回答。

## 8. 与原文结果的差异

本次实验的 Full MeshAgent reliable accuracy 为 92.13%，低于原文 MALT 中报告的约 98.6%。主要原因包括：

1. 本实验只在前 50 条 MALT queries 上运行，未覆盖完整 benchmark。
2. 每条 query 重复 3 次，而原文设置为每条 query 运行 5 次以降低方差。
3. 本实验没有复现 GPT-4o、Gemini、DeepSeek 多模型对比。
4. 本实验没有进行 fine-tuning，因此无法复现 CoT+Fine-tuned with MeshAgent。
5. 本实验没有完整复现人工审核的动态 constraint evolution。
6. 当前 validation 主要覆盖结构约束，对复杂 graph mutation 的 intent-level correctness 检查不足。
7. 当前 confidence 对等价输出的归一化不够完善，例如 list 顺序不同或数值字符串格式不同会降低一致性。

因此，本文结果应理解为 MALT 运行时核心机制的复现，而不是论文所有实验条件的完全复制。

## 9. 改进实验：Confidence 等价输出归一化

在复现实验中观察到，部分正确结果由于输出格式差异被误拒答。例如，返回 list 时不同运行的元素顺序不同，但 evaluator 使用无序列表比较；又如数字文本 `16000` 与 `16000.0` 在语义上等价，但原 confidence 计算会将其视为不同输出。这会降低 semantic consistency，从而导致正确答案被 abstain。

为此，本文新增离线分析脚本 `reanalyze_confidence.py`，不重新调用 LLM API，只读取已有结果中的 `return_preview`，在计算 semantic consistency 前进行归一化：

1. 对 `list` 输出按元素规范化后排序，使其与无序列表比较逻辑一致。
2. 对 `text` 中的纯数字字符串进行数值归一化，使 `16000` 和 `16000.0` 视为等价。
3. 对嵌套结构中的数值字段做同样的数值归一化。

运行命令如下：

```powershell
cd F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt
..\venv\Scripts\python.exe reanalyze_confidence.py --input results_malt_paper50_full_r3_Full_MeshAgent.json --output results_malt_paper50_full_r3_confidence_improved.json --summary-output results_malt_paper50_full_r3_confidence_improved_summary.json --threshold 0.7 --max-debug 5 --timeout 8 --no-reexecute
```

改进前后指标如下：

| 指标 | 改进前 | 改进后 | 变化 |
|---|---:|---:|---:|
| Raw accuracy before abstention | 84.67% | 84.67% | 0.00% |
| Total accuracy | 78.00% | 84.00% | +6.00% |
| Reliable accuracy | 92.13% | 91.97% | -0.16% |
| Abstain rate | 15.33% | 8.67% | -6.66% |
| Abstain accuracy | 86.67% | 92.00% | +5.33% |
| Abstain precision | 56.52% | 92.31% | +35.79% |
| Abstain recall | 56.52% | 52.17% | -4.35% |
| Abstained correct | 10 | 1 | -9 |

可以看到，该改进没有改变模型生成结果本身，因此 raw accuracy 不变；但它显著减少了正确答案被误拒答的问题，使 total accuracy 从 78.00% 提升到 84.00%，abstain precision 从 56.52% 提升到 92.31%。这说明 confidence 计算需要与 evaluator 的等价判断保持一致，否则会把格式差异误判为语义不一致。

不过，该改进也使 wrong answered 从 10 增加到 11，abstain recall 略有下降。这说明仅靠格式归一化无法解决 hard graph mutation 中的高置信错答问题。后续更深入的改进应针对 graph mutation 查询加入 intent-level validation。

## 10. 结论

本次复现实验证明了 MeshAgent 的核心思想在 MALT 场景中是有效的。三组 raw benchmark 显示，加入显式 constraints 后，正确率从 Baseline 的 32.00% 提升到 80.00%，说明网络结构和容量计算等领域约束能够显著改善 LLM 代码生成质量。

进一步实现更接近原文流程的 Full MeshAgent 后，在 50 条 query、每条 3 次运行的设置下，系统达到 84.67% raw accuracy 和 92.13% reliable accuracy。相比简化 Full MeshAgent 的 68.00%，改进明显，说明 query-specific constraint retrieval、CoT 分解、tool retrieval、error reduction 和 confidence-based abstention 对提升可靠性有实际作用。

不过，实验也暴露出当前复现与原文完整系统之间的差距。hard 查询中的复杂拓扑修改和优化任务仍然容易失败，且当前 verifier 主要检查结构合法性，无法完全判断输出是否满足 query intent。本文进一步通过 confidence 等价输出归一化减少了正确答案被误拒答的问题，但对于稳定生成的错误 graph，仍需要更强的 intent-level validation。

综上，本实验完成了 MeshAgent 在 MALT 应用上的核心运行时复现，并得到了与原文趋势一致的结论：显式约束和基于验证的错误修复能够显著提升 LLM 网络管理任务的准确性和可靠性。
