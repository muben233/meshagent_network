# 原始代码与论文的剩余关键差异记录

本文档只记录当前认为需要在复现实验报告中说明的关键差异；以下内容不包括已排除项：notebook/runtime hybrid retrieval 差异、per-step constraint 粒度、debug loop 次数、每题运行次数、实验矩阵范围、动态 constraint 更新闭环。

## 1. Constraint 数据结构不完整

论文中每条 constraint 的完整结构应包含：

- `label`
- `invariant`
- `validation test`

但原始 MALT 代码中的 `F:\vs_program\meshagent_network\MeshAgent-main\app-malt\data\rag_constraints.json` 主要包含：

- `id`
- `label`
- `constraint`

其中 `constraint` 基本对应论文中的自然语言 invariant，但 JSON 中没有保存对应的 validation test。

## 2. Validation Test 与 Constraint 未绑定

论文设计中，validation test 是 constraint database 的组成部分，应当和对应 invariant 一起被检索、使用和维护。

原始代码中，validation test 被单独硬编码在 `F:\vs_program\meshagent_network\MeshAgent-main\app-malt\error_check.py` 的 `MyChecker` 类中，而不是作为 `rag_constraints.json` 中每条 constraint 的字段存在。

这意味着原始实现没有形成：

```text
constraint item = label + invariant + validation test
```

这样的完整结构化约束单元。

## 3. 检索到的 Constraint 与实际执行的 Checker 不一一对应

原始代码运行时会先从 RAG 中检索若干自然语言 constraints，并将其加入 prompt；但后置验证阶段并不是根据“本次检索到哪些 constraints”来动态选择对应 validation tests。

实际执行的是 `MyChecker.evaluate_all()` 中预先写死的一组全局检查函数。因此存在两类不一致：

- 某条 constraint 即使被检索到，也不一定有对应 checker 被执行。
- 某个 checker 即使对应 constraint 没被检索到，也可能仍然被执行。

这与论文中 constraint-guided generation 和 constraint-guided validation 使用同一批 constraint entries 的设计不完全一致。

## 4. Validation 覆盖范围有限

`error_check.py` 中实际启用的检查主要包括：

- 节点是否具有合法 type
- 边是否具有合法 type
- 节点层级关系是否满足部分 hierarchy
- 图中是否存在孤立节点
- 表格中的 bandwidth 是否非零

但 `rag_constraints.json` 中还有一些重要约束没有完整后置验证覆盖，例如：

- capacity 应等于所包含 PORT 的 `physical_capacity_bps` 之和。
- 新增节点时应同时补全层级节点和关系边。
- 更新 graph 时应复制 graph，而不是直接修改输入图。
- PACKET_SWITCH 的 `switch_loc` 属性应位于 `packet_switch_attr` 中。
- 查找某类节点时应检查节点的 `type` 列表，而不只依赖 name。

因此，原始代码的 validation 更像是少量全局 sanity checks，而不是论文中完整的 constraint-level validation tests。

## 5. Confidence 与 Abstention 机制缺失

论文中 MeshAgent 使用 confidence score 判断是否拒答，confidence 基于：

- error check 是否通过；
- 多次输出之间的 semantic consistency；
- debug / correction 轮数。

当 confidence 低于 threshold 时，系统 abstain，以降低高置信错误输出。

原始 `app-malt` 代码中没有实现这一机制。代码只进行执行、验证、ground truth 对比和日志记录，不会根据 confidence threshold 主动拒答。因此论文中的 reliable accuracy / abstention 相关机制，在原始代码中并未完整落地。

## 6. 约束文本与验证逻辑存在命名不一致

原始 `rag_constraints.json` 中 edge type constraint 写的是 `RK_CONTROL`，但原始图数据和部分 prompt / checker 中使用的是 `RK_CONTROLS`。

这说明原始代码中的自然语言 constraint、图数据和 checker 之间没有完全统一的结构化绑定，也进一步体现出 validation test 并非由 constraint database 自动驱动。

## 总结

原始 `app-malt` 代码实现了论文中 constraint-guided generation、CoT、自修复和部分验证的思想，但没有完整实现论文定义的 constraint abstraction。

最关键差异是：

```text
论文：constraint = label + invariant + validation test
代码：constraint JSON = id + label + constraint 文本；validation test 另写在 error_check.py
```

因此，复现实验报告中应表述为：原始代码提供了 MeshAgent 思想的 MALT 工程实现，但其 constraint database 与 validation 机制相比论文描述存在简化。
