# LoopDetectionMiddleware Rewind 设计

## 问题

当前 `LoopDetectionMiddleware` 是二元处理：`warn`（在 `after_model` 追加警告 HumanMessage）→ `hard_stop`（剥离 tool_calls 强制终结）。两层之间缺少一个"切除污染 + 重试"的中间层。

模型一旦进入 tool-call 循环：

- 注入的警告是追加到尾部，**污染历史完整保留**，模型下一轮仍然看到失败的 tool_call 和 ToolMessage 序列
- 警告若未能自救，下一跳直接跳到 `hard_stop` 终结整个 turn
- 整个过程中，被证伪的 tool 调用始终留在 context 里——消耗 token、分散注意力、加速 context rot

Thariq（Anthropic）的"rewind 代替 correction"原则指出：**context 被污染后让模型在污染上自救，决策质量一定低于回到污染前重新决策**。现有 `scripts/trace_replay.py` 已提供离线 rewind 能力，但仅供人用、未接入 agent 运行时。

## 目标

1. **自动化 rewind** — 一旦确认 loop，自动让模型看到一个不含污染的视图
2. **零状态侵入** — 不修改 `AgentState`，不修改 LangGraph state checkpoint，不引入 sentinel/auto-resume 协议
3. **零路由侵入** — 不影响 ReAct 路由决策（条件边、END、tools 节点）
4. **规则型 hint** — V1 基于失败 trace 生成结构化提示，告知模型"哪些路径已试过且失败"
5. **可降级** — 边界情况（无法定位 loop、消息缺 id、抽取失败）自动降级，不打断 agent 运行
6. **为 V2 留好接口** — Cross-hash 无进展检测、周期性检测、judge-based hint 都能在不破坏架构的前提下加入

## V1 适用范围与现实预期

V1 的 hash-based detection 仅捕获**完全相同 tool_call 重复 ≥ N 次**的失败模式。这类失败在真实 oncall 场景中估计仅占 **15-20%**。

其余 80% 的失败模式 V1 抓不到，需要 V2 detector 覆盖：

| 失败模式 | 占比估计 | V1 抓得到？ | V2 对应 detector |
|---------|---------|-----------|------------------|
| 完全 hash 重复 | ~15% | ✓ | — |
| 变 args 同 tool 乱翻 | ~30% | ✗ | V2.5 tool spam |
| 多 tool 交错无进展 | ~25% | ✗ | V2.1 no-progress |
| 周期性 ping-pong | ~10% | ✗（除非周期=1） | V2.2 periodic |
| 长时间纯探索不收敛 | ~15% | ✗ | V2.1 no-progress |
| 真正卡死无解 | ~5% | hard_stop 兜底 | — |

（占比为估计值，无 baseline 实测；上线后应通过 metric 校准）

### V1 的核心价值

V1 单独上线**不会**消除 oncall agent 的卡死问题。但它仍然值得做：

1. **误报率近 0**：完全相同 tool_call 重复 ≥ N 次没有合法理由——一旦命中确凿是 loop，干预无误判风险
2. **命中时危害最大**：那 15% 一旦发生几乎必然导致 recursion_limit 触发或长时间高成本浪费，是单次最贵的失败模式
3. **V2 共享脚手架**：wrap_model_call patching、hint 生成、多区间合并、测试基础设施都为 V2.1/V2.5 复用——脚手架建对，每个 V2 detector 加入只需 1-2 天

### 上线后必须监控的关键 metric

```
ratio = first_detected_total / hard_stop_total
```

- `ratio < 1`：V1 detection 触发少但 hard_stop 经常触发 → 大量真实 loop V1 抓不到 → V2 应尽快推进
- `ratio > 3`：V1 已抓住主要矛盾 → V2 推进可放缓
- 中间区间 → 按 hard_stop 绝对频率决定优先级

V2 detector 中**优先实施 V2.1 (no-progress)**——它能覆盖最广的"乱探索"模式（约占 40%）。V2.5 (tool spam) 次之。

---

## 核心机制：wrap_model_call 视图层 patching

```
┌─────────────────────────────────────────────────────────────┐
│ agent.stream() 进入 model 节点                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │ wrap_model_call hook       │
            │  (LoopDetectionMiddleware) │
            └────────────┬───────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │ scan request.messages,detect loop:  │
        │   - 计算每个 AIMessage 的 tool_call  │
        │     hash                            │
        │   - 找出 count >= rewind_threshold  │
        │     的 hash                         │
        └────────────┬────────────────────────┘
                     │
              ┌──────┴──────┐
              │             │
        无 loop           有 loop
              │             │
              │             ▼
              │      ┌────────────────────────────┐
              │      │ 构造 patched messages:      │
              │      │   pre_loop + [hint] +      │
              │      │   post_loop                │
              │      │ (state.messages 不动!)      │
              │      └────────────┬───────────────┘
              │                   │
              ▼                   ▼
       ┌────────────────────────────────┐
       │ handler(request) 或             │
       │ handler(request.override(       │
       │   messages=patched))           │
       └──────────────┬─────────────────┘
                      │
                      ▼
          ┌──────────────────────────┐
          │ 模型生成新 AIMessage      │
          │ 自然 append 到 state      │
          └──────────────┬───────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ after_model hook         │
            │  - 检查是否触发 hard_stop │
            └──────────────┬───────────┘
                           │
                           ▼
                  ReAct 路由继续
```

### 为什么是 wrap_model_call 而不是 before_model / after_model

`wrap_model_call` 包住模型调用本身：修改的是 `ModelRequest.messages`（**给模型看的视图**），**不返回 state 更新**。这意味着：

- `add_messages` reducer 不参与，state checkpointer 里的 messages 完全不变
- LangGraph 路由完全不受影响（条件边看 state，state 没动）
- 跨多次模型调用，patching 自动重新计算（幂等）

参考实现：`dangling_tool_call_middleware.py` 用同样的范式处理用户中断造成的 dangling tool_calls，已在生产验证。

### 永远 patch 哲学

**只要 loop hash 存在于 `request.messages` 中，每次 model 调用都重新 patch**。

| 性质 | 含义 |
|------|------|
| **幂等** | 同 input 永远同 output |
| **无状态** | 不需要 per-thread 跟踪"是否已 patch" |
| **稳健** | 进程重启、多 worker、trace_replay 复跑——行为一致 |
| **可观测** | state 里完整保留 loop 现场，trace_replay 能复盘 |

成本：每次 model 调用 O(N) 扫描 + O(N) hash 计算（N = messages 数）。100 条消息 ≈ 数百微秒，相对 LLM 调用本身可忽略。

## 三层响应阶梯

旧设计 warn(3) + hard_stop(5)。新设计：

| 层 | 阈值 | Hook | 动作 | State 改动 | 本轮后续 |
|----|------|------|------|-----------|---------|
| ~~warn~~ | ~~3~~ | ~~after_model~~ | ~~追加 HumanMessage~~ | ~~否~~ | **删除** |
| **rewind** | 3 | `wrap_model_call` | patch 给模型看的视图 | 否 | 模型基于干净视图继续生成 |
| hard_stop | 5 | `after_model` | 剥离 tool_calls + 哨兵 | 替换最后 1 条 | END |

### Warn 删除的理由

`warn` 当年存在是因为没有更好工具——只能"追加一条 HumanMessage 提醒模型"。但污染历史还在，提醒效果有限。

`rewind` 通过 patching 等于**每次都把"failed paths ruled out"放进模型视图**——比追加更有效，且不打断流程。warn 的价值被完全 superseded。

### Hard_stop 保留的理由

`rewind` 在 count=3 介入。如果模型即便看到 ruled-out hint 仍然在**新 hash** 上重复（例如换 args 但语义同样无效），需要 hard_stop 兜底。

`hard_stop` 的协议（`[FORCED STOP]` 哨兵 + tool_calls 清空 + END 路由）保留不变——它是 harness 层"模型彻底放弃"的暗号。

## 检测算法

### 单 hash 计数（V1）

V1 检测**所有** count ≥ rewind_threshold 的 hash，并合并 overlap/adjacent 的区间。这避免"只 patch 第一个 loop、其它 loop 残留污染"的盲点。

```python
def _detect_all_loops(self, messages: list) -> list[tuple[str, int, int]]:
    """返回所有 count >= threshold 的 (loop_hash, first_idx, last_idx),按 first_idx 升序。"""
    hash_counts: dict[str, int] = {}
    hash_first_idx: dict[str, int] = {}
    hash_last_idx: dict[str, int] = {}
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            continue
        h = _hash_tool_calls(tcs)
        hash_counts[h] = hash_counts.get(h, 0) + 1
        hash_first_idx.setdefault(h, i)
        hash_last_idx[h] = i
    
    loops = [
        (h, hash_first_idx[h], hash_last_idx[h])
        for h, count in hash_counts.items()
        if count >= self.rewind_threshold
    ]
    return sorted(loops, key=lambda t: t[1])


def _merge_overlapping(
    self, regions: list[tuple[str, int, int]]
) -> list[tuple[set[str], int, int]]:
    """合并 overlap 或 adjacent 的区间,聚合涉及的 hash 集合。
    
    overlap 判定:next_start <= prev_end + 1 (允许相邻)
    """
    if not regions:
        return []
    merged: list[tuple[set[str], int, int]] = []
    for h, start, end in regions:
        if merged and start <= merged[-1][2] + 1:
            hashes, m_start, m_end = merged[-1]
            merged[-1] = (hashes | {h}, m_start, max(m_end, end))
        else:
            merged.append(({h}, start, end))
    return merged
```

`loop_end_idx`（每个 loop 的 last_idx）取该 hash 最后一次出现的 AIMessage 位置；patch 时连带其后的 ToolMessage 一起切掉（详见下节）。

### Hash 计算

`_hash_tool_calls` 沿用现有实现（`loop_detection_middleware.py:105-123`）：基于 `name + salient args` 做有序的 MD5 hash，对 tool_calls 多重集顺序无关。

### 复用现有工具函数

- `_normalize_tool_call_args`
- `_stable_tool_key`（含 read_file 行号桶化、write_file/str_replace 全 args hash）
- `_hash_tool_calls`

抽到独立模块 `loop_hash.py` 便于 V2 检测器复用（cross-hash、periodic 都基于同一套 hash）。

## Patch 区间定义

V1 支持**多个 loop 并存**。流程：

```
1. _detect_all_loops(messages) → list[(hash, start, end)]
2. _merge_overlapping(loops)   → list[(set[hash], start, end)]
3. 对每个合并后的区间:
     expanded_end = 扩展到包含所有配对 ToolMessage
     hint         = build_multi_hint(messages, hashes, start, expanded_end)
4. 从后往前依次替换(避免 index 漂移):
     patched = patched[:start] + [HumanMessage(hint)] + patched[expanded_end+1:]
```

### 单区间 expanded_end 计算

从该区间的 `end_idx` 出发向后扫，吸收所有 `tool_call_id` 属于该区间内任一 loop AIMessage 的 ToolMessage，直到遇到一个不属于的消息。

### 从后往前应用的原因

每次替换都会改变后续消息的 index。如果按 start_idx 升序处理，第一次替换后第二个区间的 start/end 就失效了。改为按 start_idx 降序处理：先替换最末尾的区间（不影响前面 index），再依次往前。

### 删除粒度：整条 AIMessage 而非仅 tool_calls

被切除的不只是 tool_calls 字段，而是**整条 AIMessage**（包含其 `content` 文本推理）和**配对的 ToolMessage**。三个理由：

1. **避免 orphan ToolMessage**：如果只剥 `tool_calls` 保留 AIMessage，紧随其后的 ToolMessage 找不到对应的 `tool_call_id`，LLM 会报 message 格式错误（这正是 `DanglingToolCallMiddleware` 反向解决的同一类问题）。

2. **避免 ReAct 语义破坏**：ReAct 约定 `AIMessage(content=..., tool_calls=[])` 表示"最终答案"。若 loop 区间留下多条这样的 AIMessage，模型看到对话中段反复出现多个"final answer"——逻辑混乱。

3. **避免污染推理继承**：loop 区间的 AIMessage `content` 多半是模型基于已污染 context 生成的"伪自洽推理"（例："I'll try grep again because last time it returned nothing"）。保留这些文本会让模型继承错误判断。

### 信号补偿：原始意图保留

完全删除 AIMessage 会丢失一个有价值的信号——**模型最初进入 loop 时的目标描述**（往往是合理的）。例如：

```
loop 第一条 AIMessage:
  content="I need to check if foo.py has the bug"
  tool_calls=[read_file("foo.py")]   ← 这个失败了,但目标是对的
```

为补偿这个丢失，hint 中提取首条 loop AIMessage 的 `content` 作为 "Original intent" 段落（详见下节 Hint 格式）。这是**唯一被保留的 AIMessage 信号**——首条意图通常出现在污染发生之前，可信度较高。

提取规则：

- 取 `messages[loop_start_idx]` 的 `content`
- 处理 Anthropic 的 list content 结构：拼接所有 `{"type": "text", ...}` 块
- 截断到 ~120 字符
- 若 content 为空或 < 20 字符，省略 "Original intent" 段（避免无信息噪音）

## Hint 格式

```
[LOOP RECOVERY] Original intent at the start of this loop:
  "{first_aimessage_content_truncated_120_chars}"

These tool-call paths have been ruled out:

Failed with errors:
  ✗ {tool}({salient_args}) → {result_preview}
  ✗ ...

Returned unhelpful results:
  ○ {tool}({salient_args}) → {result_preview}

Do NOT retry the ruled-out paths. Reassess your approach toward the original
intent. Choose:
  (a) a different tool
  (b) different arguments
  (c) produce a final answer using partial information.
```

### "Original intent" 段落规则

- 取**合并区间起点的 AIMessage**（即 `messages[merged_start]`）的 `content`，按上节"提取规则"截断处理
- 若 content 为空或 < 20 字符，**整个 "Original intent" 段落（含标题行）省略**
- footer 措辞 "Reassess your approach toward the original intent" 保留——即便 intent 段落省略，这句话仍提示模型"基于上下文目标重新规划"，不依赖 intent 是否实际呈现

### 多 hash 合并区间的 hint

当一个合并区间涵盖多个 loop hash（场景 B/C）时：

- "Original intent" 段落只取一次（合并区间起点）
- "Failed with errors" 和 "Returned unhelpful results" 段落混合该区间内**所有 hash** 的 tool_call/result 配对
- 仍按 `(tool_name, salient_args)` 去重（不分 hash），避免同一调用在两个 hash 都出现而双列

### salient_args 字段白名单

与 `_stable_tool_key` 完全一致（避免两处漂移）：

```
path, url, query, command, pattern, glob, cmd
```

无匹配字段时 fallback 到 `str(args)[:40]`。

### 错误/非错误分组

- **Failed with errors**: `ToolMessage.content` 以 `"Error:"` 开头
- **Returned unhelpful results**: 其他（含空返回、低信息返回）

### 去重规则

按 `(tool_name, salient_args)` 键去重，只保留首次结果。避免同 loop 内相同调用重复展示。

### 抽取失败降级

若 `loop_start_idx` 之后无法配对出任何 `(AIMessage.tool_call, ToolMessage)`，hint 降级为常量字符串：

```
[LOOP RECOVERY] Repeated tool calls detected. Stop calling tools and produce
your final answer using whatever information you have so far.
```

## 接口变更

### LoopDetectionMiddleware.__init__ 变更

```python
def __init__(
    self,
    rewind_threshold: int = 3,    # NEW (取代 warn_threshold 同位)
    hard_limit: int = 5,           # 保留
    # 删除:warn_threshold, window_size, max_tracked_threads
):
```

### LoopDetectionMiddleware 状态字段

| 字段 | V1 | 新设计 |
|------|----|----|
| `_history` | OrderedDict（per-thread sliding window） | **删除** |
| `_warned` | dict（per-thread warn 跟踪） | **删除** |
| `_lock` | threading.Lock | **删除** |
| 新增 | — | `_reported_loops: dict[str, set[str]]`（观测去重缓存） |

新设计 middleware **不持有任何影响 patching 行为的可变状态**。`_reported_loops` 是观测层去重用的纯优化缓存（详见"可观测性"），清空后行为完全等价。

### LoopDetectionMiddleware hooks 变更

| Hook | V1 | 新设计 |
|------|----|----|
| `before_model` | 无 | 无 |
| `wrap_model_call` | 无 | **新增**（rewind patching） |
| `after_model` | 含 warn + hard_stop 两路径 | 仅 hard_stop |

### AgentState

**不改**。

### DeerFlowClient

**不改**。新设计不需要 harness 层 auto-resume，旧 spec 中的 `_post_run_resume_check`、`max_auto_resumes`、`RESUME_SENTINEL` 全部移除。

### agent.yaml 配置示例

```yaml
loop_detection:
  rewind_threshold: 3
  hard_limit: 5
```

## 不变量

| # | 不变量 |
|---|-------|
| I1 | `wrap_model_call` 不返回 state 更新；`request.override(messages=...)` 仅修改本次模型请求视图 |
| I2 | 同样 `request.messages` 输入永远产生同样 patched 输出（幂等） |
| I3 | Middleware 不持有**影响 patching 行为**的 per-thread 可变状态。观测/优化用缓存（`_reported_loops`、V2 的 `_hint_cache`）不在此约束内——这些缓存清空后行为完全等价 |
| I4 | Hint 中 `salient_args` 字段白名单与 `_stable_tool_key` 完全一致 |
| I5 | Patched 视图中不存在 orphan ToolMessage（被切走的 AIMessage 的 tool_call_id 对应的 ToolMessage 也被切走） |
| I6 | Loop 区间任一 AIMessage 的 tool_calls 无法 hash（异常 args）时跳过 patching，降级为透传 |
| I7 | Hard_stop 的协议不变（`[FORCED STOP]` 哨兵 + tool_calls 清空 + 路由 END） |
| I8 | `state["messages"]` 在 wrap_model_call 前后完全一致；只有模型自然产出的 AIMessage 才会写入 state |
| I9 | 所有 count ≥ rewind_threshold 的 loop hash 必须被某个 patched 区间完全覆盖（不允许"只 patch 部分 loop"） |
| I10 | 多区间 patching 必须从后往前应用（按 start_idx 降序），保证前置区间的 index 不被后置 patch 影响 |

## 失败模式与降级

| 故障 | 处理 |
|-----|------|
| `_detect_loop` 异常 | 透传 request，记 error 日志 |
| Hint 抽取空配对 | 降级为常量 fallback hint |
| `loop_end_idx` 之后找不到完整 ToolMessage 配对 | 不切 patch，透传 request |
| Patched 后 messages 为空 | 透传 request（理论上不会发生，防御性） |
| Hash 计算异常（args 含不可序列化对象） | 该消息 hash 跳过，记 warning |

**核心原则**：任何降级都不打断 agent 运行；最坏情况退化为"没装 LoopDetectionMiddleware"。

## 可观测性

### 设计原则

"永远 patch" 哲学下，同一 loop 会在每次 model 调用都被 patch。如果直接每次上报，会出现：

- 单个 loop 产生数十条相同日志（噪音）
- counter 累加到 `count = patch 次数`，看起来像"高频 loop 故障"，实际是 1 个 loop 被持续 mask（语义失真）

因此**区分两类事件**：

| 事件类型 | 频率 | 语义 | 上报策略 |
|---------|------|------|---------|
| **首次检测到 loop** | 每个 (thread, hash) 仅 1 次 | semantic：agent 真的卡住了 | WARNING 日志 + counter +1 |
| **持续 patching** | 每次 model 调用 | mechanical：mask 一个已知 loop | 不打日志、不计数 |
| **降级/跳过** | 罕见 | 容错路径触发 | WARNING 日志 + counter +1 |
| **Hard_stop 触发** | 极罕见 | agent 彻底放弃 | ERROR 日志 + counter +1 |

### 去重机制

Middleware 持有观测专用缓存：

```python
self._reported_loops: dict[str, set[str]] = defaultdict(set)
# thread_id → 已上报过的 loop_hash 集合
```

`wrap_model_call` 检测到 loop 时判断：

```python
is_first = loop_hash not in self._reported_loops[thread_id]
if is_first:
    self._reported_loops[thread_id].add(loop_hash)
    logger.warning("loop.rewind.first_detected", extra={...})
    metrics.first_detected.inc()
# patching 不论是否首次都执行（行为不受观测影响）
```

**性质**：
- 该缓存与 V2.3 的 `_hint_cache` 同性质——纯优化、可丢失、不影响行为
- 进程重启后最多导致部分 loop 被重复上报一次，不破坏正确性
- 不需要锁（dict/set 操作在 GIL 下原子）
- 不在 I3 约束内

### 日志事件

```
loop.rewind.first_detected   (WARNING)
    extra={thread_id, loop_hash, loop_start_idx, loop_end_idx,
           patch_size_messages, hint_chars, hint_pairs_count}

loop.rewind.skipped          (WARNING)
    extra={thread_id, loop_hash, reason ∈ {no_pairs, hash_error, empty_patch}}

loop.hard_stop.fired         (ERROR)
    extra={thread_id, loop_hash, count}
```

`still_patching` 事件**不打日志**——观测层无需感知。如需调试可临时把检测到的 loop 信息加 DEBUG 级日志，生产环境关闭。

### Metric

```
oncall.loop_detection.first_detected_total
    type=counter
    labels: outcome ∈ {patched, skipped_no_pairs, skipped_hash_error,
                       fallback_hint}

oncall.loop_detection.hard_stop_total
    type=counter

oncall.loop_detection.patch_size_messages
    type=histogram
    note: 仅在 first_detected 时记录
```

**避免高 cardinality label**：counter 不带 `loop_hash` 或 `thread_id` 作为 label——这两者会让 Prometheus 时间序列爆炸。

## 测试矩阵

| ID | 输入 | 期望 |
|----|------|------|
| T1 | 同 hash 重复 1-2 次 | 透传，不 patch |
| T2 | 同 hash 重复 3 次 | patch：messages[:loop_start] + [HumanMessage(hint)] + tail |
| T3 | 重复 5 次 | wrap_model_call 仍 patch；after_model 触发 hard_stop（剥离 tool_calls + 哨兵） |
| T4 | Hint 内容结构 | 包含 "Failed with errors" 段落、salient_args、按 (tool, args) 去重 |
| T5 | Hint 抽取空配对 | fallback 到常量 hint |
| T6 | 同 patched 输入跑 2 次 | 两次输出完全相同（幂等） |
| T7 | Loop 后模型产出新 AIMessage,再次进入 wrap_model_call | 新 patched 包含原 hint + 新 AIMessage（loop 区间持续被切） |
| T8 | 多个 hash 同时达到 threshold | 选最早出现的那个 hash 进行 patch |
| T9 | 切走 AIMessage 后留下 orphan ToolMessage 风险 | patched 内确实把对应 ToolMessage 也切走（验证 I5） |
| T10 | Args 含不可序列化对象 | 该消息 hash 跳过，不抛异常 |
| T11 | 中间件实例被多 thread 并发调用 | 行为正确（验证无共享状态、无锁需求） |
| T12 | 同一 (thread, loop_hash) 在 5 次 model 调用中持续 patch | `first_detected` 日志仅 1 次、counter +1；后续 4 次不打日志、不计数（验证去重） |
| T13 | 不同 thread 出现相同 loop_hash | 各 thread 各自上报 1 次 first_detected（验证 thread 隔离） |
| T14 | 首条 AIMessage content 含 "Check foo.py for bug" | hint 含 `Original intent: "Check foo.py for bug"` 段落 |
| T15 | 首条 AIMessage content 为空或 < 20 字符 | hint 省略整个 Original intent 段落,仅保留 ruled-out 列表 + footer |
| T16 | 首条 AIMessage content 是 list（Anthropic thinking 模式） | 拼接所有 text 块、截断 120 字符、正确放进 hint |
| T17 | 两个分离 loop（hash_A [5-7]、hash_B [12-14]） | 两个独立 patched 区间，各自含 hint；中间未污染消息保留 |
| T18 | 两个交错 loop（hash_A [5-9]、hash_B [6-10] overlap） | 合并为单个区间 [5-10]，hint 含两个 hash 的 ruled-out 列表 |
| T19 | 嵌套 loop（hash_X [5-15]、hash_Y [8-10] inside X） | 合并为单个区间 [5-15]，hint 同时呈现 X 和 Y 的失败 |

## 文件结构

```
deer-flow/backend/packages/harness/deerflow/agents/middlewares/
    loop_detection_middleware.py    # 重构:新增 wrap_model_call,删除 warn 路径
    loop_hint_builder.py            # NEW:build_rule_hint 等纯函数
    loop_hash.py                    # NEW:抽出 _hash_tool_calls / _stable_tool_key 等
                                    #     便于 V2 检测器复用
```

`loop_hint_builder.py` 和 `loop_hash.py` 都是无副作用纯函数模块，便于直接单测、不依赖 LangGraph runtime。

## 不做的事（V1 范围之外）

- 不修改 `AgentState` 类型
- 不修改 LangGraph ReAct 路由 / 条件边
- 不引入 RESUME_SENTINEL、auto-resume 协议、`max_auto_resumes`、`max_rewinds_per_thread`
- 不集成 `trace_replay.py`（rewind 是运行时能力，trace_replay 是离线诊断，保持分离）
- 不替换 `hard_stop`（它是终极安全网）
- 不做 tool 级别豁免（`rewind_exempt_tools` 不引入；loop 就是 loop）
- 不与 `SubagentLimitMiddleware` 交互

## V2 候选扩展

V2 的扩展都是在 `wrap_model_call` 内部加新检测器或 hint 生成器。**核心架构不变，扩展点清晰**。

### V2.1 Cross-hash 无进展检测

**问题**：模型轮换调多个不同工具（每个 hash 都没到 V1 阈值），但整体无进展——AIMessage 全是 tool_call、几乎不形成假设/结论/进展文本。这是 oncall 卡死最常见的形态。

**核心信号**：进展不是看 tool 调用次数，而是看**模型是否在思考**——"meaningful text" 在 AIMessage.content 中的占比。

**检测**：

```python
def _detect_no_progress(
    self,
    messages: list,
    window: int = 15,
    min_window_to_trigger: int = 10,
    no_progress_ratio: float = 0.85,
) -> tuple[int, int] | None:
    """检测"近 window 个 AIMessage 中 ratio 比例是裸 tool_call、无 meaningful text"。
    
    返回 (patch_start_idx, patch_end_idx)，None 表示未触发。
    """
    # 取最近 window 个 AIMessage(若不足 min_window_to_trigger 则放弃)
    recent_ai_with_idx: list[tuple[int, AIMessage]] = []
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            recent_ai_with_idx.append((i, messages[i]))
            if len(recent_ai_with_idx) >= window:
                break
    if len(recent_ai_with_idx) < min_window_to_trigger:
        return None
    
    recent_ai_with_idx.reverse()   # chronological
    
    no_progress = sum(
        1 for _, ai in recent_ai_with_idx
        if getattr(ai, "tool_calls", None)
        and not _has_meaningful_text(ai.content)
    )
    if no_progress / len(recent_ai_with_idx) < no_progress_ratio:
        return None
    
    patch_start = recent_ai_with_idx[0][0]
    patch_end = recent_ai_with_idx[-1][0]
    return patch_start, patch_end
```

**`_has_meaningful_text` 定义**：

```python
_FILLER_PHRASES = {
    "let me", "i'll", "i will", "let's", "now i'll", "now let me",
    "i need to", "going to", "next i'll",
}

def _has_meaningful_text(content) -> bool:
    """判断 AIMessage content 是否包含有信息量的推理/结论。
    
    规则:
      1. 提取所有 text(支持 list 结构)
      2. 总长度 < 80 字符 → False(只是过场话)
      3. 全部以 filler phrase 开头 → False(机械叙述)
      4. 否则 → True
    """
    text = _extract_text(content)
    if len(text.strip()) < 80:
        return False
    sentences = [s.strip().lower() for s in text.split(".") if s.strip()]
    if not sentences:
        return False
    # 全部 sentence 都以 filler 开头 → 视为无意义
    if all(any(s.startswith(p) for p in _FILLER_PHRASES) for s in sentences):
        return False
    return True
```

**Hint**：

```
[NO PROGRESS] You've made {N} tool calls in the last {window} turns without
forming a hypothesis, conclusion, or progress statement.

Stop exploring. Choose:
  (a) commit to ONE specific approach and pursue it
  (b) summarize what you know and produce a final answer
  (c) explicitly state why the task may not be solvable with available tools
```

**Patch 区间**：

```
patch_start = 最近 window 个 AIMessage 中第一条的 idx
patch_end   = 最近 window 个 AIMessage 中最后一条的 idx + 配对 ToolMessage
patched     = messages[:patch_start] + [HumanMessage(hint)] + messages[patch_end+1:]
```

注意：no-progress 的 patch 区间**比 V1 hash detection 大很多**（典型 15+ AIMessage + 同等数量 ToolMessage）。这是必要的——污染范围本身就广。

**与 V1 hash detection 的优先级**：

V1 hash detection 优先（更确定）。仅当 V1 未触发时才尝试 V2.1。

**边界处理**：

| 场景 | 处理 |
|------|------|
| Thread 早期（AIMessage 数 < min_window_to_trigger） | 直接返回 None，避免误报正常起步阶段 |
| 模型正在做合法长探索（每条都有 meaningful text） | 不触发（ratio 不够）|
| 单条 AIMessage 含数千字 reasoning（罕见但合法） | `_has_meaningful_text` 返回 True，不计入 no_progress |
| 全是 filler phrase 但总长度 > 80 | 视为无意义（filler 检测兜底）|

**调参建议**：

| 参数 | 默认 | 调优方向 |
|------|------|---------|
| `window` | 15 | 短任务降到 10；超长任务可升到 25 |
| `no_progress_ratio` | 0.85 | 误报多→升到 0.9；漏报多→降到 0.75 |
| `min_window_to_trigger` | 10 | 与 window 同步 |

`_has_meaningful_text` 启发式相对脆弱，**建议上线后采集"误报样本"反向调 filler 列表**。

### V2.2 周期性 loop 检测

**问题**：hash 序列呈 `[A, B, A, B, ...]` 或 `[A, B, C, A, B, C, ...]` 等周期模式，单 hash 计数缓慢。

**检测**：

```python
def _detect_periodic(self, hashes, periods=(2, 3, 4), min_cycles=2):
    for p in periods:
        needed = (min_cycles + 1) * p
        if len(hashes) < needed:
            continue
        recent = hashes[-needed:]
        chunks = [recent[i*p:(i+1)*p] for i in range(min_cycles + 1)]
        if all(chunk == chunks[0] for chunk in chunks[1:]):
            return p
    return None
```

**触发条件**：period-2 在 6 步、period-3 在 9 步、period-4 在 12 步。

**Hint**：
```
You're cycling through {tool_A, tool_B, tool_C} in a {p}-step pattern.
This is a non-converging loop. Stop and answer, or pick ONE path and stick with it.
```

### V2.3 Judge-based hint

**问题**：复杂 loop（多 tool 交错、多种错误类型）下，规则模板的 hint 信息密度不够，模型理解不到要点。

**机制**：在 `wrap_model_call` 内对复杂 loop 调一次 Haiku 生成更精炼的 hint。

```python
self._hint_cache: dict[str, str] = {}    # loop_hash → 已生成的 judge hint

def wrap_model_call(self, request, handler):
    loop_info = self._detect_loop(request.messages)
    if loop_info is None:
        return handler(request)
    
    loop_hash, loop_start, loop_end = loop_info
    
    # 缓存命中:直接用,零额外 LLM 成本
    if loop_hash in self._hint_cache:
        hint = self._hint_cache[loop_hash]
    else:
        if self._should_use_judge(request.messages, loop_start, loop_end):
            hint = call_judge_haiku(request.messages, loop_start, loop_end)
        else:
            hint = build_rule_hint(request.messages, loop_start, loop_end)
        self._hint_cache[loop_hash] = hint
    
    patched = self._build_patched(request.messages, loop_start, loop_end, hint)
    return handler(request.override(messages=patched))
```

**Judge 触发条件**：

```python
def _should_use_judge(messages, loop_start, loop_end) -> bool:
    region = messages[loop_start:loop_end+1]
    distinct_tools = len({tc["name"] for ai in region 
                          for tc in getattr(ai, "tool_calls", []) or []})
    region_size = loop_end - loop_start
    return distinct_tools >= 3 or region_size >= 8
```

**关键澄清**：`_hint_cache` 是**纯优化缓存**，不是语义状态。清空它只导致下次重新调一次 judge，行为完全等价。这跟"性能 cache"是一回事，与 V1 的"无状态"哲学并不冲突。

**成本估算**：典型 oncall thread 50 次 model call、1 个 loop、10 次 patch 周期 → 1 次 Haiku 调用 + 9 次缓存命中。

### V2.5 Tool spam 检测（同 tool 高频）

**问题**：模型用同一个 tool、不同 args 反复尝试，每次产生 unique hash，V1 抓不到。例：

```
read_file("a.py")    → "no bug here"
read_file("b.py")    → "no bug here"
read_file("c.py")    → "no bug here"
... 共 10 个不同文件 ...
```

每条都是 unique hash，V1 永远不触发。但模型在乱翻。

**核心信号**：最近 N 个 tool_call 中**同一 tool_name 占比 ≥ dominance_ratio**，且**总数 ≥ min_total**。

**检测**：

```python
from collections import Counter

def _detect_tool_spam(
    self,
    messages: list,
    window: int = 20,
    min_total: int = 8,
    dominance_ratio: float = 0.8,
) -> tuple[str, int, int] | None:
    """检测某 tool 在最近窗口内压倒性占主导。
    
    返回 (dominant_tool_name, patch_start_idx, patch_end_idx),None 表示未触发。
    """
    # 收集最近 window 个 AIMessage 的位置 + tool_call
    recent_ai_with_idx: list[tuple[int, AIMessage]] = []
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage) and getattr(messages[i], "tool_calls", None):
            recent_ai_with_idx.append((i, messages[i]))
            if len(recent_ai_with_idx) >= window:
                break
    if not recent_ai_with_idx:
        return None
    recent_ai_with_idx.reverse()
    
    # 统计 tool_call 数量(注意单条 AIMessage 可能含多个 tool_call)
    tool_calls_flat: list[tuple[int, str]] = []   # (ai_idx, tool_name)
    for ai_idx, ai in recent_ai_with_idx:
        for tc in ai.tool_calls or []:
            tool_calls_flat.append((ai_idx, tc.get("name", "?")))
    
    if len(tool_calls_flat) < min_total:
        return None
    
    counts = Counter(name for _, name in tool_calls_flat)
    top_tool, top_count = counts.most_common(1)[0]
    if top_count / len(tool_calls_flat) < dominance_ratio:
        return None
    
    # patch 区间:仅覆盖 dominant tool 调用所在的 AIMessage 范围
    dominant_ai_indices = sorted({
        ai_idx for ai_idx, name in tool_calls_flat if name == top_tool
    })
    return top_tool, dominant_ai_indices[0], dominant_ai_indices[-1]
```

**Hint**：

```
[TOOL SPAM] You've called {tool_name} {top_count} times in the last {window}
turns with varying arguments, producing little new information.

The tool isn't converging on an answer. Choose:
  (a) try a different tool
  (b) acknowledge what's not findable with this tool and answer with partial info
  (c) form a specific hypothesis before the next call (don't blind-search)

Recent {tool_name} attempts ({top_count} unique args):
  {samples}    # 最多列 5 个 (args, result_summary)
```

**Patch 区间**：

只切**dominant tool 涉及的 AIMessage 范围**（含中间穿插的其它 AIMessage 一起切）+ 配对 ToolMessage。这样保留区间外的合法消息：

```
patch_start = 第一个 dominant tool 调用的 AIMessage idx
patch_end   = 最后一个 dominant tool 调用的 AIMessage idx + 配对 ToolMessage
patched     = messages[:patch_start] + [HumanMessage(hint)] + messages[patch_end+1:]
```

**与 V1/V2.1 的优先级**：

```
V1 hash → V2.2 periodic → V2.5 tool spam → V2.1 no-progress
```

V2.5 在 V2.1 之前——tool-spam 是更具体的信号，能给模型更精准的 hint（"换 tool"），而 V2.1 的 hint 更通用（"停下来思考"）。能用具体 hint 就不要用通用的。

**边界处理**：

| 场景 | 处理 |
|------|------|
| 合法多文件读取（read_file 5 次, 每次产生 useful 内容） | dominance_ratio=0.8 + min_total=8 共同过滤；若仍误报，调高阈值 |
| Tool 总数 < min_total | 不触发（避免短任务起步阶段误判） |
| 多个 tool 都很高频但都没到 dominance | 不触发 V2.5；V2.1 可能触发兜底 |
| 单条 AIMessage 并发 ≥ 3 个 tool_call | tool_call 计数按"次"算（一条消息可能贡献多次） |

**调参建议**：

| 参数 | 默认 | 调优方向 |
|------|------|---------|
| `window` | 20 | 短任务降到 12；超长任务升到 30 |
| `min_total` | 8 | 防止短窗口误报；任务粒度细可降到 5 |
| `dominance_ratio` | 0.8 | 误报多→升到 0.9；漏报多→降到 0.7 |

### V2.4 检测器并联

V2 的 wrap_model_call 升级为多检测器并联，**优先级递减**：

```python
def wrap_model_call(self, request, handler):
    messages = request.messages
    
    detection = (
        self._detect_v1_hash(messages)         # 最确定:同 hash 重复
        or self._detect_periodic(messages)     # V2.2:周期性 ping-pong
        or self._detect_tool_spam(messages)    # V2.5:同 tool 高频
        or self._detect_no_progress(messages)  # V2.1:全局无进展(兜底)
    )
    
    if detection is None:
        return handler(request)
    
    patched = self._build_patch(messages, detection)
    return handler(request.override(messages=patched))
```

**优先级理由**：

| 优先级 | Detector | 理由 |
|-------|----------|------|
| 1 | V1 hash | 误报率近 0,确凿无疑 |
| 2 | V2.2 periodic | 周期性是强信号,误报极低 |
| 3 | V2.5 tool spam | 中等确定性,但 hint 最具体（换 tool） |
| 4 | V2.1 no-progress | 兜底,信号最弱、hint 最通用 |

**核心思想**：能用具体 detector 的 hint 就不要用通用的——具体 hint 能给模型更明确的下一步方向。

不同 detection 类型 → 不同 hint 模板 + 不同 patch 区间。**所有 detector 正交扩展**，patching 引擎不变。

---

## 附录 A：方案演化历程

记录从概念到最终架构的关键决策点，便于未来读者理解"为什么不那样做"。

### A.1 起点：Thariq 的 rewind 直觉

Thariq（Anthropic）在 _Using Claude Code: Session Management & 1M Context_ 一文中提出：

> If I had to pick one habit that signals good context management, it's rewind. ... Rewind is often the better approach to correction.

人类用 `Esc Esc` 在 Claude Code 里手动 rewind。问题是：能否在 agent 运行时**自动**做这件事？

DeerFlow 已有 `scripts/trace_replay.py` 提供离线 rewind（同 thread_id + checkpoint_id 重新跑），但只供人用——agent 自己出现 loop 时不会主动 rewind。

### A.2 第一版尝试：after_model + RemoveMessage（被推翻）

**思路**：在 `after_model` hook 里检测到 loop，返回 `{"messages": [RemoveMessage(id=...) × N, HumanMessage(hint)]}`，让 `add_messages` reducer 删掉污染历史 + 注入 hint。

**致命问题**：ReAct 路由依赖 `state["messages"][-1]` 是否为带 tool_calls 的 AIMessage。删除最后一条 AIMessage 后，`messages[-1]` 变成 HumanMessage(hint) → 条件边判定"无 tool_calls" → **路由直接到 END，本轮 turn 结束**。

agent 不会自己接着用 hint 继续——必须等用户/harness 发新消息才能重启。这违背了"自动化"目标。

### A.3 第二版尝试：双阶段 arm/fire（被推翻）

**思路**：拆成两步：

- `after_model`：只设标记 `self._pending_rewind[thread_id] = call_hash`，不动 state；让本轮 tool 正常执行
- `before_model`：下一轮 model 前检查标记，执行 RemoveMessage + HumanMessage 注入

这样路由不被打断（after_model 阶段 state 还是合法的"AIMessage with tool_calls"），但 state 在 before_model 时被清洗。

**问题**：

1. 引入 per-thread 状态字典 `_pending_rewind`，违背"无状态"理想
2. 需要**多套保护**：`max_rewinds_per_thread` 防无限 rewind、harness 端配合 `RESUME_SENTINEL` + auto-resume 防 turn 中断
3. 整体协议复杂——哨兵字符串、auto-resume 计数、清哨兵的时机都要小心
4. 仍然**真删 state**：trace_replay 看不到 loop 历史；其他 middleware 若依赖完整历史会受影响

虽然能跑，但复杂度爆炸。

### A.4 关键发现：DanglingToolCallMiddleware 启发

调研用户 Ctrl+C 后 messages 里留下什么时，发现 `dangling_tool_call_middleware.py` 用了 `wrap_model_call` hook：

```python
def wrap_model_call(self, request, handler):
    patched = self._build_patched_messages(request.messages)
    if patched is not None:
        request = request.override(messages=patched)
    return handler(request)
```

它**不修改 state**，只修改 `request.messages`——给模型看的视图。state 永远是真实的"中断历史"，但模型每次被调用前看到的是"已自愈的虚拟版"。

这个 pattern 直接套到 rewind 场景：

- 不需要 RemoveMessage（不动 state）
- 不需要哨兵（不结束 turn）
- 不需要 auto-resume（model 调用后 ReAct 自然继续）
- 不需要 arm/fire 两阶段（单步搞定）
- 不需要 per-thread 状态字典（每次重扫即可）

**所有复杂度一次性消除**。

### A.5 终态：wrap_model_call 视图层 patching

最终设计的核心三句话：

1. **检测在 `wrap_model_call` 里做**，永远扫 `request.messages` 现场判断
2. **patching 是视图操作**，`request.override(messages=patched)` 不进 state
3. **永远 patch**，loop hash 一旦在历史里就持续 patch out（幂等无成本）

附带收益：

- 中间件变**纯函数**——同 input 永远同 output
- 兼容多 worker / 进程重启 / trace_replay 复跑
- V2 扩展点干净——加新检测器只需在 detect 阶段加 `or` 分支，patching 引擎不变
- Judge-based hint 仍可做，只需加一个**纯优化缓存** `_hint_cache: dict[loop_hash, str]`，不破坏无状态哲学

### A.6 决策对照表

| 决策点 | 推翻方案 | 终态选择 | 关键理由 |
|--------|---------|---------|---------|
| Hook 位置 | `after_model` / `before_model` | `wrap_model_call` | 不动 state、不影响路由 |
| State 处理 | RemoveMessage 删消息 | `request.override` 改视图 | 保留真实历史 |
| 重复 patch | 第一次 patch 后状态记忆 | 永远重 patch | 无状态、幂等 |
| Turn 中断 | 改路由 + 哨兵 + auto-resume | turn 不中断，自然继续 | 简洁，零 harness 改造 |
| 三层结构 | warn/rewind/hard_stop 全保留 | warn 删除，rewind 取代 | warn 被 patching 完全 superseded |
