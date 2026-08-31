# Prompt-Native 数学建模：原生 Codex 运行版

本目录以朋友参考 HTML 的第一性原理和 Prompt 分层为地基：当前 Chat 是完整研究者，短研究语境常驻，专业认知 Prompt 只在当前缺口需要时读取或发送。它不把所有能力挤进一份统一长 Prompt，也不把原生 Codex 再包装成 Controller、固定角色群或状态机。

Codex 本身已经能够重构问题、联网、使用工具、写代码、委派和审查。额外 Prompt 不会增加一种新的基础智力，也不能保证正确；它的作用是在长程研究出现可观察失焦时，临时提高某个问题的显著性。若 Codex 已经自然处理该问题，就不加载对应 Prompt。

## 使用

把下面目录作为独立 Codex 项目打开：

```text
D:\prompt-native-math-modeling\prompt-native-v2
```

直接发送题目、附件、数据、预算和交付要求，并说：

```text
使用这个项目完成这道数学建模任务。
```

Codex 自动读取根 `AGENTS.md`。当前 Chat 成为主 Agent，基础只读取无人值守启动、主 Agent 和按需索引；专业 Prompt 只有出现相应研究缺口、且 Codex 尚未自行处理时才进入上下文。普通联网、工具、代码和文件操作不需要 Prompt 审批。

## 长程任务的审查门

按需 Lens 分为两类：探索型 Lens 由 Codex 根据研究缺口自主选择；承诺型 Lens 只在准备冻结高影响主张、出现真实不变量异常、正式交付物已经生成或发生真实跨 Chat 恢复时进入准入判断。详细规则见 `contracts/REVIEW_GATE.zh-CN.md`。

这不是固定阶段，也不是后台 Router。`AGENTS.md` 只提供短政策；Codex 在观察到事件后先做机械检查，再决定是否复制 `templates/REVIEW_ENVELOPE.zh-CN.md` 到任务的 `audit/reviews/` 并发给独立 Chat。机械检查足够、影响较低、不会改变判断或无法形成有界审查时，明确跳过，不启动新 Chat。

默认每个持久项目最多一次核心主张审查和一次正式交付审查；异常优先并入核心主张审查。默认总计不超过两个独立审查上下文。审查信封和返回结果是一次审查的证据工件，不是第二套项目状态或共享可写内存。

## 文件结构

```text
AGENTS.md
prompts/
├── 00-UNATTENDED_BOOTSTRAP.zh-CN.md
├── 01-MAIN_AGENT.zh-CN.md
├── 02-AUTONOMOUS_RESEARCHER.zh-CN.md
├── INDEX.zh-CN.md
└── on-demand/
    ├── MODEL_CREATION.zh-CN.md
    ├── MODEL_DATA_COEVOLUTION.zh-CN.md
    ├── INFORMATION_GAIN_ACTION.zh-CN.md
    ├── ADVERSARIAL_RESEARCHER.zh-CN.md
    ├── REFLECTION_MIRROR.zh-CN.md
    ├── EXTERNAL_DISCOVERY.zh-CN.md
    ├── MODEL_COMPARISON.zh-CN.md
    ├── CLAIM_CHECK.zh-CN.md
    ├── DELIVERY_REVIEW.zh-CN.md
    └── HANDOFF_RESUME.zh-CN.md
context/RESEARCH_CONTEXT.zh-CN.md
contracts/COMPLETION_CONTRACT.zh-CN.md
contracts/REVIEW_GATE.zh-CN.md
templates/
└── REVIEW_ENVELOPE.zh-CN.md
projects/
```

## 运行边界

朋友版本提供方法学地基：研究主持语境、模型创造、模型与数据共演化、信息增益、反方研究、反思镜面、工具使用、模型比较和完成判断。运行版保留这些结构，并做四类窄调整：

- 不假定 Prompt 自动赋予浏览器、权限、长期记忆或后台执行，只使用宿主实际提供的能力；
- 允许重构技术问题和模型，但不得静默改变用户目标、价值、权限或交付用途；
- 增加高影响主张检查、正式交付审查和真实中断恢复三份按需 Prompt，并用有预算的审查门决定是否值得发送独立上下文；
- 把完成契约缩为任务相关的收敛判断，不强制每题制造多模型、实验、论文或状态 JSON。

`context/RESEARCH_CONTEXT.zh-CN.md` 和 `templates/` 是可选视图与参考，不是活动状态系统。真实原题、来源、方程、代码、输入、参数、结果和最终交付才是研究工件。

参考架构原文完整复制在 `prompt-native-math-modeling-architecture.html`，不会随运行版改写。朋友原始 `MAIN_AGENT` 和 `TOOL_USE` 保存在 `docs/history/friend-reference-prompts/`；上一版统一 Prompt 设计保存在 `docs/history/unified-prompt-v0/`。这些历史文件不会进入当前任务上下文。当前完整设计展示在 `prompt-native-math-modeling.html`。
