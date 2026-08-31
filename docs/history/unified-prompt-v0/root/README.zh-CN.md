# Prompt-Native 数学建模 v2

这是一个直接给普通 Codex 使用的数学建模项目。当前 Chat 就是完整研究者；项目不创建永久中控，不强制启动第二个研究 Chat，不设置阶段、角色链、Lens Router、状态标签、固定 trackers 或共享可写研究内存。

## 使用

第一次使用时，把下面目录作为独立本地项目添加到 Codex：

```text
D:\prompt-native-math-modeling\prompt-native-v2
```

然后在该项目中直接发送题目、附件、数据和交付要求，并说：

```text
使用这个项目完成这道数学建模任务。
```

Codex 会自动读取根 `AGENTS.md`，再读取统一的 `PROMPT.zh-CN.md`，并在当前 Chat 中直接开始研究。无需复制其他启动 Prompt。

## 真实运行面

```text
AGENTS.md
  自动激活、权限与证据底线、原生能力边界

PROMPT.zh-CN.md
  完整数学建模方法学；描述目标和认识论，不规定流程

work/
  仅在需要实际文件时，由 Codex 自主建立任务工件
```

完整架构总览位于 `prompt-native-math-modeling.html`，其中逐字展示两份运行文本、第一性原理设计、v1 → v2 删除矩阵和冻结实验。`docs/`、README 和 HTML 都是设计与评估材料，不自动进入研究上下文。

## 原生能力

Codex 自主决定是否联网、发现领域工具、写代码、运行实验、使用子 Agent、寻找反例、独立重做或改变模型。子 Agent 是局部工具，不是永久角色；额外的用户可见任务只在用户明确请求时创建。

## 与 v1 的区别

归档版本位于 `..\thin-connection-v1\`。v2 删除了：

- Controller 与强制 R0；
- create/wait/read/send 驱动协议；
- 七类事件路由和独立 Lens 文件；
- Task/Thread Brief；
- current、claims、decisions、tools 等固定 trackers；
- `COMPLETE / NEEDS_USER / PAUSED_BY_USER / HOST_YIELD` 状态接口。

朋友设计中有价值的模型创造、模型—数据共演化、信息增益、外部发现、反例、模型比较和论文交付思想，已经作为统一 Prompt 中的可选认知能力保留，由 Codex 自主调用，而不是由外部中控调度。

## 证据边界

这是一个更彻底的 Prompt-Native 假设，不是已证明更优的架构。它减少上下文和维护负债，但也可能降低恢复保真度、遗漏独立复核或让 Codex 过早收敛。是否值得采用必须通过冻结未见题与 v1、裸 Codex 做同条件比较。
