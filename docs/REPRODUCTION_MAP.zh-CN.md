# 参考设计复现映射

参考文件：

```text
C:\Users\charles\Downloads\prompt-native-math-modeling-architecture.html
SHA-256: 13701DF1836EED5AB5426A0C504FD78D558284DC15B68647DDF401D9FD05BFD6
```

本目录中的 `prompt-native-math-modeling-architecture.html` 是逐字节副本，作为不会被运行版覆盖的参考地基。下面记录参考章节和当前运行文件的来源关系；朋友原文保留在历史目录，运行文件允许做明确列出的 Prompt-Native 调整。

| 原 HTML 章节 | 复现文件 |
|---|---|
| 八、无人值守启动 Prompt | 运行版修正能力声明、用户目标边界和任务相关交付；参考原文仍在 HTML |
| 九、主 Agent Prompt | 原文保存在 `docs/history/friend-reference-prompts/01-MAIN_AGENT.zh-CN.md`；运行版增加主动联网、原生能力信任和目标边界 |
| 十、自主研究者 Prompt | 运行版修正“修改目标”和预设权限；参考原文仍在 HTML |
| 十一至十五、十七、六类专业 Prompt | `prompts/on-demand/` 原文复现 |
| 十六、工具使用 Prompt | 原文保存在 `docs/history/friend-reference-prompts/TOOL_USE.zh-CN.md`；运行版替换为 `prompts/on-demand/EXTERNAL_DISCOVERY.zh-CN.md` |
| 七、共享研究语境 | `context/RESEARCH_CONTEXT.zh-CN.md` |
| 十九、完成契约 | 运行版改为任务相关的五问收敛判断，不强制通用清单 |
| 二十一、最终输出结构 | `templates/PROJECT_LAYOUT.txt` |
| 二十二、完成状态示例 | `templates/completion.json` |

当前新增或调整内容：

- `AGENTS.md`：自动激活基础 Prompt，规定原生能力优先、按需加载、材料和用户目标边界；
- `prompts/INDEX.zh-CN.md`：把章节名称转化为最小触发索引，并明确“已自行处理则不加载”；
- `prompts/01-MAIN_AGENT.zh-CN.md`：增加主动外部发现、委派边界和原生能力信任；
- `prompts/on-demand/EXTERNAL_DISCOVERY.zh-CN.md`：只处理联网、来源核验、领域工具发现和跨领域迁移，不审批普通工具操作；
- `prompts/on-demand/CLAIM_CHECK.zh-CN.md`：按主张类型选择机械检查或隔离上下文挑战；
- `prompts/on-demand/DELIVERY_REVIEW.zh-CN.md`：正式交付时做逐问逆向追溯；
- `prompts/on-demand/HANDOFF_RESUME.zh-CN.md`：只在真实中断和交接时保存最小恢复信息；
- `contracts/COMPLETION_CONTRACT.zh-CN.md`：把完成判断收缩到用户任务、相称证据、复现、边界和下一行动价值；
- `README.zh-CN.md`：使用说明。

这些调整只改变触发概率和注意分配，不声称给 Codex 增加了新的基础认知能力，也不声称已经提高建模质量。需要在冻结未见题上分别消融 Prompt，观察问题表征、正确性、有效工具发现、失败利用、主张可追溯、交付效用、恢复正确率以及 token/时间成本。
