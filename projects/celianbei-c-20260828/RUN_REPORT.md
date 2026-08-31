# 2026 策联杯 C 题首轮运行报告

## 状态

READY（首轮可复现交付已形成）。输入目录保持只读；未读取 `D:\modeling\competition`、本项目其他 `projects/`、`docs/history/` 或旧参考HTML。

## 实际运行

- 主命令：`python code/solve.py --input-dir input --output-dir .`
- 随机种子：`20260828`；Python 3.12；依赖见 `requirements.txt`。
- 输入：官方Excel `C题_数据附件.xlsx`，SHA-256 `23eecb09565bb064da8e4753933b88fe9d882d3b92f0c23574bb2aebdc530907`。
- 按需 Prompt：未读取/未使用（数据审计、模型构造、比较和交付检查均由主 Agent 直接完成）。

## 关键结果

### 问题一

历史数据700场，训练560、测试140。时间后20%验证：ExtraTrees MSE 199.2648、MAE 11.4320、R² 0.6751（单位为百万人及其平方），优于随机森林和梯度提升；已用全训练集生成140条测试预测和72场小组赛预测。测试真实标签未公开，未伪造官方MSE。

### 问题二

启发式时段—场馆搜索目标值 `21.5820668729`。审计：72场且比赛ID唯一；场馆—UTC冲突0；最短球队休息63 h（≥60）；每日场馆超限0；场馆总量上下限违约0；黄金时段次数极差2；第三轮每日高安保容量违约0。该值是可行解，不是全局最优性证明。

### 问题三

按题设公式进行20,000次泊松蒙特卡洛和晋级排序。动态总净效益7.883197，静态7.967812，平均逐场改善率1.446%。容量修复后每个参考日的高转播、高安保、强化交通计数和资源预算均不超过 `dynamic_resource_limits`；明细在 `audit/q3_resource_checks.json`。动态并非每场优于静态，已在论文中保留该结果。

### 问题四

公开来源抓取2022世界杯小组赛48场：<https://en.wikipedia.org/wiki/2022_FIFA_World_Cup>，采集日期2026-08-28。实际赛程平均/最短休息96.75/96 h、场馆利用率6场/馆；优化方案143.125/63 h、4.5场/馆。实际开球时间和容量存在页面缺失，采用代理值并在论文和审计中明确标注。

## 主要工件

- 论文PDF：[paper/celianbei_C_参赛论文.pdf](paper/celianbei_C_参赛论文.pdf)；Markdown：[paper/论文初稿.md](paper/论文初稿.md)。
- 结果CSV：`results/result_1_test_prediction.csv`（140行）、`result_1_match_prediction.csv`（72行）、`result_2_group_schedule.csv`（72行）、`result_3_dynamic_strategy.csv`（24行）、`result_4_schedule_comparison.csv`（5项）。
- 实际赛程：`results/actual_schedule.csv`（48行，含来源URL和采集日期）。
- 代码与复现：`code/solve.py`、`README.md`、`requirements.txt`。
- 审计：`audit/run_metadata.json`、`audit/audit_summary.json`、`audit/q3_team_states.csv`、`audit/q3_resource_checks.json`、`audit/failure_notes.md`、`audit/q4_source_note.md`。
- 图表：`figures/q1_model_mse.png`、`q3_static_dynamic.png`、`q4_comparison.png`。
- AI声明：[paper/AI工具使用详情.pdf](paper/AI工具使用详情.pdf)；支撑材料压缩包：`支撑材料.zip`（约220 KB）。

## 失败、边界与风险

- 初始时段贪心曾因窗口过晚无解，已通过轮次日期窗口修复；初始场馆同UTC冲突6处，已加入唯一性约束并复核为0。
- 赛程搜索是启发式，未给出MILP/CP-SAT最优性界；目标值不可与未知最优值比较。
- 观众模型只在时间切分验证，隐藏测试标签不可见；小组赛预测存在外推误差。
- 晋级平局用可复现随机键实现等比例近似；第三轮不使用其他第三轮结果，符合题设同一信息截面。
- 问题四现实开球时间、场馆容量含代理，不据此宣称历史事实优劣。

## 下一项最有价值行动

若获得组委会测试评分或官方实际赛程接口，优先做观众模型校准和现实字段替换；随后用CP-SAT/MILP对缩小实例给出启发式目标的最优性界。当前首轮交付不因该外部资源缺失而阻塞。

