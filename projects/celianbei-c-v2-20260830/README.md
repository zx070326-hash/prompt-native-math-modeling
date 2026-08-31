# 2026 “策联杯” C 题候选交付

本目录是第二轮冻结实验的独立候选成果。`input/` 为官方只读输入；所有模型、结果、论文和审计均由本目录内代码从官方输入重新产生。

## 一键复现

在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python src\run_all.py
python src\make_figures.py
python src\validate_outputs.py
```

构建 PDF 与支撑包：

```powershell
Push-Location support
xelatex -interaction=nonstopmode -halt-on-error '-output-directory=..\outputs\pdf' "AI 工具使用详情.tex"
Pop-Location
Push-Location paper
xelatex -interaction=nonstopmode -halt-on-error '-jobname=000_参赛论文' '-output-directory=..\outputs\pdf' main.tex
xelatex -interaction=nonstopmode -halt-on-error '-jobname=000_参赛论文' '-output-directory=..\outputs\pdf' main.tex
Pop-Location
python src\package_support.py
```

`package_support.py` 为避免覆盖既有候选包而默认拒绝同名输出；需要重建时请先将旧包和 `tmp/support_package` 移出项目，再运行。脚本先在 `tmp` 生成临时 ZIP，完成 CRC 与成员清单核验后才原子落到正式文件名。

固定随机种子为 `20260830`。问题一 OOF 上的集成权重拟合与误差评价不是嵌套外层验证。问题二使用 SciPy/HiGHS MILP，默认设置 300 秒有界求解；最优性缺口只针对代码中披露的轮次日期预筛候选域，不覆盖未预筛问题。问题三两个资源分配 MILP 均达到 gap=0，其改善只是在给定泊松、反馈与效用模型内的信息价值。问题四的加权总分是按官方 Q2 八项权重和共同边界得到的代理比较，不是对真实赛事的因果判断。

## 主要输出

- `outputs/result_1_test_prediction.csv`：140 条隐藏测试集预测，单位百万人。
- `outputs/result_1_match_prediction.csv`：72 场模拟小组赛预测，单位人。
- `outputs/result_2_group_schedule.csv`：72 场场馆与时段方案。
- `outputs/result_3_dynamic_strategy.csv`：24 场第三轮动态资源方案。
- `outputs/actual_schedule_2022.csv`：FIFA 官方接口整理的 2022 年世界杯小组赛。
- `outputs/result_4_schedule_comparison.csv`：实际赛程与优化赛程比较。
- `logs/q4_weighted_audit.json`：问题四 Q2 八项同定义、共同标准化边界、官方权重总分及 81 组代理敏感性审计；11 项 CSV 继续作为补充诊断。
- `outputs/pdf/000_参赛论文.pdf`：匿名候选论文；`000` 是待替换队号。
- `outputs/pdf/AI 工具使用详情.pdf`：AI 使用详情候选稿。

## 数据来源与边界

- 官方题面、工作簿和模板：`input/`。
- 2022 年世界杯赛程、比分、场馆与上座：FIFA API，访问日期 2026-08-30。
- 场馆坐标：OpenStreetMap Nominatim，访问日期 2026-08-30，原始响应保存在 `data/external/`。
- 隐藏测试标签不可见；论文只报告时间阻断交叉验证结果，不把内部验证写成官方测试成绩。
- 问题四实际场馆容量以整届赛事 FIFA 官方记录的该场馆最大上座作为代理；实际气候和安保能力缺失时固定为候选场馆中位数，只比较上座驱动的风险部分。
- 问题四 U/H/C/D/R 的跨赛事缺口分别使用 Elo 概率、官方 72 场吸引力桥接回归、候选场馆成本中位数、相邻场馆旅行时间和风险中位数代理；全部代理及敏感性范围均写入上述审计 JSON。

## 审计入口

`logs/mechanical_validation.json` 包含全部机械检查与六个 CSV 的 SHA-256。`RUN_REPORT.md` 记录候选交付边界、未决风险与建议独立审查工件。
