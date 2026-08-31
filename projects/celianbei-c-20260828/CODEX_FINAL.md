已完成首轮可复现交付，全部产物位于：

`D:\prompt-native-math-modeling\prompt-native-v2\projects\celianbei-c-20260828\`

关键结果：

- 问题一：ExtraTrees 时间后20%验证 MSE `199.2648`、MAE `11.4320`、R² `0.6751`；生成140条测试预测和72场小组赛预测。
- 问题二：赛程目标值 `21.5820668729`；72场、场馆-UTC冲突0、最短休息63小时、场馆容量/总量约束通过、黄金时段次数极差2。
- 问题三：20,000次泊松晋级仿真；动态总净效益 `7.883197`，静态 `7.967812`，平均逐场改善率 `1.446%`。动态方案容量约束逐日通过，但并非每场优于静态。
- 问题四：抓取2022世界杯48场小组赛，现实开球时间和容量代理值已在论文及来源说明中明确标注。

主要文件：

- [论文PDF](D:/prompt-native-math-modeling/prompt-native-v2/projects/celianbei-c-20260828/paper/celianbei_C_参赛论文.pdf)
- [运行报告](D:/prompt-native-math-modeling/prompt-native-v2/projects/celianbei-c-20260828/RUN_REPORT.md)
- [主求解代码](D:/prompt-native-math-modeling/prompt-native-v2/projects/celianbei-c-20260828/code/solve.py)
- [结果目录](D:/prompt-native-math-modeling/prompt-native-v2/projects/celianbei-c-20260828/results)
- [审计目录](D:/prompt-native-math-modeling/prompt-native-v2/projects/celianbei-c-20260828/audit)
- [支撑材料压缩包](D:/prompt-native-math-modeling/prompt-native-v2/projects/celianbei-c-20260828/支撑材料.zip)

复现命令：

```powershell
cd D:\prompt-native-math-modeling\prompt-native-v2\projects\celianbei-c-20260828
python -m pip install -r requirements.txt
python code\solve.py --input-dir input --output-dir .
python code\validate_outputs.py
python paper\make_pdf.py
python paper\make_ai_pdf.py
```

最终硬约束检查已通过；未使用任何按需 Prompt。重要边界、失败记录、外部来源、输入哈希和后续最有价值行动均已写入 `RUN_REPORT.md`。