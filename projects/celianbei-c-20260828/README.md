# 2026 策联杯 C 题首轮复现说明

输入仅使用 `input/` 中官方题面、Excel 和模板。运行环境 Python 3.12；依赖见 `requirements.txt`。

```powershell
cd D:\prompt-native-math-modeling\prompt-native-v2\projects\celianbei-c-20260828
python -m pip install -r requirements.txt
python code\solve.py --input-dir input --output-dir .
python code\validate_outputs.py
python paper\make_pdf.py
python paper\make_ai_pdf.py
```

随机种子固定为 `20260828`。主脚本输出 `results/`、`audit/` 和 `figures/`。`audit/run_metadata.json` 保存输入Excel SHA-256；`audit/audit_summary.json` 保存预测评估、赛程约束、仿真和现实赛程来源。Windows 字体 `C:\Windows\Fonts\simhei.ttf` 用于中文PDF；无该字体时可替换为同等中文TTF。

主张边界：问题二为启发式可行解，不提供全局最优性证明；问题四的公开网页赛程中开球时间和容量含代理值，已在论文中标注。测试集真实观众人数由组委会保留，无法本地计算官方MSE。
