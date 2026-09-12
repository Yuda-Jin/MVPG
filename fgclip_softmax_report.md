# softmax 放大验证报告

- 数据：`/root/autodl-tmp/data/SA1B_5_threshold`（同 12 组主图）；模型 FG-CLIP2-base（CPU）

- 方法：每组把 base+real+hall 短语对图像算 `max_patch_sim`（box/short 头），再组内 softmax(s/τ)，比较 hall 与 real 短语的概率分布。


## 结果（Cohen's d 越大越好，目标 d≥1.0）

| 信号 | hall prob | real prob | Cohen's d | 组级 margin>0 |
|---|---|---|---|---|
| raw s (box，无softmax) | 0.2320 | 0.2455 | **+0.75** | 10/12 |
| raw s (short，无softmax) | 0.2054 | 0.2212 | **+0.56** | 10/12 |
| softmax box τ=1.0 | 0.1276 | 0.1328 | **+0.28** | 10/12 |
| softmax box τ=0.5 | 0.1266 | 0.1337 | **+0.38** | 10/12 |
| softmax box τ=0.2 | 0.1235 | 0.1364 | **+0.61** | 10/12 |
| softmax box τ=0.1 | 0.1183 | 0.1412 | **+0.83** | 10/12 |
| softmax box τ=0.05 | 0.1079 | 0.1511 | **+0.96** | 10/12 |
| softmax short τ=1.0 | 0.1272 | 0.1327 | **+0.30** | 10/12 |
| softmax short τ=0.5 | 0.1257 | 0.1335 | **+0.40** | 10/12 |
| softmax short τ=0.2 | 0.1213 | 0.1359 | **+0.60** | 10/12 |
| softmax short τ=0.1 | 0.1139 | 0.1403 | **+0.71** | 10/12 |
| softmax short τ=0.05 | 0.0994 | 0.1496 | **+0.78** | 10/12 |

## 结论

- 最强 softmax 组合 **softmax_box_tau0.05**，Cohen's d=+0.96，仍不达稳定(d<1.0)。

- 原始相似度 d_box=+0.75、d_short=+0.56，作为 softmax 放大前的基线。

