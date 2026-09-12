# caption 内均值 baseline：子句级 reward 符号判别验证

- 信号：联合 `.,;` 分割 → 子句 box 头 × dense patch max_patch_sim（raw）。

- baseline：`reward = s_box − 同 caption 子句均值`（句内中心化）。

- 判定：幻觉子句 reward<0 视为「正确抑制」，真实子句 reward>0 视为「正确鼓励」。


## 结果

- 幻觉子句 reward 均值：**-0.01834**（应为负）

- 真实子句 reward 均值：**+0.00458**（应为正）

- reward 上 Cohen's d：**+1.20**

- 组级 margin>0：**20/20**


## 符号判别准确率

| 范围 | 抑制正确率(幻觉<0) | 鼓励正确率(真实>0) | 整体 |
|---|---|---|---|
| 全部子句 | 89.3% | 53.7% | 60.8% |
| 仅含幻觉子句的 caption | 89.3% | - | 62.2% |
