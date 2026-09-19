# RCGR 的 FG-CLIP 口径对照验证
- 图像目录：`/root/autodl-tmp/data/SA1B_5_threshold`；组数：256；文本来源：`llava-v1.5-7b`
- caption token 长度：p50=105，p90=116，p99=120，max=123；超过 62 token 的比例 = 0.969。short/box 的位置表宽只有 64，故当前 `short_L64` 口径下这 97% 的 caption 被截断。
- 数据：[区域子图 K 张] × [与该区域逐行对齐的 caption K 条]，K 由网格 r{r}c{c} 决定。
- `margin = mean_i ( sim(c_i,R_i) - max_{j!=i} sim(c_j,R_i) )` 即 RCGR 的竞争奖励均值；`margin>0` 为「本区域描述在组内竞争中胜出」的比例（随机基线 = 1/K ≈ 0.25）。
- `col_acc`：区域 -> 描述 的 top-1 命中率（`argmax_j S[j,i] == i`）；`row_acc`：描述 -> 区域 的 top-1 命中率。
- `跨组diag`：用**下一组**的 caption 配本组区域的自匹配均值（越低 = 编码器整体分辨力越强，对 dense 口径它与 `diag` 的差距即「分辨余量」）。

| 口径 | diag | offmax | margin | margin>0 | col_acc | row_acc | 组内std | 跨组diag |
|---|---|---|---|---|---|---|---|---|
| short_L64_global | +0.1482 | +0.1173 | +0.0309 | 0.731 | 0.731 | 0.818 | 0.0445 | -0.0286 |
| box_L64_global | +0.1879 | +0.1523 | +0.0356 | 0.717 | 0.717 | 0.801 | 0.0559 | -0.0290 |
| long_L196_global | +0.1313 | +0.0880 | +0.0433 | 0.776 | 0.776 | 0.839 | 0.0499 | -0.0819 |
| short_L64_dense | +0.1857 | +0.1820 | +0.0037 | 0.539 | 0.539 | 0.595 | 0.0293 | +0.1082 |
| box_L64_dense | +0.3075 | +0.3062 | +0.0013 | 0.510 | 0.510 | 0.579 | 0.0357 | +0.2148 |
| long_L196_dense | +0.1995 | +0.1902 | +0.0093 | 0.598 | 0.598 | 0.640 | 0.0310 | +0.1044 |

## 判读
- `margin` 越正越大、`margin>0`/`col_acc` 越高于随机基线，该口径的区域竞争信号越可用。
- `组内std` 是 GRPO advantage 的分子：接近 0 说明组内无差异，训练拿不到梯度。
- 同一图像口径下 `short_L64` vs `long_L196` 的差值 = 「caption 被截断到 64 token」的代价（注意 walk_type 也换了，二者捆绑在 FG-CLIP2 的长度档里）。
- 同一 walk_type 下 `*_global` vs `*_dense` 的差值 = 图像特征口径的代价。

## 结论
- 当前训练口径 `short_L64_global`：col_acc = **0.731**（随机基线 0.25）、margin = +0.0309、组内 std = 0.0445 -> 竞争信号真实存在，**RCGR 这套方法本身可用**。
- 全因子最优口径是 **`long_L196_global`**（col_acc = 0.776）。
- 图像口径 `global` 在 3 种 walk_type 上的平均 col_acc = 0.742（平均跨组diag = -0.0465）。
- 图像口径 `dense` 在 3 种 walk_type 上的平均 col_acc = 0.549（平均跨组diag = +0.1425）。
- 文本档 `short`（L64）在 2 种图像口径上的平均 col_acc = 0.635。
- 文本档 `box`（L64）在 2 种图像口径上的平均 col_acc = 0.613。
- 文本档 `long`（L196）在 2 种图像口径上的平均 col_acc = 0.687。
