# 反事实幻觉敏感度（Δ）信号验证报告

- 数据：`/root/autodl-tmp/data/SA1B_5_threshold`（上一轮同 12 组主图）

- 模型：FG-CLIP2-base（`/root/autodl-tmp/cache/fgclip2-base-patch16`），CPU 推理

- 方法：对每个实体短语构造 `a photo of {base} with {e}` 与 `a photo of {base}`，算对齐分差 `Δ = align(present) − align(absent)`。真实实体 Δ>0（贡献对齐），幻觉实体 Δ≈0（无贡献）。

- align 度量：patch-short / patch-box（dense patch 的 max_patch_sim）、global-short（整图余弦）。


## 结论

- **判定：patch-short 反事实差分信号尚不能提供稳定幻觉信号**（Cohen's d=+0.47，准确率=62.5%）。


## 三种 align 度量对比

### patch_short

- 真实实体 Δ 均值 **-0.0239**（σ=0.0150），幻觉实体 Δ 均值 **-0.0299**（σ=0.0096）。

- Cohen's d **+0.47**；margin(real−hall) 均值 **+0.0070**，margin>0 的组 **11/12**；短语级分类正确率 **62.5%**。

### patch_box

- 真实实体 Δ 均值 **-0.0158**（σ=0.0088），幻觉实体 Δ 均值 **-0.0194**（σ=0.0070）。

- Cohen's d **+0.44**；margin(real−hall) 均值 **+0.0040**，margin>0 的组 **9/12**；短语级分类正确率 **56.2%**。

### global_short

- 真实实体 Δ 均值 **+0.0020**（σ=0.0094），幻觉实体 Δ 均值 **-0.0034**（σ=0.0067）。

- Cohen's d **+0.66**；margin(real−hall) 均值 **+0.0057**，margin>0 的组 **11/12**；短语级分类正确率 **63.7%**。


## 分组结果（patch-short 的 Δ：hall vs real）

| gid | 幻觉 Δ（min/mean/max） | 真实 Δ（min/mean/max） | margin(real−hall) |
|---|---|---|---|
| sa_1000 | -0.0360 / -0.0271 / -0.0202 | -0.0268 / -0.0164 / -0.0074 | +0.0107 |
| sa_183 | -0.0420 / -0.0420 / -0.0420 | -0.0478 / -0.0398 / -0.0314 | +0.0022 |
| sa_2646 | -0.0503 / -0.0364 / -0.0285 | -0.0358 / -0.0236 / -0.0106 | +0.0129 |
| sa_3439 | -0.0440 / -0.0324 / -0.0169 | -0.0335 / -0.0258 / -0.0171 | +0.0067 |
| sa_4286 | -0.0249 / -0.0176 / -0.0134 | -0.0167 / -0.0059 / +0.0050 | +0.0117 |
| sa_5122 | -0.0551 / -0.0370 / -0.0200 | -0.0579 / -0.0420 / -0.0268 | -0.0050 |
| sa_5954 | -0.0271 / -0.0209 / -0.0106 | -0.0364 / -0.0209 / -0.0053 | +0.0001 |
| sa_6749 | -0.0379 / -0.0353 / -0.0309 | -0.0392 / -0.0258 / -0.0184 | +0.0094 |
| sa_7547 | -0.0327 / -0.0269 / -0.0211 | -0.0442 / -0.0248 / -0.0002 | +0.0021 |
| sa_8367 | -0.0397 / -0.0276 / -0.0196 | -0.0315 / -0.0250 / -0.0185 | +0.0026 |
| sa_9166 | -0.0400 / -0.0331 / -0.0273 | -0.0154 / -0.0124 / -0.0088 | +0.0208 |
| sa_9998 | -0.0344 / -0.0317 / -0.0303 | -0.0499 / -0.0220 / +0.0042 | +0.0097 |

## 逐实体明细（Δ = align(present) − align(absent)）

| gid | 类别 | 实体 | Δ_patch_short | Δ_patch_box | Δ_global_short |
|---|---|---|---|---|---|
| sa_1000 | hall | `a brown dog` | -0.0239 | -0.0268 | -0.0022 |
| sa_1000 | hall | `a woven basket` | -0.0283 | -0.0279 | +0.0053 |
| sa_1000 | hall | `a brass oil lamp` | -0.0360 | -0.0346 | -0.0016 |
| sa_1000 | hall | `a wooden ladder` | -0.0202 | -0.0244 | -0.0063 |
| sa_1000 | real | `two human figures` | -0.0074 | -0.0178 | +0.0098 |
| sa_1000 | real | `a green jacket` | -0.0159 | -0.0247 | -0.0110 |
| sa_1000 | real | `a striped skirt` | -0.0268 | -0.0258 | +0.0043 |
| sa_1000 | real | `a headdress` | -0.0155 | -0.0196 | +0.0006 |
| sa_183 | hall | `a checkered flag` | -0.0420 | -0.0234 | -0.0062 |
| sa_183 | real | `a white race car` | -0.0478 | -0.0314 | -0.0086 |
| sa_183 | real | `orange and blue livery` | -0.0314 | -0.0267 | -0.0108 |
| sa_183 | real | `green grass` | -0.0402 | -0.0257 | -0.0020 |
| sa_2646 | hall | `a brown dog` | -0.0285 | -0.0191 | -0.0063 |
| sa_2646 | hall | `a red bicycle` | -0.0298 | -0.0233 | -0.0053 |
| sa_2646 | hall | `a child in a blue jacket` | -0.0370 | -0.0218 | +0.0052 |
| sa_2646 | hall | `a wooden bench` | -0.0503 | -0.0232 | -0.0136 |
| sa_2646 | real | `women walking` | -0.0106 | -0.0013 | +0.0072 |
| sa_2646 | real | `a white building` | -0.0243 | -0.0199 | -0.0083 |
| sa_2646 | real | `green trees` | -0.0358 | -0.0160 | -0.0096 |
| sa_3439 | hall | `a fire hydrant` | -0.0420 | -0.0289 | +0.0122 |
| sa_3439 | hall | `a brown briefcase` | -0.0268 | -0.0283 | -0.0067 |
| sa_3439 | hall | `a wooden bench` | -0.0440 | -0.0276 | -0.0113 |
| sa_3439 | hall | `a street lamp` | -0.0169 | -0.0141 | +0.0027 |
| sa_3439 | real | `a man in a blue blazer` | -0.0266 | -0.0288 | -0.0055 |
| sa_3439 | real | `a yellow auto-rickshaw` | -0.0335 | -0.0181 | +0.0052 |
| sa_3439 | real | `a motorcycle` | -0.0171 | -0.0140 | +0.0063 |
| sa_4286 | hall | `a stone lion statue` | -0.0156 | -0.0249 | -0.0037 |
| sa_4286 | hall | `a wooden rowboat` | -0.0134 | -0.0196 | -0.0080 |
| sa_4286 | hall | `a metal flagpole` | -0.0249 | -0.0307 | -0.0193 |
| sa_4286 | hall | `a campfire` | -0.0164 | -0.0118 | -0.0071 |
| sa_4286 | real | `an ancient stone temple` | -0.0167 | -0.0157 | +0.0185 |
| sa_4286 | real | `green mountains` | +0.0050 | -0.0151 | -0.0085 |
| sa_5122 | hall | `a black bird` | -0.0200 | -0.0145 | -0.0106 |
| sa_5122 | hall | `a brass mailbox` | -0.0551 | -0.0315 | -0.0142 |
| sa_5122 | hall | `a potted plant` | -0.0278 | -0.0143 | -0.0061 |
| sa_5122 | hall | `a set of keys` | -0.0451 | -0.0287 | -0.0181 |
| sa_5122 | real | `a tiled street sign` | -0.0579 | -0.0297 | -0.0061 |
| sa_5122 | real | `ceramic tiles` | -0.0268 | -0.0144 | +0.0002 |
| sa_5122 | real | `blue and yellow patterns` | -0.0412 | -0.0274 | -0.0165 |
| sa_5954 | hall | `a yellow crane` | -0.0106 | -0.0099 | -0.0020 |
| sa_5954 | hall | `a red bicycle` | -0.0271 | -0.0124 | -0.0011 |
| sa_5954 | hall | `a green bench` | -0.0242 | -0.0166 | +0.0047 |
| sa_5954 | hall | `a silver motorcycle` | -0.0219 | -0.0147 | -0.0090 |
| sa_5954 | real | `a blue booth` | -0.0053 | -0.0098 | -0.0081 |
| sa_5954 | real | `a crowd of people` | -0.0364 | -0.0134 | +0.0093 |
| sa_5954 | real | `a man in a blue jacket` | -0.0209 | -0.0144 | +0.0011 |
| sa_6749 | hall | `a red double-decker bus` | -0.0375 | -0.0166 | +0.0061 |
| sa_6749 | hall | `a group of cyclists` | -0.0379 | -0.0144 | +0.0099 |
| sa_6749 | hall | `a large fountain` | -0.0347 | -0.0186 | +0.0001 |
| sa_6749 | hall | `a yellow taxi` | -0.0309 | -0.0193 | +0.0011 |
| sa_6749 | real | `a tall cylindrical tower` | -0.0392 | -0.0157 | +0.0187 |
| sa_6749 | real | `a modern building` | -0.0199 | -0.0112 | +0.0102 |
| sa_6749 | real | `a concrete walkway` | -0.0184 | -0.0119 | +0.0113 |
| sa_7547 | hall | `a brown dog` | -0.0211 | -0.0182 | +0.0015 |
| sa_7547 | hall | `a red bicycle` | -0.0268 | -0.0224 | -0.0008 |
| sa_7547 | hall | `a cat` | -0.0272 | -0.0112 | -0.0080 |
| sa_7547 | hall | `a wooden bench` | -0.0327 | -0.0146 | +0.0007 |
| sa_7547 | real | `parked motorcycles` | -0.0002 | +0.0078 | +0.0127 |
| sa_7547 | real | `a metal shutter` | -0.0300 | -0.0112 | +0.0106 |
| sa_7547 | real | `a group of people` | -0.0442 | -0.0222 | +0.0072 |
| sa_8367 | hall | `a wooden bench` | -0.0250 | -0.0094 | -0.0104 |
| sa_8367 | hall | `a blue bicycle` | -0.0196 | -0.0143 | -0.0073 |
| sa_8367 | hall | `a small fountain` | -0.0262 | -0.0170 | -0.0020 |
| sa_8367 | hall | `a group of pigeons` | -0.0397 | -0.0291 | -0.0062 |
| sa_8367 | real | `a statue` | -0.0185 | -0.0100 | +0.0019 |
| sa_8367 | real | `tiled roofs` | -0.0315 | -0.0177 | +0.0007 |
| sa_9166 | hall | `a red stop sign` | -0.0400 | -0.0233 | +0.0003 |
| sa_9166 | hall | `a wooden bench` | -0.0371 | -0.0174 | -0.0025 |
| sa_9166 | hall | `a brown dog` | -0.0273 | -0.0167 | -0.0036 |
| sa_9166 | hall | `a yellow bicycle` | -0.0281 | -0.0182 | -0.0004 |
| sa_9166 | real | `a police officer on a motorcycle` | -0.0154 | -0.0002 | +0.0248 |
| sa_9166 | real | `a blue truck` | -0.0088 | -0.0119 | +0.0024 |
| sa_9166 | real | `a silver car` | -0.0129 | -0.0050 | +0.0044 |
| sa_9998 | hall | `a wooden rowboat` | -0.0344 | -0.0068 | +0.0004 |
| sa_9998 | hall | `a green rowboat` | -0.0306 | -0.0117 | +0.0022 |
| sa_9998 | hall | `a white seagull` | -0.0303 | -0.0072 | +0.0027 |
| sa_9998 | hall | `a red kayak` | -0.0315 | -0.0118 | -0.0080 |
| sa_9998 | real | `white sailboats` | +0.0042 | -0.0009 | +0.0055 |
| sa_9998 | real | `wooden docks` | -0.0203 | -0.0173 | -0.0059 |
| sa_9998 | real | `a person with a camera` | -0.0499 | -0.0170 | -0.0014 |
