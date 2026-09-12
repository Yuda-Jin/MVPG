# 短语级 Patch Grounding 幻觉信号验证报告

- 数据：`/root/autodl-tmp/data/SA1B_5_threshold`（上一轮同 12 组主图）

- 模型：FG-CLIP2-base（`/root/autodl-tmp/cache/fgclip2-base-patch16`），CPU 推理

- 方法：`get_text_features(phrase, walk_type=box/short)` × `get_image_dense_feature()`，对每个短语取「所有有效 patch 上余弦相似度的最大值」`max_patch_sim`

- 对比：**幻觉物体短语**（图中不存在） vs **真实物体短语**（图中存在），分数越低表示越缺乏图像局部支撑


## 结论

- **box 头**：真实短语 max_patch_sim 均值 **0.2451**（σ=0.0194），幻觉短语 **0.2320**（σ=0.0166）；每组 margin(real−hall) 均值 **+0.0146**，margin>0 的组 **10/12**；效应量 Cohen's d **+0.72**；短语级分类正确率 **64.7%**。

- **short 头（对照）**：真实短语 **0.2226**（σ=0.0298），幻觉短语 **0.2054**（σ=0.0265）；margin 均值 **+0.0182**，margin>0 的组 **10/12**；Cohen's d **+0.61**；短语级分类正确率 **63.5%**。

- box 头信号判定：**不稳定/弱**（Cohen's d=+0.72，margin 强度 +0.0146）。

- short 头信号判定：**不稳定/弱**（Cohen's d=+0.61，margin 强度 +0.0182）。

- **总体判定：短语级 patch grounding（box 头 max_patch_sim）不能提供稳定幻觉信号**。


## 分组结果（每组：幻觉 vs 真实 的 box max_patch_sim）

| gid | 幻觉短语 box（min/mean/max） | 真实短语 box（min/mean/max） | margin(box) | margin(short) |
|---|---|---|---|---|
| sa_1000 | 0.2215 / 0.2401 / 0.2583 | 0.2421 / 0.2608 / 0.2789 | +0.0208 | +0.0294 |
| sa_183 | 0.2052 / 0.2052 / 0.2052 | 0.1961 / 0.2253 / 0.2528 | +0.0201 | +0.0288 |
| sa_2646 | 0.2148 / 0.2267 / 0.2374 | 0.2222 / 0.2386 / 0.2546 | +0.0119 | +0.0388 |
| sa_3439 | 0.2286 / 0.2403 / 0.2496 | 0.2117 / 0.2221 / 0.2417 | -0.0182 | -0.0310 |
| sa_4286 | 0.2060 / 0.2254 / 0.2381 | 0.2346 / 0.2494 / 0.2701 | +0.0240 | +0.0319 |
| sa_5122 | 0.2302 / 0.2500 / 0.2741 | 0.2285 / 0.2423 / 0.2619 | -0.0077 | -0.0198 |
| sa_5954 | 0.2140 / 0.2225 / 0.2351 | 0.2246 / 0.2403 / 0.2520 | +0.0179 | +0.0077 |
| sa_6749 | 0.2058 / 0.2115 / 0.2227 | 0.2346 / 0.2386 / 0.2449 | +0.0271 | +0.0402 |
| sa_7547 | 0.2177 / 0.2334 / 0.2479 | 0.2268 / 0.2485 / 0.2672 | +0.0150 | +0.0155 |
| sa_8367 | 0.2039 / 0.2334 / 0.2506 | 0.2331 / 0.2496 / 0.2667 | +0.0162 | +0.0341 |
| sa_9166 | 0.2406 / 0.2512 / 0.2600 | 0.2708 / 0.2746 / 0.2771 | +0.0234 | +0.0303 |
| sa_9998 | 0.2174 / 0.2243 / 0.2320 | 0.2312 / 0.2493 / 0.2656 | +0.0251 | +0.0121 |

## 逐短语明细（box max_patch_sim）

| gid | 类别 | 短语 | box_sim | short_sim |
|---|---|---|---|---|
| sa_1000 | hall | `a brown dog` | 0.2480 | 0.2311 |
| sa_1000 | hall | `a woven basket` | 0.2325 | 0.2083 |
| sa_1000 | hall | `a brass oil lamp` | 0.2215 | 0.2036 |
| sa_1000 | hall | `a wooden ladder` | 0.2583 | 0.2492 |
| sa_1000 | real | `two human figures` | 0.2718 | 0.2689 |
| sa_1000 | real | `a wall painting` | 0.2789 | 0.2857 |
| sa_1000 | real | `a green jacket` | 0.2521 | 0.2559 |
| sa_1000 | real | `a striped skirt` | 0.2421 | 0.2145 |
| sa_1000 | real | `a headdress` | 0.2592 | 0.2372 |
| sa_183 | hall | `a checkered flag` | 0.2052 | 0.1742 |
| sa_183 | real | `a white race car` | 0.2353 | 0.2206 |
| sa_183 | real | `an asphalt track` | 0.2169 | 0.2164 |
| sa_183 | real | `green grass` | 0.1961 | 0.1398 |
| sa_183 | real | `orange and blue livery` | 0.2528 | 0.2351 |
| sa_2646 | hall | `a brown dog` | 0.2374 | 0.1992 |
| sa_2646 | hall | `a red bicycle` | 0.2148 | 0.1666 |
| sa_2646 | hall | `a child in a blue jacket` | 0.2278 | 0.1780 |
| sa_2646 | hall | `a wooden bench` | 0.2268 | 0.1801 |
| sa_2646 | real | `women walking` | 0.2546 | 0.2323 |
| sa_2646 | real | `a white building` | 0.2378 | 0.2287 |
| sa_2646 | real | `green trees` | 0.2222 | 0.1969 |
| sa_2646 | real | `a paved plaza` | 0.2398 | 0.2211 |
| sa_3439 | hall | `a fire hydrant` | 0.2286 | 0.1780 |
| sa_3439 | hall | `a brown briefcase` | 0.2496 | 0.2489 |
| sa_3439 | hall | `a wooden bench` | 0.2370 | 0.1917 |
| sa_3439 | hall | `a street lamp` | 0.2459 | 0.2443 |
| sa_3439 | real | `a man in a blue blazer` | 0.2129 | 0.1638 |
| sa_3439 | real | `a yellow auto-rickshaw` | 0.2117 | 0.1784 |
| sa_3439 | real | `a motorcycle` | 0.2417 | 0.2119 |
| sa_4286 | hall | `a stone lion statue` | 0.2230 | 0.2108 |
| sa_4286 | hall | `a wooden rowboat` | 0.2381 | 0.2173 |
| sa_4286 | hall | `a metal flagpole` | 0.2060 | 0.2002 |
| sa_4286 | hall | `a campfire` | 0.2345 | 0.1757 |
| sa_4286 | real | `an ancient stone temple` | 0.2701 | 0.2419 |
| sa_4286 | real | `green mountains` | 0.2346 | 0.2287 |
| sa_4286 | real | `a grassy field` | 0.2434 | 0.2279 |
| sa_5122 | hall | `a black bird` | 0.2741 | 0.2769 |
| sa_5122 | hall | `a brass mailbox` | 0.2368 | 0.2327 |
| sa_5122 | hall | `a potted plant` | 0.2588 | 0.2515 |
| sa_5122 | hall | `a set of keys` | 0.2302 | 0.2018 |
| sa_5122 | real | `a tiled street sign` | 0.2285 | 0.1934 |
| sa_5122 | real | `ceramic tiles` | 0.2619 | 0.2268 |
| sa_5122 | real | `blue and yellow patterns` | 0.2364 | 0.2427 |
| sa_5954 | hall | `a yellow crane` | 0.2265 | 0.2380 |
| sa_5954 | hall | `a red bicycle` | 0.2142 | 0.1838 |
| sa_5954 | hall | `a green bench` | 0.2351 | 0.2125 |
| sa_5954 | hall | `a silver motorcycle` | 0.2140 | 0.1906 |
| sa_5954 | real | `a blue booth` | 0.2520 | 0.2544 |
| sa_5954 | real | `a crowd of people` | 0.2445 | 0.2037 |
| sa_5954 | real | `a man in a blue jacket` | 0.2246 | 0.1836 |
| sa_6749 | hall | `a red double-decker bus` | 0.2058 | 0.1478 |
| sa_6749 | hall | `a group of cyclists` | 0.2227 | 0.1864 |
| sa_6749 | hall | `a large fountain` | 0.2069 | 0.1625 |
| sa_6749 | hall | `a yellow taxi` | 0.2106 | 0.1888 |
| sa_6749 | real | `a tall cylindrical tower` | 0.2449 | 0.2077 |
| sa_6749 | real | `a modern building` | 0.2362 | 0.2151 |
| sa_6749 | real | `a concrete walkway` | 0.2346 | 0.2119 |
| sa_7547 | hall | `a brown dog` | 0.2399 | 0.2104 |
| sa_7547 | hall | `a red bicycle` | 0.2177 | 0.1851 |
| sa_7547 | hall | `a cat` | 0.2479 | 0.2384 |
| sa_7547 | hall | `a wooden bench` | 0.2282 | 0.1960 |
| sa_7547 | real | `parked motorcycles` | 0.2514 | 0.2149 |
| sa_7547 | real | `a metal shutter` | 0.2672 | 0.2512 |
| sa_7547 | real | `a group of people` | 0.2268 | 0.2030 |
| sa_8367 | hall | `a wooden bench` | 0.2506 | 0.2169 |
| sa_8367 | hall | `a blue bicycle` | 0.2331 | 0.2152 |
| sa_8367 | hall | `a small fountain` | 0.2459 | 0.2200 |
| sa_8367 | hall | `a group of pigeons` | 0.2039 | 0.1772 |
| sa_8367 | real | `a statue` | 0.2488 | 0.2432 |
| sa_8367 | real | `a cobblestone square` | 0.2331 | 0.2131 |
| sa_8367 | real | `tiled roofs` | 0.2667 | 0.2680 |
| sa_9166 | hall | `a red stop sign` | 0.2558 | 0.2191 |
| sa_9166 | hall | `a wooden bench` | 0.2485 | 0.1957 |
| sa_9166 | hall | `a brown dog` | 0.2600 | 0.2173 |
| sa_9166 | hall | `a yellow bicycle` | 0.2406 | 0.2076 |
| sa_9166 | real | `a police officer on a motorcycle` | 0.2758 | 0.1958 |
| sa_9166 | real | `a blue truck` | 0.2708 | 0.2569 |
| sa_9166 | real | `a silver car` | 0.2771 | 0.2681 |
| sa_9998 | hall | `a wooden rowboat` | 0.2320 | 0.1980 |
| sa_9998 | hall | `a green rowboat` | 0.2174 | 0.1989 |
| sa_9998 | hall | `a white seagull` | 0.2238 | 0.1905 |
| sa_9998 | hall | `a red kayak` | 0.2240 | 0.2261 |
| sa_9998 | real | `white sailboats` | 0.2513 | 0.1966 |
| sa_9998 | real | `wooden docks` | 0.2656 | 0.2485 |
| sa_9998 | real | `a person with a camera` | 0.2312 | 0.2015 |
