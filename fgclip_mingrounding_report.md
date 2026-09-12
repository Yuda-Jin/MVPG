# 短语级最小支撑（min-grounding）句级信号验证报告

- 数据：`/root/autodl-tmp/data/SA1B_5_threshold`（同 12 组主图）

- 模型：FG-CLIP2-base（`/root/autodl-tmp/cache/fgclip2-base-patch16`），CPU 推理

- 方法：正常句 = `[base] + real`，幻觉句 = `[base] + real + [1个hall]`；逐短语算 patch 支撑度 `g_k = max_patch cos(text(box/short), dense)`，句级分数取 `min_k g_k`（或 `mean_k g_k` 作对照）。幻觉句应显著更低。


## 结论

- **min + box**：正常句 0.2320 → 幻觉句 0.2250；Cohen's d **+0.53**；margin 均值 +0.0070；单短语 margin>0 比例 **46.7%**；组级 margin>0 **9/12**。

- **min + short**：正常句 0.2010 → 幻觉句 0.1897；Cohen's d **+0.56**；margin 均值 +0.0114；单短语 margin>0 比例 **46.7%**；组级 margin>0 **8/12**。

- **mean + box**：正常句 0.2477 → 幻觉句 0.2444；Cohen's d **+0.31**；margin 均值 +0.0033；单短语 margin>0 比例 **84.4%**；组级 margin>0 **10/12**。

- **mean + short**：正常句 0.2276 → 幻觉句 0.2229；Cohen's d **+0.32**；margin 均值 +0.0047；单短语 margin>0 比例 **80.0%**；组级 margin>0 **10/12**。

- **判定：最强组合为 min+short，Cohen's d=+0.56，仍不达稳定(d<1.0)**。


## 分组结果（min+box 的句级 min 分数）

| gid | 正常句 min | 幻觉句 min（min/mean） | 组级 margin |
|---|---|---|---|
| sa_1000 | 0.2421 | 0.2215 / 0.2345 | +0.0076 |
| sa_183 | 0.1961 | 0.1961 / 0.1961 | +0.0000 |
| sa_2646 | 0.2222 | 0.2148 / 0.2204 | +0.0019 |
| sa_3439 | 0.2117 | 0.2117 / 0.2117 | +0.0000 |
| sa_4286 | 0.2346 | 0.2060 / 0.2246 | +0.0101 |
| sa_5122 | 0.2285 | 0.2285 / 0.2285 | +0.0000 |
| sa_5954 | 0.2246 | 0.2140 / 0.2194 | +0.0052 |
| sa_6749 | 0.2346 | 0.2058 / 0.2115 | +0.0231 |
| sa_7547 | 0.2268 | 0.2177 / 0.2245 | +0.0023 |
| sa_8367 | 0.2488 | 0.2039 / 0.2329 | +0.0159 |
| sa_9166 | 0.2555 | 0.2406 / 0.2500 | +0.0055 |
| sa_9998 | 0.2312 | 0.2174 / 0.2241 | +0.0071 |

## 逐短语明细（patch 支撑度 g_k，box 头）

| gid | 类型 | 短语 | g_k(box) | g_k(short) |
|---|---|---|---|---|
| sa_1000 | hall | `a brown dog` | 0.2480 | 0.2311 |
| sa_1000 | hall | `a woven basket` | 0.2325 | 0.2083 |
| sa_1000 | hall | `a brass oil lamp` | 0.2215 | 0.2036 |
| sa_1000 | hall | `a wooden ladder` | 0.2583 | 0.2492 |
| sa_183 | hall | `a checkered flag` | 0.2052 | 0.1742 |
| sa_2646 | hall | `a brown dog` | 0.2374 | 0.1992 |
| sa_2646 | hall | `a red bicycle` | 0.2148 | 0.1666 |
| sa_2646 | hall | `a child in a blue jacket` | 0.2278 | 0.1780 |
| sa_2646 | hall | `a wooden bench` | 0.2268 | 0.1801 |
| sa_3439 | hall | `a fire hydrant` | 0.2286 | 0.1780 |
| sa_3439 | hall | `a brown briefcase` | 0.2496 | 0.2489 |
| sa_3439 | hall | `a wooden bench` | 0.2370 | 0.1917 |
| sa_3439 | hall | `a street lamp` | 0.2459 | 0.2443 |
| sa_4286 | hall | `a stone lion statue` | 0.2230 | 0.2108 |
| sa_4286 | hall | `a wooden rowboat` | 0.2381 | 0.2173 |
| sa_4286 | hall | `a metal flagpole` | 0.2060 | 0.2002 |
| sa_4286 | hall | `a campfire` | 0.2345 | 0.1757 |
| sa_5122 | hall | `a black bird` | 0.2741 | 0.2769 |
| sa_5122 | hall | `a brass mailbox` | 0.2368 | 0.2327 |
| sa_5122 | hall | `a potted plant` | 0.2588 | 0.2515 |
| sa_5122 | hall | `a set of keys` | 0.2302 | 0.2018 |
| sa_5954 | hall | `a yellow crane` | 0.2265 | 0.2380 |
| sa_5954 | hall | `a red bicycle` | 0.2142 | 0.1838 |
| sa_5954 | hall | `a green bench` | 0.2351 | 0.2125 |
| sa_5954 | hall | `a silver motorcycle` | 0.2140 | 0.1906 |
| sa_6749 | hall | `a red double-decker bus` | 0.2058 | 0.1478 |
| sa_6749 | hall | `a group of cyclists` | 0.2227 | 0.1864 |
| sa_6749 | hall | `a large fountain` | 0.2069 | 0.1625 |
| sa_6749 | hall | `a yellow taxi` | 0.2106 | 0.1888 |
| sa_7547 | hall | `a brown dog` | 0.2399 | 0.2104 |
| sa_7547 | hall | `a red bicycle` | 0.2177 | 0.1851 |
| sa_7547 | hall | `a cat` | 0.2479 | 0.2384 |
| sa_7547 | hall | `a wooden bench` | 0.2282 | 0.1960 |
| sa_8367 | hall | `a wooden bench` | 0.2506 | 0.2169 |
| sa_8367 | hall | `a blue bicycle` | 0.2331 | 0.2152 |
| sa_8367 | hall | `a small fountain` | 0.2459 | 0.2200 |
| sa_8367 | hall | `a group of pigeons` | 0.2039 | 0.1772 |
| sa_9166 | hall | `a red stop sign` | 0.2558 | 0.2191 |
| sa_9166 | hall | `a wooden bench` | 0.2485 | 0.1957 |
| sa_9166 | hall | `a brown dog` | 0.2600 | 0.2173 |
| sa_9166 | hall | `a yellow bicycle` | 0.2406 | 0.2076 |
| sa_9998 | hall | `a wooden rowboat` | 0.2320 | 0.1980 |
| sa_9998 | hall | `a green rowboat` | 0.2174 | 0.1989 |
| sa_9998 | hall | `a white seagull` | 0.2238 | 0.1905 |
| sa_9998 | hall | `a red kayak` | 0.2240 | 0.2261 |
