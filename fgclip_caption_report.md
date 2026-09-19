# 段落级判别：幻觉 caption vs 事实 caption（冻结 FG-CLIP2）

把整条 caption 当一个单位，问冻结的 FG-CLIP2 能否区分「含幻觉物体的 caption」（`neg_captions.json`，强 VLM 生成、每条恰好注入 1 个图中不存在的物体）与「原模型（LLaVA-7B/13B）生成的 caption」（`model_captions.json`，**未经事实核验**）。两类文本描述**同一张主图**，是天然配对。

- 图 4848 张；正 caption 9696 条（每图 2.0 条）、负 caption 19392 条（每图 4.0 条）
- `global` = 整图 pooled 特征 × 文本余弦；`dense` = patch 特征与文本余弦取 max；`swap_global` = 同一段文本打到**另一张无关图**上的 global 余弦（对照）

## 0. 样本抽查（同一张图上的负 / 正 caption）

先看清两类文本的真实差别，再读后面的分数——这一点很关键，见第 5 节。

**`sa_1000`**

- 负（幻觉，366 字符）：The image features an ancient wall painting depicting two figures standing close together against a weathered orange and beige background. The figure on the left is a shirtless man with elaborate tattoos and a dark headdress, while the figure on the right wear…
- 正（llava-v1.5-7b，471 字符）：The image displays a beautiful wall painting featuring two characters, a couple with blurry heads. The people in the painting are facing each other and seemingly greeting or bidding farewell. Both of the main figures are wearing crowns.  There is a chair that …

**`sa_1001`**

- 负（幻觉，334 字符）：The image features a brightly lit clothing booth with two matching sets hanging on the wall, one in dusty pink and one in cream. A group of shoppers gathers around a table of folded garments, while a large poster of a car and models is displayed on the right. …
- 正（llava-v1.5-7b，548 字符）：The image features a group of people gathered around a booth with a row of mannequins wearing clothes, possibly attending a mall event or fashion show. There are several attendees, with one person looking at the mannequins on their own, and a couple of people …

负 caption 由强 VLM 按「先推断场景 → 再自然描述 + 塞入 1 个图中不存在的物体」生成，**除那 1 个物体外基本忠实**；而正 caption 是 LLaVA-7B/13B 的开放式描述，**本身也常含幻觉/冗长推测**（上例 LLaVA 说的 crowns、chair 大概率并不存在于图中）。所以本实验的「事实正样本」并不干净，这是读结论时必须记住的前提。

## 1. 池化 AUC（正 caption 分 > 负 caption 分，随机 0.5）

| 信号 | 正均值 | 负均值 | AUC | 正−负 差 |
|---|---|---|---|---|
| global | 0.1569 | 0.1918 | **0.222** | -0.0349 |
| dense | 0.1936 | 0.2132 | **0.290** | -0.0196 |
| swap_global | -0.0282 | -0.0399 | **0.567** | 0.0117 |

## 2. 逐图「正 caption 排第一」准确率

每张图有若干正/负 caption，问**正 caption 的最高分**能否压过**全部负 caption**（随机基线 = 正样本占比）。另给「最弱正 vs 最强负」的分离 margin。

| 信号 | n 图 | top-1 准确率 | 随机基线 | 增益 | margin>0 比例 | 中位 margin |
|---|---|---|---|---|---|---|
| global | 4848 | **0.086** | 0.333 | -0.248 | 0.013 | -0.0699 |
| dense | 4848 | **0.121** | 0.333 | -0.212 | 0.019 | -0.0468 |
| swap_global | 4848 | **0.514** | 0.333 | +0.180 | 0.199 | -0.0271 |

## 3. 长度混淆：长度本身就能判到多少

- 正 caption 平均 530 字符，负 caption 平均 299 字符（比 0.56×）。
- **只用长度**判别两类文本的 AUC = 0.986（>0.5 = 越长越像正样本；≈0.5 = 长度无信息）。
- 两类长度差这么大时，任何「正/负分数差」都可能只是长度造成的。下表用**长度分层**（按长度分位分 10 箱，只在同箱内比正 vs 负再加权）把长度这一解释剔掉；「同长度窗」列则只保留长度落在正 caption 均值 ±25% 内的负样本，但本数据负样本普遍短得多，窗口会删掉绝大多数负样本。

| 信号 | 长度分层 AUC | 有效箱数 | 同窗保留图数 | 同窗 top-1 | 同窗 AUC | 同窗保留负样本数 |
|---|---|---|---|---|---|---|
| global | 0.257 | 8 | 511 | 0.133 | 0.174 | 1185 |
| dense | 0.314 | 8 | 511 | 0.229 | 0.291 | 1185 |
| swap_global | 0.456 | 8 | 511 | 0.634 | 0.552 | 1185 |

## 4. 分模型（正 caption 只取 7B / 只取 13B）

| 正样本来源 | 信号 | 池化 AUC |
|---|---|---|
| llava-v1.5-7b | global | 0.210 |
| llava-v1.5-7b | dense | 0.283 |
| llava-v1.5-7b | swap_global | 0.570 |
| llava-v1.5-13b | global | 0.233 |
| llava-v1.5-13b | dense | 0.298 |
| llava-v1.5-13b | swap_global | 0.564 |

## 5. 判读

1. **段落级判别力**：`global` AUC 0.222 < 0.5，**方向反转**（幻觉 caption 反而更贴近本图）；`dense` AUC 0.290 < 0.5，**方向反转**（幻觉 caption 反而更贴近本图）。逐图「正 caption 最高分压过全部负 caption」的准确率：`global` 0.086、`dense` 0.121，随机基线 0.333。
   分数方向与「幻觉 ⇒ 低分」的直觉相反。这**不能**直接读成「FG-CLIP2 判反了」——更合理的解释是登记为「事实」的那一类（LLaVA caption）本身不可靠（见第 0 节）：它们更长、更空泛、且自身夹带幻觉物体与大量推测，整段文本的视觉贴合度被自己拉低。
2. **换图对照（关键）**：同一段文本打到**另一张无关图**上，`swap_global` AUC 0.567，与打在本图上的 `global`（0.222）**方向相反**。换图对照里不含任何「这段文本是否贴合本图」的信息，只有文本侧统计量（长度 / 用词 / 句式）；它在无关图上让正样本略占优，而一放回**本图**，优势就翻转成负样本——说明**本图视觉证据是往「幻觉 caption 分更低」的方向作用的**，只是被文本侧先验和正样本本身的不忠实掩盖了。
3. **长度混淆**：只用长度就能达到 AUC 0.986，是压倒性混淆。剔除长度（按长度分箱再加权）后：`global` 0.257（8 箱）、`dense` 0.314（8 箱）、`swap_global` 0.456（8 箱）。方向没有因为剔除长度而翻正，说明「正样本分更低」不只是长度造成的。
4. **结论**：本配对设计**回答不了**「段落级能否区分幻觉 vs 事实」——被登记为「事实」的 LLaVA caption 本身也不忠实（第 0 节）且长得多（第 3 节），两类并不是干净的「事实 vs 幻觉」。就本数据能确证的是：
   - 负 caption 按生成约定**只注入 1 个**图中不存在的物体，其余部分忠实；这么「1 个假物体」摊进整段余弦后会被其余忠实内容稀释，段落级相似度几乎感知不到它；
   - `global`/`dense` 的段落级 AUC 停在 0.222/0.290（反向），逐图 top-1 只有 0.086/0.121（随机 0.333），即**按段落级分数排序，忠实 caption 反而排更后**；
   - 要干净地回答这个问题，需要**同源同长**的对照（例如同一条 caption 保留 / 删掉它的幻觉子句），而不是拿两个不同模型、不同长度的 caption 家族对比。

