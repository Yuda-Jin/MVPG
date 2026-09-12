# FG-CLIP 区分「易负样本 vs 难负样本」验证报告

- 数据：`/root/autodl-tmp/data/SA1B_5_threshold`（主图 + 子图 + `neg_captions.json` + `model_captions.json`）

- 模型：FG-CLIP2-base（`/root/autodl-tmp/cache/fgclip2-base-patch16`），CPU 推理

- 抽样：从 4848 个有效组中均匀抽 12 组

- 指标：FG-CLIP 图文余弦相似度（L2 归一化后点积），越高越接近图像内容


## 结论

- 容易负样本（幻觉）平均相似度 **0.1478**，难负样本 7B **0.1520**、13B **0.1494**。

- 每组「最难的易负样本」相对「最弱的难负样本」的分离 margin：均值 -0.0487，最小 -0.0967；margin>0 的组 **1/12**。

- **判定：部分区分（1/12 组 margin>0）**。


## 主图结果（每组：易负样本相似度列表 vs 7B/13B 难负样本）

| gid | 易负样本 sim（min/mean/max） | 7B 难负 sim | 13B 难负 sim | margin |
|---|---|---|---|---|
| sa_1000 | 0.1677 / 0.1802 / 0.1981 | 0.1715 | 0.1476 | -0.0505 |
| sa_183 | 0.0922 / 0.1335 / 0.2124 | 0.1436 | 0.1619 | -0.0688 |
| sa_2646 | 0.0568 / 0.0940 / 0.1557 | 0.1902 | 0.1599 | +0.0042 |
| sa_3439 | 0.0889 / 0.1644 / 0.2213 | 0.1496 | 0.1822 | -0.0718 |
| sa_4286 | 0.0602 / 0.1246 / 0.1932 | 0.1755 | 0.1818 | -0.0176 |
| sa_5122 | 0.1382 / 0.1590 / 0.1847 | 0.1503 | 0.1296 | -0.0551 |
| sa_5954 | 0.1228 / 0.1453 / 0.1630 | 0.0867 | 0.1375 | -0.0763 |
| sa_6749 | 0.1064 / 0.1381 / 0.1686 | 0.1256 | 0.0719 | -0.0967 |
| sa_7547 | 0.1146 / 0.1315 / 0.1420 | 0.1403 | 0.1723 | -0.0017 |
| sa_8367 | 0.1235 / 0.1644 / 0.1955 | 0.1690 | 0.1648 | -0.0307 |
| sa_9166 | 0.1621 / 0.1936 / 0.2202 | 0.1836 | 0.1725 | -0.0477 |
| sa_9998 | 0.1058 / 0.1449 / 0.1816 | 0.1378 | 0.1106 | -0.0710 |

## 汇总统计

| 类别 | 平均相似度 | 最小 | 最大 |
|---|---|---|---|
| 易负样本（幻觉，整图级） | 0.1478 | 0.0568 | 0.2213 |
| 难负样本 7B（原模型主图 caption） | 0.1520 | - | - |
| 难负样本 13B（原模型主图 caption） | 0.1494 | - | - |

## 子图级难负样本（区域 caption vs 其区域图）

共 48 张子图；7B 区域 caption 平均相似度 **0.1364**，13B **0.1302**。


## 抽样示例（前 3 组）


### sa_1000

- 7B 难负样本 caption：`The image displays a beautiful wall painting featuring two characters, a couple with blurry heads. The people in the painting are facing each other and seemingly greeting or bidding farewell. Both of `

- 13B 难负样本 caption：`The image portrays a beautifully painted fresco of two people, likely a man and a woman, having an intimate moment next to each other in an artistic manner. The fresco is a classic size and features v`

- 易负样本（幻觉）示例：`The image features an ancient wall painting depicting two figures standing close together against a weathered orange and beige background. The figure on the left is a shirtless man with elaborate tatt`


### sa_183

- 7B 难负样本 caption：`The image features a silver car racing down a paved road between the track's edge. The car is cruising around a curve, accelerating down the track, giving the appearance of an enthusiastic race. Since`

- 13B 难负样本 caption：`The image portrays a black and white racetrack with an orange and blue Honda Civic driving around a curve near the center of the scene. The racing car appears to be skillfully navigating the turn, as `

- 易负样本（幻觉）示例：`The image features a white Honda race car driving on an asphalt track, with a grassy verge visible along the right side. The car has an orange and blue livery with the number 16 on the windshield, and`


### sa_2646

- 7B 难负样本 caption：`The scene depicts four women wearing different outfits walking across a brick promenade together. They appear to be enjoying their time strolling down the walkway. Among them, one woman can be seen we`

- 13B 难负样本 caption：`In this long, blurry image, a group of four young Asian ladies are standing on a closer road. They seem to be enjoying their time together outdoors. The women are all wearing colorful clothing, sugges`

- 易负样本（幻觉）示例：`The image features a group of five women walking across a paved plaza in front of a white building with classical columns. A large brown dog is walking alongside the women on the stone path, and lush `

