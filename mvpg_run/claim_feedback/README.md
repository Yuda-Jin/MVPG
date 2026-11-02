# 声明级视觉反馈与回答内更新调制

本目录实现当前 story 的研究原型：**不提供完整正确回答、外部预标注幻觉负样本和响应级偏好对，利用自生成回答中的局部视觉反馈，研究如何纠错并保留受支持内容。**

链路为：图像与问题 → 生成回答 → 声明分解 → 视觉支持核验 → 原文 token 对齐 → 正负目标 → 更新方向调制。研究目标是当前图像条件下的事实性优化；同一对象词在另一张支持它的图像中仍应正常使用。

此处包含可运行机制原型和合成检查，**没有新方法的真实模型效果或 benchmark 结果**。细粒度反馈、无偏好对设定和梯度投影均已有相关研究基础；方法贡献需要由受控比较支撑。

## 监督条件与资源

- 训练输入是图像和问题，回答由待优化模型生成。
- 内部仍会产生受支持、疑似错误、未决片段，属于自动伪监督；“不提供正样本”不能解释成没有正向训练信号。
- 冻结解析器、视觉模型、预训练权重、阈值校准和反馈审核仍是实际依赖，必须计量。
- 阈值使用独立开发集校准；最终测试图像及标签不能参与阈值或超参数选择。
- 首版固定一次生成的回答池，不自动刷新。不同更新规则应共享回答、反馈、可训练参数和预算。

## 命令与模块

入口是 `python -m mvpg_run.claim_feedback`。使用 `--help` 或子命令的 `--help` 查看当前选项。

| 子命令 | 作用与边界 |
|---|---|
| `generate` | 从图像、问题生成回答，保存原文和生成 token IDs |
| `decompose` | 调用官方 DiscoSG 并保守投影为对象存在性单元；标点切句是明确命名的消融 |
| `verify` | 冻结视觉组件打分，使用显式双阈值与事实类型白名单 |
| `audit` | 检查弱标签数量、来源锚点和未决原因；token 覆盖率在训练日志中 |
| `train` | 固定池上按一条回答执行一次有界 SGD 更新，保存适配器与日志 |
| `smoke` | 用合成小模型检查完整接口，不读取真实视觉信息，不代表效果实验 |

`data.py` 只依赖 Python 标准库；`optimization.py` 依赖 PyTorch；真实 LLaVA、LoRA、视觉核验依赖由对应后端加载。测试另需开发依赖 `pytest`，无需下载模型即可运行 `python -m pytest -q tests/test_claim_feedback_data.py`。

## 快速运行

先检查无需真实权重的机制链路（需要 PyTorch）：

```bash
python -m mvpg_run.claim_feedback smoke --output-dir output/claim_feedback_smoke
python -m pytest -q tests
```

真实实验沿用根 README 的 patched LLaVA 环境，再安装 `requirements-claim-feedback.txt`。
输入格式见 `examples/claim_feedback/inputs.example.jsonl`，将图片名改为实际文件。
以下命令中的变量由实验者设置；阈值必须来自独立开发集，代码不提供虚构默认值。

```bash
python -m mvpg_run.claim_feedback generate \
  --input examples/claim_feedback/inputs.example.jsonl --image-root "$IMAGE_ROOT" \
  --model-name "$MODEL_NAME" --output output/responses.jsonl
python -m mvpg_run.claim_feedback decompose \
  --input output/responses.jsonl --output output/claims.jsonl \
  --disco-root "$DISCO_ROOT" --seed-model "$DISCO_SEED_MODEL" \
  --refiner-model "$DISCO_REFINER_MODEL" --device cuda
python -m mvpg_run.claim_feedback verify \
  --input output/claims.jsonl --output output/feedback.jsonl \
  --image-root "$IMAGE_ROOT" --fgclip-root "$FGCLIP_ROOT" \
  --score-mode global --walk-type short \
  --negative-threshold "$NEGATIVE_THRESHOLD" --positive-threshold "$POSITIVE_THRESHOLD" \
  --calibration-id "$CALIBRATION_ID"
python -m mvpg_run.claim_feedback audit --input output/feedback.jsonl
python -m mvpg_run.claim_feedback train \
  --input output/feedback.jsonl --model-name "$MODEL_NAME" --image-root "$IMAGE_ROOT" \
  --output-dir output/claim_modulated_seed42 --mode modulated --audit-loss-change
```

DiscoSG 使用[官方仓库](https://github.com/ShaoqLin/DiscoSG)，适配参考提交
`d2be2768ddc80199ab56f38cf50c132351a8d248`，记录实际 checkout 版本。
种子模型可从 `lizhuang144/flan-t5-base-VG-factual-sg` 获取；官方发布的
`sqlinn/DiscoSG-Refiner-Large-t5-only` 提供兼容当前源码的 T5 权重和 `classifier.pt`。
还需从 `sqlinn/DiscoSG-Refiner-Large` 取得 `model_info.json` 和 tokenizer 文件，组成完整本地 refiner。
完整 `Large` 检查点的封装键与当前加载器不同，不能直接替代 `t5-only`。
离线运行时将 `model_info.json` 的 `base_model_name` 指向本地同一 tokenizer，并保留上游文件和修改记录。
在 transformers 4.34 / tokenizers 0.14 环境，新 refiner 的 tokenizer JSON 可能不兼容。
可以用 `AutoTokenizer.from_pretrained(refiner, from_slow=True)` 从官方 SentencePiece
重建 fast tokenizer，保存到独立目录，再通过独立 refiner 配置指向它。
删除步骤需要字符偏移，不能直接换成不支持偏移的 slow tokenizer；应核对完整词表 ID、特殊标记和实际文本分词。
可在独立 DiscoSG 环境执行分解，以 JSONL 交接。
其图没有原文偏移；本实现只接受唯一、明确、非否定/非不确定的实体提及，并保留完整图用于审计。
属性与关系不会冒充已支持的训练类型。已有来源可靠的声明可直接写入相同 schema，跳过 `decompose`。

每次训练的目录必须不存在。`manifest.json` 保存回答池 SHA-256、配置、实际适配器参数范围、版本及运行状态；
`steps.jsonl` 保存各回答的 token 所有权、损失、冲突、回退、实际步长与跳过原因。
`--audit-loss-change` 增加一次前向，记录更新后选中位置的正负 NLL，有额外成本。
`final/` 和可选中间检查点保存标准 LoRA adapter；可用 `generate --adapter-path .../final` 自由生成，
或接入仓库已有 LLaVA 评估器。`--checkpoint-every` 按尝试回答数计，`--max-steps` 包含跳过的回答。
`--max-updates 100` 则要求实际成功更新 100 次；若数据或尝试预算耗尽，会保存部分 adapter、标记 `incomplete` 并返回错误。
`--adapter-path` 是从已有 adapter 开始一轮新训练，不恢复旧运行的游标。

比较五种 `--mode` 时复用同一个反馈文件、基座、LoRA 配置和 seed，并使用不同输出目录。
`--max-update-norm` 仅提供相同上限，不保证所有模式实际步长相等；严格范数匹配仍需检查日志并匹配实验预算。
如要刷新回答池，显式用新 adapter 重新 `generate → decompose → verify`，阶段重跑会清除失效的下游标注。

## 记录格式与来源约束

中间数据为 UTF-8 JSONL，每行一条回答。简化示例：

```json
{
  "id": "image-0001",
  "image": "images/0001.jpg",
  "question": "Describe the image.",
  "response": "A dog and a cat.",
  "claims": [
    {"id": "dog", "text": "A dog is present", "source_spans": [[2, 5]],
     "kind": "object", "label": "?", "weight": 0.0},
    {"id": "cat", "text": "A cat is present", "source_spans": [[12, 15]],
     "error_spans": [[12, 15]], "kind": "object", "label": "?", "weight": 0.0}
  ]
}
```

`response_ids` 是可选字段；生成阶段应保存对应 tokenizer 的真实 IDs，导入既有文本时只能在精确解码校验通过后恢复。字符跨度为 Python 字符索引的半开区间 `[start, end)`，均相对于未经裁剪、清理或改写的 `response`，不是字节偏移或 prompt 偏移。

- `source_spans` 表示声明对应的原文事实片段；`error_spans` 是其中能可靠定位的错误槽位，必须包含在来源跨度中。
- `text` 可以是解析后的声明表达，但表达改写不自动提供可信来源。不得用字符串搜索为 DiscoSG 改写结果编造锚点，重复对象、否定、作用域与指代需要解析端明确处理。
- 缺少锚点、实体绑定有歧义、声明与源事实不等价时，保留记录用于诊断，标签设为未决，不能进入训练。
- `label` 取 `+`、`-`、`?`；`weight` 是 `[0,1]` 内的有限权重；`support_score` 可选，仅保存工具分数。
- 额外的模型版本、采样配置、解析器、实体实例、对齐置信度等来源信息会保留。数值非法、重复声明 ID 或越界跨度会被拒绝。
- `sentence_claims` 是标点切分基线，输出 `kind="sentence"`、`parser="punctuation_baseline"`；不实现 DiscoSG，也不保证原子事实粒度。核验时必须显式允许该类型。

`validate_record` 验证并复制记录；`import_claims` 导入来源明确的声明；`read_jsonl` 流式读取并在异常中报告行号。验证模式拒绝同一文件内重复的回答 ID。`write_jsonl` 先完整验证与序列化，默认拒绝覆盖已有文件；只有显式指定覆盖时才替换。文件系统不支持硬链接时使用独占创建，避免无意覆盖。

## 视觉反馈是有边界的弱监督

`classify_claims(record, scorer, negative_threshold, positive_threshold, ...)` 的回调约定为：

```python
scores = scorer(record["image"], [claim["text"] for claim in eligible_claims])
```

必须返回等长的分数序列。两个阈值必须有限且负阈值小于正阈值；原型不继承其他实验的固定常数。分数达到正阈值时接受支持反馈，低于负阈值且存在可靠错误跨度时接受疑似错误反馈，中间区间保持未决。非有限分数不提供方向。

首版默认只接受 `object` 类型；额外类型必须显式开放，并独立评估验证器能力。全图图文相似度不是事实真假概率：低分可能来自小对象、遮挡、图像分辨率或表述差异；高分也可能由文本先验或局部语义匹配造成。低相似度和检测器未检出均不等于证明不存在。对象定位也不能验证完整属性、数量或关系。普通 DINO 特征不能直接替代文本条件验证器。

因此负标签表示校准条件下接受的**弱负反馈**。默认 `allow_whole_claim_negative=False`：只有声明级负判断、没有可靠错误槽位时继续未决。开启整声明负反馈是消融，应单独报告。正负反馈 precision、选择覆盖率、未决比例、自然错误与位置偏置必须先于效果结论审核。

## 原文 token 对齐

`align_claims(record, offsets, ...)` 返回训练单元和审计统计。单元格式为 `{"claim_id": ..., "label": "+/-", "weight": ..., "token_indices": [...]}`，索引仅针对回答 token。

1. 生成 IDs 必须精确解码回保存原文。慢 tokenizer 的增量解码若改写已有前缀则拒绝；不能近似猜测位置。
2. 空跨度的特殊 token 不参与反馈；选中来源内容必须完整被覆盖。截断和不完整覆盖排除整条单元，不能当成更短的声明。
3. token 跨越来源边界并包含其他非空白内容时排除该单元；边界外只有空白可接受，以兼容带前导空格的子词。
4. 正反馈使用来源事实片段，负反馈默认仅使用错误槽位；未知、零权重或绑定歧义不参与目标。
5. 正负声明共享的 token 从双方移除。同极性重复位置按更高权重、再按稳定声明 ID 分配唯一所有者，避免重复计权。

统计包含有效正负 token 数、冲突、重复、边界穿越、截断、未决、覆盖率和排除原因。字符定位只给出训练位置，不构成错误因果责任证明。

## 目标归一化与方向调制

二者是分开的步骤。对去重后声明单元的有效 token 集合 `T_j`，先定义单位平均 NLL：

\[
\ell_j=-\frac{1}{|T_j|}\sum_{t\in T_j}\log\pi_\theta(y_t\mid I,q,y_{<t}),\qquad
L_\pm=\frac{\sum_{j:a_j=\pm}w_j\ell_j}{\sum_{j:a_j=\pm}w_j}.
\]

正负目标分别归一化，空集合跳过。先对每条声明的 token 求均值，再对同极性声明加权，避免仅因长度不同改变目标相对强度。

令 `g_- = ∇L_-`、`g_+ = ∇L_+`。实际参数方向为纠错 `d_- = +g_-` 和主动鼓励 `d_+ = -g_+`。前者增加错误片段的 NLL，后者降低受支持片段的 NLL。

当两个方向冲突且保护方向范数足够大时，不对称调制为：

\[
\widetilde d_- = d_--\frac{\min(0,d_-^\top d_+)}{\|d_+\|^2}d_+,\qquad
d=\alpha\widetilde d_-+\beta d_+,\qquad \theta'=\theta+\eta d.
\]

保护模式还检查 `g_-ᵀd > tolerance`；主动正项抵消纠错时缩小其系数，仍无有效方向则跳过。候选步经过共同正比例缩放和更新范数上限，并对参数精度舍入后的实际步重新检查；数值不满足检查时不提交。

| `mode` | 目的 |
|---|---|
| `fixed` | 固定加权正负方向，不投影、不回退正系数，作为机制对照 |
| `positive` | 仅受支持片段 CE |
| `negative` | 仅错误片段 masked GA |
| `project_only` | 用正方向作投影约束，不加主动正向学习 |
| `modulated` | 冲突投影、主动正项及纠错检查 |

只有正反馈时执行正向学习，只有负反馈时只能执行受限负向更新；没有正方向的样本不具有回答内保护信号。`project_only` 没有纠错目标时不进行主动正向更新。记录各分支比例、冲突率、回退和跳过原因，而非只记录训练损失。

## 当前执行范围与实验解释

- 首版以**单条回答形成一个实际参数步**，使用无动量的手工 SGD 以对应方向公式。不是分别训练两个模型后相减。
- LLaVA 后端冻结视觉塔、映射层及原模型，只训练语言模型的 LoRA。各机制对照必须使用同一参数范围。
- batch 平均后投影、跨回答累积以及 Adam/AdamW 的动量、预条件均会改变实际方向；当前解释不能自动迁移过去。
- 局部 mask 限制损失位置，参数更新仍全局作用于共享适配器。选定平均正目标的一阶检查，不保证每个事实、有限步损失或一般能力保持。
- 降低错误片段概率不指定正确替代分布；还可能提前结束、少说或产生新幻觉。固定原前缀的 NLL 改善不能替代自由生成评估。
- 第一阶段先固定回答和 mask 比较 `fixed`、单极性、投影及调制，并匹配步长/更新强度；之后才接入自动反馈噪声和周期刷新。
- 评估需同时报告幻觉、正确事实语义保留、信息覆盖、对象 Recall、长度/EOS、一般能力与总成本。合成 smoke 成功只证明接口和机制可执行。

BLR、Top-K 修订、修订前缀保护及其他扩展均不是本原型的前提。若替换负目标，其梯度符号与纠错检查应按新目标重推。
