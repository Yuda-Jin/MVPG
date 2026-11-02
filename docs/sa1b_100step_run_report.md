# SA1B / LLaVA-1.5-7B：100 次实际更新与完整评测

日期：2026-11-02（Asia/Shanghai）。执行状态：100 次实际训练更新与完整 benchmark 评测均已完成。

评测于当日 20:37:48–21:58:31 执行，耗时约 80 分 43 秒；16 个阶段全部成功，退出码 0。
最终校验共 24,706 条回答，未将部分输出或缺失评分当成完成；任务结束后 GPU 已空闲。

## 已确认的配置

- 只运行 LLaVA-1.5-7B；不运行 13B。
- 云端运行目录：`/root/autodl-tmp/output/claim_feedback_sa1b_7b_100step_20261102`。
- 数据目录：`/root/autodl-tmp/data/SA1B_5_threshold`，找到 4,848 张原图，未将四个裁剪图当成独立原图。
- 用 Python seed 42 打乱排序后的原图路径：先独立留出 16 张校准图，再取 256 张生成训练回答。
- 256 条回答已生成并验证格式；温度 0.7、top-p 0.9、最多 192 个新 token，保留原始生成 token IDs。
- 没有使用旧的 `model_captions.json`、`neg_captions.json`，也没有重复上一轮单图人工诊断样本来凑步数。

## 自动反馈与监督边界

完整链路使用官方 DiscoSG seed → 三轮 delete/insert/refine → 唯一来源对象投影 → 冻结 FG-CLIP2 → 双阈值弃权 → token 对齐 → 回答内正负目标调制。

当前实现只训练对象存在性单元；属性、关系、重复提及、否定、猜测及无法可靠定位的片段会保留为未决。视觉相似度是弱反馈，不能证明对象不存在。

校准图上保存了 128 个对象探针（64 present、64 absent），标签来自本次助手对独立图像的视觉检查，不是人工真值。它们不参与训练。阈值是探索性的：negative = -0.0056416427，positive = 0.0576395076；选择规则、逐项分数和标签来源均保存在 `data/calibration.json`。这不是经过代表性人工开发集验证的幻觉分类器，不能把本次运行作为最终方法效果结论。

## 模型来源

使用的来源均固定版本，并对大文件核对 SHA256：

| 组件 | 来源 / 固定版本 |
|---|---|
| DiscoSG 源码 | [官方仓库](https://github.com/ShaoqLin/DiscoSG)，`d2be2768ddc80199ab56f38cf50c132351a8d248` |
| seed | [VG factual SG](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg)，`b7923dae0e6f924fe37957a1cee920dac1d50a7f` |
| refiner T5 / classifier | [官方 t5-only 检查点](https://huggingface.co/sqlinn/DiscoSG-Refiner-Large-t5-only)，`afd62c2e9ede1078322ae87e2a9eb277220bf45f` |
| refiner 元数据 / tokenizer | [官方 Large 检查点](https://huggingface.co/sqlinn/DiscoSG-Refiner-Large)，`6652bad9f844f10c0df96fdf3e762a1baa0676d5` |

完整 Large 的权重封装与当前官方加载器不同，因此使用兼容的 t5-only 权重，并组装官方 tokenizer 和元数据。`model_info.json` 只将 tokenizer 路径改为本地目录，保留上游文件。源码由固定提交的 ZIP 取得，额外保存 archive SHA256；它没有伪造为 Git checkout。

初次分块下载遇到镜像 HTTP 错误，自动链路在训练前明确失败；保留旧状态并恢复下载，未使用不完整权重。
随后两个大权重及下载文件的完整哈希复验通过。

真实运行发现 refiner 的新版 tokenizer JSON 不兼容当前 tokenizers 0.14.1。
已从官方 SentencePiece 重建 fast tokenizer，保存到独立的 `refiner-tokenizer-transformers434`，
并在独立 `refiner-transformers434` 配置中引用它；原始权重、SentencePiece 与官方源码均未改动，元数据路径修改另有记录。
完整 32,100 个词表 ID 与上游一致，256/256 条实际输出的分词与 SentencePiece 对照一致，字符偏移接口通过。
兼容性记录保存在 `data/discosg_tokenizer_compatibility.json`。

## 训练协议

目标为 **100 次实际成功更新**，跳过回答不计数。使用 `--max-updates 100`，而不是把 `--max-steps 100` 当成成功更新数。

- 固定自动反馈池，最多 3 轮 / 768 次尝试；若未达到目标，保存部分 adapter、标记 `incomplete` 并拒绝进入正式评测。
- `modulated`，LLM q/v LoRA，rank 16、alpha 32，基础模型 BF16、可训练 adapter FP32。
- 手动 SGD，lr 0.001，实际更新范数上限 0.001，seed 42。
- 保存 token 所有权、正负目标、冲突及回退信息；额外记录更新后 NLL。
- 在池内完全没有正或负反馈时停止，避免将空调制实验冒充完整运行。

正式训练完成 **109 次尝试、100 次实际更新、9 次跳过，100 张不同图片**，没有跨轮重复图片。
训练进程耗时 121.19 秒，其中逐步计算合计 69.73 秒，其余主要为加载与保存。
97 步为正反馈单独更新，3 步同时使用正负反馈；3 步均没有触发梯度冲突，因此这轮没有实际检验冲突投影分支的效果。

分解共得到 2,731 个对象候选。实际检查发现背景、场景、区域等独立元场景词被误当作对象，
代码已修正为未决并对保存的图重新投影，无需重跑模型；304 个此类候选不参与核验和训练。
初次训练启动后的 3 步已经中止并隔离到 `train_rejected_nonvisual_feedback`，正式训练从原始底模重新开始，未继承它们。
最终反馈池为 760 个正标签、6 个弱负标签、1,965 个未决候选。
正式训练读取 `data/feedback_object_only.jsonl`，检查点为 `train/final`。

损失审计体现了限制：100 步均满足记录的一阶正向方向约束，3 个负反馈步均满足负向方向约束；
但实际更新后的正向 NLL 有 59 步下降、41 步上升，负向 NLL 只有 1/3 步上升。
实际更新范数范围为 0.0000049597–0.001。这表明方向约束不能被写成有限步长下的性能保持保证，
BF16 数值影响和有限步长效应需要后续对照，当前不做单一归因。

训练数据抽查还发现 `sa_6549` 的背景中可见树，自动核验却将 `trees` 标为弱负例。
这项抽查没有用来人工修改训练标签；它进一步说明 global cosine 的低分不能直接解释为物体不存在。
本轮负反馈少且含噪声，适合作为流程验证，不足以证明幻觉改善或通用能力保持。

## 评测协议与准备检查

| benchmark | 本轮完整数量 | 评分 |
|---|---:|---|
| MMHal-Bench | 96 | DeepSeek 替代裁判，原 MMHal 评分提示 |
| LLaVA-Bench qa90 | 90 | DeepSeek 替代裁判，与原基线回答比较 |
| POPE | random / popular / adversarial 各 3,000 | Accuracy / Precision / Recall / F1 / Yes ratio |
| AMBER | 15,220（生成 1,004；判别 14,216） | CHAIR / Cover / Hal / Cog；判别总分及分维度指标 |
| Object HalBench | 300 | 本地 CHAIR 与 DeepSeek 辅助对象提取两套评分 |

合计 24,706 条问答。云端图片已逐项检查：以上图片全部存在。NLTK 资源、`en_core_web_lg` 和 `en_core_web_trf` 的真实加载及处理检查通过。

最终结果：

| benchmark | 完成数量 | 结果 |
|---|---:|---|
| MMHal-Bench | 96/96 | 平均分 2.42708/6；幻觉 55/96 = 57.2917% |
| LLaVA-Bench qa90 | 90/90 | 相对分数 all 77.3；complex 85.5、conv 84.5、detail 61.2 |
| POPE random | 3,000/3,000 | Accuracy 88.1333%、Precision 97.3510%、Recall 78.4000%、F1 86.8538%、Yes 40.2667% |
| POPE popular | 3,000/3,000 | Accuracy 86.9000%、Precision 94.4578%、Recall 78.4000%、F1 85.6831%、Yes 41.5000% |
| POPE adversarial | 3,000/3,000 | Accuracy 85.0667%、Precision 90.4615%、Recall 78.4000%、F1 84.0000%、Yes 43.3333% |
| AMBER 全量 | 15,220/15,220 | 生成式 CHAIR 7.7%、Cover 51.4%、Hal 33.8%、Cog 4.2%；判别式 Accuracy 81.2%、F1 85.9% |
| Object HalBench | 300/300 | DeepSeek 辅助提取：CHAIRs 47.33%、CHAIRi 25.83%、CHAIRs_refine 48.97% |

Object HalBench 对同一组 300 条回答还计算了本地对象提取版本：CHAIRs 52.00%、CHAIRi 16.06%、CHAIRs_refine 52.17%。
两套分数使用不同对象提取方式，应分开报告，不能把它们的差异归因于模型训练。
其中 CHAIRs_refine 的分母是含有可映射 COCO 对象的回答数，本地为 299、DeepSeek 为 290；这不是缺失图片或标注。

AMBER 使用官方严格 Yes/No 解析，14,216 条判别回答中无格式无效项。

| AMBER 判别维度 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 总体 | 81.2 | 85.3 | 86.5 | 85.9 |
| 存在性 | 90.2 | 100.0 | 90.2 | 94.8 |
| 属性总体 | 79.9 | 80.3 | 79.4 | 79.8 |
| 状态 | 77.7 | 78.2 | 76.8 | 77.5 |
| 数量 | 82.9 | 82.5 | 83.5 | 83.0 |
| 动作 | 85.7 | 86.8 | 84.3 | 85.5 |
| 关系 | 60.2 | 51.0 | 99.4 | 67.4 |

表中 AMBER 判别指标均为百分数，沿用仓库中官方评分定义。

仓库原 POPE random 只有 2,910 条，且另外两组有旧拼写错误；三组均恢复为[官方当前问题集](https://github.com/RUCAIBox/POPE/tree/main/output/coco)，记录下载内容 SHA256，未拼接缺失问题或更改标签来凑数。

裁判固定 `deepseek-flash`、关闭 thinking。真实 API 请求已成功返回，凭据仅保存在被 Git 忽略的 `.env`，云端权限 `0600`。结果必须标为 DeepSeek 裁判，不能直接和原 GPT-4 数值比较。

批量推理使用等长、无 padding 的输入并保持原始问题顺序。7B 技术探针测得 batch 16 峰值 allocated 23.74 GiB；16 条中有一条与逐条生成不同。重复图、混合图、置换图探针排除了明显行索引错误：批内相同图 logits 完全一致，置换后也一致；单条与批量的最大 logit 差为 0.625，足以改变该条 Yes/No 首选。因此本轮批量设置会明确记录，后续底模对照必须使用同一精度与批量协议，不能把不同推理设置造成的变化归因于训练。

qa90、POPE、AMBER 使用 batch 16；MMHal 和 Object HalBench 保留逐条推理。

每阶段独立保存日志和状态；完整回答按数量、顺序、ID 和来源检查后才能复用。失败阶段继续执行其他 benchmark，但汇总不会把缺失评分标成完成。

## 本地代码验证

- 已提交实际更新计数支持：`6114b25`。
- 已提交完整评测、批量支持、官方 POPE 恢复、可配置裁判及续跑：`de366b1`。
- 元场景词过滤修复：`f151087`；正式训练清单记录该提交，工作区干净。
- 最终本地全套测试：176 passed，另 16 个 subtests passed；1 个环境自带 anyio 插件弃用警告。
- 云端评测 Bash 语法、官方 DiscoSG 模块导入、评分依赖与真实 DeepSeek 接口检查通过。
- AMBER 缓存仅复用相同输入的完整 spaCy Doc，保留原相似度算法；64 对真实词语的差分检查完全一致。
- 本地检查点留档验证：128 个 adapter 张量，共 8,388,608 参数，全部有限；64 个 LoRA B 张量均非零。

## 交付与解释边界

云端 `evaluation/summary_sa1b7b_modulated_100step_seed42_deepseekflash.json` 和 `evaluation/summary.csv`
汇总全部指标，`final_audit.json` 核对样本数、16 个阶段状态、适配器哈希和运行配置。
MMHal 汇总脚本按惯例只打印两位小数；本文的 57.2917% 根据 96 条完整评分中的 55 条幻觉重新计算。

本地留档在 `output/remote_sa1b_100step_20261102/`：

- `training-evidence.zip`：最终 LoRA、训练清单/逐步日志、固定反馈池、校准来源和运行配置。
- `evaluation-evidence.zip`：完整回答、逐项评分、阶段日志与最终汇总。

当前已跑通完整自动反馈链路和全套评测，尚未运行同协议的未训练底模对照。
结合负反馈稀少、已有误报、没有冲突投影触发和有限步 NLL 审计结果，本轮不支持宣称幻觉改善或通用能力保持。
后续比较必须固定裁判、问题集、推理精度、批量设置、反馈池与训练预算。
