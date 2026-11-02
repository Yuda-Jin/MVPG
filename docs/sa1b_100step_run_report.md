# SA1B / LLaVA-1.5-7B：100 次实际更新与完整评测

日期：2026-11-02（Asia/Shanghai）。执行状态：真实 DiscoSG 分解中，训练及最终评分尚未完成。

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
并在独立 `refiner-transformers434` 配置中引用它；原始模型文件与官方源码均未改动。
完整 32,100 个词表 ID 与上游一致，256/256 条实际输出的分词与 SentencePiece 对照一致，字符偏移接口通过。
兼容性记录保存在 `data/discosg_tokenizer_compatibility.json`。

## 训练协议

目标为 **100 次实际成功更新**，跳过回答不计数。使用 `--max-updates 100`，而不是把 `--max-steps 100` 当成成功更新数。

- 固定自动反馈池，最多 3 轮 / 768 次尝试；若未达到目标，保存部分 adapter、标记 `incomplete` 并拒绝进入正式评测。
- `modulated`，LLM q/v LoRA，rank 16、alpha 32，基础模型 BF16、可训练 adapter FP32。
- 手动 SGD，lr 0.001，实际更新范数上限 0.001，seed 42。
- 保存 token 所有权、正负目标、冲突及回退信息；额外记录更新后 NLL。
- 在池内完全没有正或负反馈时停止，避免将空调制实验冒充完整运行。

实际更新次数、唯一图像数、正负覆盖及训练耗时待训练结束后填写。

## 评测协议与准备检查

| benchmark | 本轮完整数量 | 评分 |
|---|---:|---|
| MMHal-Bench | 96 | DeepSeek 替代裁判，原 MMHal 评分提示 |
| LLaVA-Bench qa90 | 90 | DeepSeek 替代裁判，与原基线回答比较 |
| POPE | random / popular / adversarial 各 3,000 | Accuracy / Precision / Recall / F1 / Yes ratio |
| AMBER | 15,220（生成 1,004；判别 14,216） | CHAIR / Cover / Hal / Cog；判别总分及分维度指标 |
| Object HalBench | 300 | 本地 CHAIR 与 DeepSeek 辅助对象提取两套评分 |

合计 24,706 条问答。云端图片已逐项检查：以上图片全部存在。NLTK 资源、`en_core_web_lg` 和 `en_core_web_trf` 的真实加载及处理检查通过。

仓库原 POPE random 只有 2,910 条，且另外两组有旧拼写错误；三组均恢复为[官方当前问题集](https://github.com/RUCAIBox/POPE/tree/main/output/coco)，记录下载内容 SHA256，未拼接缺失问题或更改标签来凑数。

裁判固定 `deepseek-flash`、关闭 thinking。真实 API 请求已成功返回，凭据仅保存在被 Git 忽略的 `.env`，云端权限 `0600`。结果必须标为 DeepSeek 裁判，不能直接和原 GPT-4 数值比较。

批量推理使用等长、无 padding 的输入并保持原始问题顺序。7B 技术探针测得 batch 16 峰值 allocated 23.74 GiB；16 条中有一条与逐条生成不同。重复图、混合图、置换图探针排除了明显行索引错误：批内相同图 logits 完全一致，置换后也一致；单条与批量的最大 logit 差为 0.625，足以改变该条 Yes/No 首选。因此本轮批量设置会明确记录，后续底模对照必须使用同一精度与批量协议，不能把不同推理设置造成的变化归因于训练。

每阶段独立保存日志和状态；完整回答按数量、顺序、ID 和来源检查后才能复用。失败阶段继续执行其他 benchmark，但汇总不会把缺失评分标成完成。

## 本地代码验证

- 已提交实际更新计数支持：`6114b25`。
- 已提交完整评测、批量支持、官方 POPE 恢复、可配置裁判及续跑：`de366b1`。
- 本地测试：150 passed，另 16 个 subtests passed；1 个环境自带 anyio 插件弃用警告。
- 云端评测 Bash 语法、官方 DiscoSG 模块导入、评分依赖与真实 DeepSeek 接口检查通过。

最终训练统计与 benchmark 分数尚待执行完成后更新，当前不报告性能提升或能力保持结论。
