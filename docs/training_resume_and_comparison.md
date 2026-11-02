# 续训、指标记录与实验对照

## 检查点与继续训练

默认每 **500 次成功的参数更新**保存一次 `checkpoint-000500/`、`checkpoint-001000/`；
被跳过的回答不计数，也不会反复保存同一个检查点。训练结束始终保存 `final/`，即使不足500步。
这些目录仍是标准的扁平 LoRA 目录，可直接作为现有评测器的适配器路径。

新增检查点同时包含 `trainer_state.json`、`rng_state.pt`、`history.jsonl`：
模型参数、全局成功/尝试/跳过计数、下一条数据位置、Python/Torch/CUDA/NumPy随机状态、
固定反馈池及文件哈希、目标设置都与同一个快照绑定。当前使用无动量手动SGD，没有另需恢复的Adam或调度器状态。
保存动作不改变训练随机流；失败或中断的未完成更新不会被发布为可恢复快照。

续训写入**新目录**，原来的检查点、训练记录和评测结果保留。
`--max-updates`、`--max-steps`、`--epochs` 都是从实验起点累计的总预算。
例如从500步恢复到1000步，实际再更新500次，不会重置为0，也不会回放已经消费的数据。

```bash
python -m mvpg_run.claim_feedback train \
  --model-name /root/autodl-tmp/cache/llava-v1.5-7b \
  --input /path/to/feedback.jsonl \
  --resume-from /path/to/run/checkpoint-000500 \
  --output-dir /root/autodl-tmp/output/continue_to_1000 \
  --epochs 6 --max-updates 1000 --tensorboard
```

目标、学习率、反馈池、底模、精度及LoRA结构不匹配时会拒绝真正的断点恢复。
改变这些条件开展新实验应使用已有的 `--adapter-path` 热启动方式，它会开启新计数和新数据游标。
输出频率和总预算可以在续训时调整；检查点频率默认使用当前的500步配置。

### 已有100步结果

之前的100步 `final/` 只有适配器和父目录训练日志，没有随机状态与游标快照。
保留它作为已评测的原始结果。若要继续它，需要显式使用 `--allow-legacy-resume`：
程序核对完整日志与反馈池，推断全局步数及下一条数据，并记录 `legacy_inferred_no_rng`，
不会把这种恢复标成保存过完整随机状态的精确恢复。

```bash
python -m mvpg_run.claim_feedback train \
  --model-name /root/autodl-tmp/cache/llava-v1.5-7b \
  --input /root/autodl-tmp/output/claim_feedback_sa1b_7b_100step_20261102/data/feedback_object_only.jsonl \
  --resume-from /root/autodl-tmp/output/claim_feedback_sa1b_7b_100step_20261102/train/final \
  --allow-legacy-resume \
  --output-dir /root/autodl-tmp/output/claim_feedback_continue_to_1000 \
  --epochs 6 --max-updates 1000 --tensorboard
```

旧版本按“尝试次数”命名的中间检查点不支持这种推断恢复，避免把25次尝试的参数错配到100步游标。
一旦新续训保存了完整状态，之后从新检查点恢复便不再需要旧格式选项。

## 训练时记录

每次尝试都会追加：

- `steps.jsonl`：原始token对齐、反馈单元及完整更新诊断。
- `metrics.jsonl`：方便画曲线的扁平指标，包含全局成功步数、尝试数、正负NLL、token数、梯度范数、
  实际更新范数、冲突/投影、方向约束、系数、耗时；CUDA训练还记录显存。
- `training_metrics_summary.json`：累计分支覆盖、跳过数、冲突及实际NLL变化统计。
- `manifest.json`：持续更新计数、游标、来源和运行状态。

`--log-every 10` 默认每10次尝试输出进度；`--tensorboard` 额外写入 `tensorboard/` 事件文件。
若环境未安装该可选依赖，运行 `python -m pip install tensorboard`。
可以用 `tensorboard --logdir /path/to/run/tensorboard` 查看。
续训会继承检查点封存的历史日志，并用累计步数衔接指标；不会从原实验可能损坏或多写的日志尾部恢复。
这些是训练诊断，不能作为幻觉准确率或能力保持指标。

## 实验对照表

`output/experiment_comparison/comparison.md` 是人读表格，`registry.json` 保存完整数值和来源。
论文固定数据位于受Git管理的 `run/reference_results/opa_dpo.json`，而输出表由程序维护。

表格已纳入所给PDF Table 2的18行、7B/13B及Qwen分组，以及本次7B/100步实测。
未报告的F1、qa90或AMBER判别指标留空；不会补0或推算。论文原报与本项目实测分组，
保留论文引用/作者复测标记及裁判差异，不据此直接计算效果提升。

`run/eval_all_metrics.sh` 结束后自动登记评测摘要；检查点自身状态优先于父运行清单，
因此 `checkpoint-000500` 不会被父目录后来1000步的计数误标。
每个检查点应使用独立的 `MODEL_SUFFIX`，如 `claim7b_step500`。

也可手动登记：

```bash
python run/compare_experiments.py \
  --summary /path/to/evaluation/summary_claim7b_step500.json \
  --run-id claim7b_step500 --method claim_feedback/modulated \
  --model llava-v1.5-7b --judge deepseek-flash \
  --training-manifest /path/to/checkpoint-000500/trainer_state.json \
  --output-dir output/experiment_comparison
```

相同ID用于更新同一实验，原有其他实验会保留。裁判或协议变化需要新ID；
部分评分不会静默覆盖完整记录。训练反馈来源与评测裁判分开登记。
