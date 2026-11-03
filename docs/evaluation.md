# 一个命令评测检查点

在服务器项目目录、已安装模型依赖的Python环境中运行：

```bash
python run/evaluate_checkpoint.py --checkpoint /path/to/train/checkpoint-005000
```

默认从检查点推断底模及**实际成功更新数**，使用
`/root/autodl-tmp/data/eval` 下的评测数据、批量16及全部POPE分组。
裁判凭据、接口和模型从项目根目录被Git忽略的`.env`读取。
需要改变位置时，提供`--model-base`、`--data-root`或`--output-dir`；
也可用环境变量`EVAL_DATA_ROOT`指定数据目录。

底层仍执行现有16个阶段，涵盖MMHal、LLaVA-Bench、POPE三类、AMBER及ObjectHal，
不改变题目、生成参数和评分器。默认结果页只汇报参考论文Table 2的10项指标。

## 看结果

默认输出位于数据盘`output/benchmark_results/<模型>_step<实际步数>_<权重哈希>/`。
不在该服务器环境中时，使用项目的`output/benchmark_results/`。
每个结果目录只保留三个入口：

```text
results.md       # 直接阅读的结果表
results.json     # 10项指标、检查点/裁判、完整性状态
details/         # 原始回答、裁判记录、阶段日志、恢复信息
```

10项指标为MMHal Score/幻觉率、AMBER CHAIR/Cover/Hal/Cog、
ObjectHal CHAIRs/CHAIRi、POPE adversarial Accuracy/Precision。
ObjectHal同时记录采用本地还是裁判对象提取，替代裁判名称也会展示。
完整的其他指标仍保存在`details/summary_*.json`中。

全部16阶段及24,706条回答通过数量、ID顺序和评分文件检查，结果才标记为完整。
失败时保留已经完成的阶段和可用分数，返回非零状态；同一命令可继续未完成部分。
检查点、题目、裁判、批量等设置改变时，会要求使用新的输出目录，避免混用结果。
已有100步旧格式检查点只接受`final/`推断计数；没有状态的旧中间检查点不能借用父目录的最终计数。

## 整理已有评测，无需重跑

```bash
python run/evaluate_checkpoint.py \
  --checkpoint /path/to/train/checkpoint-005000 \
  --import-existing /path/to/old/evaluation \
  --output-dir /root/autodl-tmp/output/benchmark_results/llava7b_step5000
```

这会复制原始评测到新目录的`details/`，校验并生成两个结果文件。
原目录、绝对路径引用和历史对照表保留，不调用模型或裁判，也不重新登记历史实验。
`.env`、Notebook缓存和Python缓存不复制。
整理目录必须和原评测目录分离，且不能包含无关文件。
原结果包含多个`summary_*.json`时，应先选定单独的一次评测目录。

只想重新生成两个结果文件，可使用：

```bash
python run/evaluate_checkpoint.py --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results --refresh-only
```

导入及刷新不需要API密钥；校验所需的问题文件应仍在项目及数据目录中。
旧的`run/eval_all_metrics.sh`保留为内部执行器，也兼容原有调用方式。
