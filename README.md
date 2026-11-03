# MVPG：声明级视觉反馈与回答内更新调制

本分支实现当前 story：图像与问题 → MLLM 生成回答 → DiscoSG 申明分解 → 冻结视觉支持核验 → 原文 token 对齐 → 正负梯度调制 → LLM LoRA 更新。训练不要求外部完整修订回答或响应级偏好对，内部仍使用自动产生的正负伪反馈和校准阈值。

完整方法边界、JSONL 格式、五种更新对照和运行命令见 [声明级视觉反馈说明](mvpg_run/claim_feedback/README.md)。

7B/13B 的真实训练、检查点回读及显存诊断已通过，具体输入边界、数值限制和复现路径见 [云端训练调试报告](docs/server_training_report.md)。7B 的100步与5000步全套评测已完成，结果见 [5000步评测报告](docs/sa1b_5000step_evaluation_report.md)；当前结果呈混合变化，尚无同协议未训练底模对照。

## 目录与入口

| 路径 | 用途 |
|---|---|
| `mvpg_run/claim_feedback/` | 当前训练链路：生成、分解、核验、审计及单回答 SGD 调制 |
| `mvpg_run/llava_inference.py` | 当前训练和 LoRA 评估共用的 LLaVA 加载、prompt 与图像处理 |
| `mvpg_run/fgclip_local.py`、`mvpg_run/fgclip2/` | 冻结 FG-CLIP2 后端及兼容实现 |
| `eval_llava_rlhf_coco/` | 幻觉与能力评估代码、问题、标注及评分汇总 |
| `run/` | 当前训练启动、保留的评估及模型准备脚本 |
| `llava_setup/`、`utils/` | patched LLaVA 补丁及运行兼容依赖 |
| `tests/` | 数据对齐、DiscoSG 接口、梯度调制和端到端离线检查 |

旧 MVPG、OPA/OPA-DPO 训练链路，以及旧方案的分块、RCGR、子句/句子奖励实验、校准、负样本生成和历史结果已清理。旧版本可从 Git 历史恢复。当前方案的五种更新对照属于本方案的机制实验，继续保留。

## 快速检查

在已安装 PyTorch 和 pytest 的环境中，从仓库根目录执行：

```bash
python -m mvpg_run.claim_feedback --help
python -m mvpg_run.claim_feedback smoke --output-dir output/claim_feedback_smoke
python -m pytest -q tests
```

`smoke` 只使用合成小模型和人工构造反馈，不读取真实图像，不构成幻觉缓解效果实验。输出目录必须是新的目录。

## 真实模型环境

真实 LLaVA 和保留的评估流程沿用 Linux patched LLaVA 环境。`environment.yaml` 是上游环境的兼容快照，包含较旧版本和历史依赖；保留它便于复现，不代表其中每个包都是新训练的必要依赖。不要直接用它覆盖当前正在使用的环境。

```bash
conda env create -f environment.yaml
conda activate OPA_DPO
pip install -r requirements-claim-feedback.txt
```

patched LLaVA 基于上游提交 `817a4af`。以下步骤用于尚未准备源码的新环境：

```bash
git clone https://github.com/haotian-liu/LLaVA llava_setup/LLaVA
git -C llava_setup/LLaVA checkout 817a4af
(cd llava_setup && patch -p1 < llava_modifications.patch)
cp -R llava_setup/LLaVA/llava ./llava
```

`llava/` 是本地运行依赖，不纳入本仓库追踪。补丁只保留当前运行所需的模型、视觉、prompt 和推理工具变更；旧训练、演示服务与历史评估结果补丁已移除。推理组件需要 `utils/constants.py` 和 `utils/checkpoint_utils.py`，因此这些兼容模块保留。

模型下载与配置工具也保留：`run/prepare_basemodels.sh`、`base_operations/modify_base_model_config.py`。它们默认准备 LLaVA-1.5-7B/13B 和 CLIP，运行前检查路径与存储空间；自备模型可按实际目录配置。设置 `LLAVA_CLIP_DIR` 可指定本地视觉塔。DiscoSG 种子模型、训练好的 refiner 和视觉核验阈值的要求见方法说明，不能用切句或随机初始化替代。

## 幻觉及能力评估

断点续训、默认每500次成功更新保存检查点、逐步指标/TensorBoard与自动实验对照表，见
[续训与实验对照说明](docs/training_resume_and_comparison.md)。

以下代码、脚本及其仓库内问题/标注文件均保留：

- MMHal-Bench：生成、模型评分及结果汇总。
- Object HalBench：对象幻觉生成、评分及结果汇总。
- POPE：random、popular、adversarial 问题集及判别指标。
- AMBER：生成/判别查询、标注和评估。
- LLaVA-Bench qa90 与其他已有问答/能力评估入口。

指定检查点即可运行当前的全套16阶段评测：

```bash
python run/evaluate_checkpoint.py --checkpoint /path/to/train/checkpoint-005000
```

入口自动读取底模和实际更新步数，复用被Git忽略的`.env`评分配置。当前云端默认数据目录为`/root/autodl-tmp/data/eval`；其他位置使用`--data-root`指定。
输出目录只需查看`results.md`（成绩）、`results.json`（程序读取）和`details/`（回答、评分和日志）。重复运行相同命令会校验并复用完整阶段。
完整参数、已有结果导入及恢复说明见 [评测说明](docs/evaluation.md)。数值精度及批量设置应在底模与各实验间保持一致；替代裁判的分数不能直接当作原GPT-4结果比较。

`run/eval_all_metrics.sh`保留为内部执行器；评分器、原始回答及来源校验继续保留在明细目录中。

图片数据、COCO 标注、部分外部数据和实际权重需要另行准备。GPT 评分通过环境变量配置服务与凭据；运行前检查路径与参数。原始图片和本地权重作为运行资源保留，不是当前分支追踪的实验代码。生成成功和弱反馈统计不能替代 benchmark 指标，也不能证明一般能力保持。

## 来源与许可

本仓库基于 Microsoft [OPA-DPO](https://github.com/microsoft/OPA-DPO)，保留其 [MIT 许可](LICENSE)。同时使用或适配 LLaVA、LLaVA-RLHF 和 FG-CLIP2 等项目的代码与接口；外部代码、模型及数据遵循各自许可。删除旧训练入口不改变保留代码的来源归属。

上游 OPA-DPO 论文引用：

```bibtex
@article{yang2025opadpo,
  title={Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key},
  author={Yang, Zhihe and Luo, Xufang and Han, Dongqi and Xu, Yunjian and Li, Dongsheng},
  journal={arXiv preprint arXiv:2501.09695},
  year={2025}
}
```
