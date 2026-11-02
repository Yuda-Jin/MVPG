# MVPG：声明级视觉反馈与回答内更新调制

本分支实现当前 story：图像与问题 → MLLM 生成回答 → DiscoSG 申明分解 → 冻结视觉支持核验 → 原文 token 对齐 → 正负梯度调制 → LLM LoRA 更新。训练不要求外部完整修订回答或响应级偏好对，内部仍使用自动产生的正负伪反馈和校准阈值。

完整方法边界、JSONL 格式、五种更新对照和运行命令见 [声明级视觉反馈说明](mvpg_run/claim_feedback/README.md)。当前已完成离线接口与合成机制验证，真实权重训练和 benchmark 效果尚待验证。

## 目录与入口

| 路径 | 用途 |
|---|---|
| `mvpg_run/claim_feedback/` | 当前训练链路：生成、分解、核验、审计及单回答 SGD 调制 |
| `mvpg_run/llava_inference.py` | 新训练和保留的诊断脚本共用的 LLaVA 加载、prompt 与图像处理 |
| `mvpg_run/fgclip_local.py`、`mvpg_run/fgclip2/` | 冻结 FG-CLIP2 后端及兼容实现 |
| `eval_llava_rlhf_coco/` | 幻觉与能力评估代码、问题、标注及评分汇总 |
| `data/` | 保留的视觉支持、幻觉反馈诊断和校准脚本 |
| `run/` | 当前训练启动、保留的评估及模型准备脚本 |
| `llava_setup/`、`utils/` | patched LLaVA 补丁及运行兼容依赖 |
| `tests/` | 数据对齐、DiscoSG 接口、梯度调制和端到端离线检查 |

旧 MVPG、OPA/OPA-DPO 训练链路及其专用数据准备与配置已清理。旧版本可从 Git 历史恢复；旧 projector 检查点的评估入口仍保留。

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

`llava/` 是本地运行依赖，不纳入本仓库追踪。补丁中的旧训练模块不作为当前分支入口；使用这里的生成、训练及评估脚本。补丁的推理组件需要 `utils/constants.py` 和 `utils/checkpoint_utils.py`，因此这些兼容模块保留。

模型下载与配置工具也保留：`run/prepare_basemodels.sh`、`base_operations/modify_base_model_config.py`。它们默认准备 LLaVA-1.5-7B/13B 和 CLIP，运行前检查路径与存储空间；自备模型可按实际目录配置。设置 `LLAVA_CLIP_DIR` 可指定本地视觉塔。DiscoSG 种子模型、训练好的 refiner 和视觉核验阈值的要求见方法说明，不能用切句或随机初始化替代。

## 幻觉及能力评估

以下代码、脚本及其仓库内问题/标注文件均保留：

- MMHal-Bench：生成、模型评分及结果汇总。
- Object HalBench：对象幻觉生成、评分及结果汇总。
- POPE：random、popular、adversarial 问题集及判别指标。
- AMBER：生成/判别查询、标注和评估。
- LLaVA-Bench qa90 与其他已有问答/能力评估入口。
- `data/` 中的视觉支持、错图、反事实、细粒度幻觉与阈值诊断工具。

`run/eval_all_metrics.sh` 保留 LoRA 评估流程；新训练生成的 `final/` 是 LoRA adapter，按脚本参数设置底模与适配器路径。`run/eval_all_metrics_mvpg.sh` 保留旧 `mm_projector` 检查点评估能力。`run/summarize_eval_results.py` 和评估目录内的汇总脚本继续保留。

图片数据、COCO 标注、部分外部数据和实际权重需要另行准备。GPT 评分需要通过环境变量配置服务与凭据；评估脚本的历史默认路径、服务与参数应在使用前检查。生成成功和弱反馈统计不能替代 benchmark 指标，也不能证明一般能力保持。

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
