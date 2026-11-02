# 云端真实训练调试报告

日期：2026-11-02（北京时间）。结论：**LLaVA-1.5-7B 和 13B 均已完成真实图像上的参数更新、检查点保存、回读和继续生成。**

## 执行位置与版本

- 当前聊天仍绑定本地 MVPG；应用项目列表中未发现这台云主机的连接，采用用户授权的 SSH 直接调试方式。
- 云端代码：`/root/MVPG`，分支 `codex/claim-visual-feedback`。
- 使用本聊天提供的 SSH 连接；登录凭据和连接端口不写入仓库。
- 本轮运行基于 `01a1fc9` 加本报告所在提交的运行时兼容修复；当时运行记录中的 `git_dirty=true` 来自这些尚未提交的修复。
- 环境：Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.34.1、PEFT 0.6.2、Accelerate 0.21.0。
- 显卡报告：NVIDIA GeForce RTX 4090 D，49140 MiB 显存，驱动 595.71.05。容量依据服务器实测，不代表所有同名显卡配置。

## 已修复的兼容问题

1. 云端已安装的 LLaVA 使用带 `past_key_values` 的五项多模态返回接口；仓库运行补丁使用三项接口。现在按函数签名选择调用方式，统一扩展后的 embedding、label、mask，并继续检查完整回答和因果预测位置。不捕获并重试模型内部的 `TypeError`。
2. 同时设置本地 CLIP 的 `mm_vision_tower` 和 `image_checkpoint.image_model`。兼容官方配置缺失后者、或旧配置指向其他机器目录的情况；只修改内存中的配置，不修改模型权重目录。
3. 增加两种多模态接口和 CLIP 配置的回归检查，PEFT 依赖固定到本轮实际验证的 0.6.2。本地完整测试：**127 项通过，另有 16 个子检查通过**。

## 诊断输入的边界

使用 SA1B 的一张非 benchmark 图片：
`/root/autodl-tmp/data/SA1B_5_threshold/sa_1000/sa_1000.png`。

7B 先真实生成一段描述，保存了 74 个生成 token（含特殊 token）。图像是人物壁画，生成内容中的 `hat`、`dress` 可见，`couch` 不可见。人工核对后，将这三个对象的原文字段标为两个正单元、一个负单元，保留精确 token IDs 和字符跨度。对齐后为两个正 token、两个负 token，无冲突、截断或被排除单元。

**这是训练器集成诊断样本，不是自动监督数据构造实验。** 两个底模共享该 7B 生成的固定诊断回答；13B 运行仅验证该底模下的优化与保存接口。不能把它写成 13B 的严格在线采样训练，也不能用这一张图得出幻觉缓解或能力保持结论。

## 训练与保存结果

均使用 BF16 底模、FP32 的语言模型 q/v LoRA、rank=16、alpha=32、dropout=0；冻结视觉塔、投影层和原参数。单回答手工 SGD，`mode=modulated`，lr=0.001，实际更新范数上限 0.01，打开更新后 NLL 审计。

| 底模 | 真实更新/尝试 | 跳过 | 可训练参数 | adapter 文件大小 | 保存后回读 |
|---|---:|---:|---:|---:|---|
| 7B | 3/3 | 0 | 8,388,608 | 33,601,315 bytes | 成功，128 个张量逐项最大差 0 |
| 13B | 3/3 | 0 | 13,107,200 | 52,487,331 bytes | 成功，160 个张量逐项最大差 0 |

两者都进入正负双目标分支，正负梯度非零；实际参数增量非零，纠错方向内积为正，正目标一阶内积非正。LoRA B 矩阵分别有 64/80 个非零张量。回读后重复三次前向，损失完全一致；回读模型均可继续生成描述。

短诊断样本下，7B 单次更新约 0.72–1.11 秒，13B 约 1.12–1.49 秒，包含更新后额外一次 NLL 前向，不包含模型加载。不能据此直接外推长段落训练吞吐。

### 显存测量

| 验证运行 | 测量值 | 口径 |
|---|---:|---|
| 7B BF16，一次更新复核 | 19.58 GiB / 20.10 GiB | PyTorch 峰值 allocated / reserved |
| 13B BF16，三次更新 | 36.21 GiB | nvidia-smi 每 200ms 采样的设备占用峰值 |
| 7B FP32，一次方向诊断 | 33.43 GiB / 34.17 GiB | PyTorch 峰值 allocated / reserved |

因此本服务器的约 48GB 显存在该短样本、单图单回答配置下可运行 7B/13B。两种测量口径不完全相同；更长回答、更多图像或更大 batch 会提高峰值。

### 有限步损失观察

| BF16 三次更新 | 正单元 NLL：前→后 | 负单元 NLL：前→后 |
|---|---:|---:|
| 7B | 1.358966 → 1.359328 | 1.364722 → 1.397854 |
| 13B | 1.555510 → 1.557547 | 2.160890 → 2.114063 |

虽然实际参数方向的一阶检查通过，BF16 的有限步 NLL 仍会波动，13B 本例负 NLL 也未按预期提高。不能把方向检查写成有限步纠错或保持保证。

另用 7B FP32 完成一个同配置诊断步：正 NLL 从 1.354307 降至 1.350259，负 NLL 从 1.408705 升至 1.426629。该检查支持目标符号和实际更新实现正确；后续机制实验应单独控制数值精度、步长，并审核有限步损失和自由生成，不能只看梯度内积。

## 视觉支持组件

FG-CLIP2-base 已在本环境对同一图片完成真实打分，global/short 模式下，hat、dress、couch 的余弦分数分别约为 0.03637、0.07107、0.01689。这里只验证加载和评分接口，未据这三个数拟合阈值，也未声称它们是真值概率。

本轮没有验证 DiscoSG 自动分解与完整自动反馈训练；正式运行仍需实际解析模型、来源锚点审核和独立开发集的阈值校准。benchmark 评估也尚未执行。

## 输出与复现

全部云端诊断产物位于：

```text
/root/autodl-tmp/output/claim_feedback_debug_20261102_996cfd/
  inputs.jsonl
  responses.jsonl
  feedback_diagnostic.jsonl
  train7b/final/
  train13b/final/
  train7b/steps.jsonl
  train13b/steps.jsonl
  reload7b-verification.json
  reload13b-verification.json
  precision-memory-report.json
  visual-support-check.json
  training-summary.json
```

训练命令示例（新输出目录必须不存在）：

```bash
source /root/.config/autodl-cache-env.sh
cd /root/MVPG
/root/miniconda3/envs/mvpg/bin/python -m mvpg_run.claim_feedback train \
  --input /root/autodl-tmp/output/claim_feedback_debug_20261102_996cfd/feedback_diagnostic.jsonl \
  --model-name /root/autodl-tmp/cache/llava-v1.5-7b \
  --output-dir /root/autodl-tmp/output/claim_feedback_new_run \
  --device cuda --precision bf16 --mode modulated \
  --epochs 3 --lr 0.001 --max-update-norm 0.01 --audit-loss-change
```

切换 13B 时替换底模目录并使用新的输出目录。该命令复现诊断输入，不代表正式训练数据配方。
