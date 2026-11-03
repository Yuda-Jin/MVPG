# COCO 1000 图声明接地性校准

本实验从 `C:/data/coco_raw` 本地可用的 10000 张 COCO val2014 图片中，
用 seed 42 均匀随机抽取 1000 张。它不是从完整 40504 张验证图中直接抽样。
图片先与官方 ZIP 中央目录的文件名、大小和 CRC 核对，再记录内容 SHA256。
生成之前按图像内容冻结 400 张开发图和 600 张测试图；重复图不能跨组。
图片仅用于工具校准，不作模型训练，已知评测图片重合记录在抽样清单中。

## 自动流程

原始 LLaVA-v1.5-7B 生成描述，DiscoSG 保持三轮细化，保留完整回答、token IDs、
场景图和解析异常。评分子集仍为可精确映射回生成文本的肯定对象存在性候选，
不把该子集解释成已覆盖全部属性、关系、数量或 OCR 幻觉。

Grounding DINO 使用官方 `IDEA-Research/grounding-dino-tiny` HF 权重，
独立 Python 环境运行，避免升级 LLaVA / DiscoSG 的依赖。输入是独立的原对象
短语，小写并补末尾句点；所有 decoder query 参与评分，不先按 box/text 阈值或
NMS 筛选。主分数 `dino_max` 是所有 query 与目标内容 token 的 sigmoid 匹配分数
最大值；排除特殊符号、句点与 padding。另存 `dino_mean` / `dino_min`，用于分析
复合词被部分匹配的问题，不能根据测试结果悄悄替换主分数。

每条候选保留实际提示、token、最高框及前五个候选框。低检测分数不直接等于
不存在，也不直接生成负反馈标签。

外部视觉裁判与上一轮一致，实际接收图片并独立给出支持、不支持、不确定及依据；
它看不到检测或 CLIP 分数，输出是银标而非人工 GT。本轮从头使用请求内短编号，
严格映射回原始声明 ID，失败时只重试失败图片，最多三轮。

阈值只在开发图上选择，再冻结到测试图。报告最大 F1 工作点，以及目标 95% 精度、
至少 30 条样本的保守工作点；无法满足时报告没有可用阈值，不能放宽条件凑负例。
在同一声明、划分和裁判标签上自动补跑 FG-CLIP2 全图原短语分数，避免混淆
更换数据集与更换核验工具的效果。SA-1B 与 COCO 的跨数据差异只作描述性比较。

## 运行与续跑

抽样与打包：

```powershell
python run/prepare_coco_grounding_sample.py --help
```

云端在 `.env` 配置完成、图片及模型已准备后启动：

```bash
python -u run/run_coco_grounding_calibration.py \
  --run /root/autodl-tmp/output/grounding_calibration/llava7b_coco1000_dino_20261103 \
  --score-python /root/autodl-tmp/envs/grounding-dino/bin/python \
  --dino-root /root/autodl-tmp/cache/grounding-dino-tiny \
  --env-file /root/MVPG/.env
```

以后只传相同 `--run` 即可自动校验并续跑。生成按批、解析按图、DINO 按完整图片
保存原子分片，输入、模型、协议或运行环境改变时拒绝混用已有结果。

根目录只看 `results.md` / `results.json`；运行状态在 `details/status.json`，
原始记录、框、裁判日志和图表在 `details/`。同批 FG-CLIP 结果在
`details/fgclip_baseline/results.md`。所有缓存、图片、权重和实验输出不纳入 Git。
