# 当前分支的代码清理

清理前版本为 `44b415d`。旧代码可按具体路径从该提交恢复，未改写历史。

移除的内容：

- `opadpo/`：旧 OPA/OPA-DPO 训练和偏好数据生成。
- `configs/`：上述训练链路专用的 Hydra 配置。
- 旧 MVPG 训练主程序、启动脚本和训练曲线绘图。
- 旧 OPA 数据准备、训练及在线偏好生成启动脚本。
- 无保留入口引用的训练用 `utils` 模块。
- SAM 演示、对应示例图和标签表，上游展示图片/幻灯片，以及旧命令备忘文件。

共移除 56 个已追踪文件。模型权重、图片数据、本地环境和 `output/` 内已有产物未清理。

保留的内容：

- `eval_llava_rlhf_coco/` 中的全部脚本、问题、标注和结果汇总入口。
- `run/eval_all_metrics.sh`、`run/eval_all_metrics_mvpg.sh`、`run/summarize_eval_results.py`。
- 全部 `data/verify_*.py`，视觉支持评分、校准、申明标注及依赖的数据生成工具。
- 已有诊断报告、分数、标签和统计图片；包括历史 RCGR 检查以便比较。
- FG-CLIP2 实现、patched LLaVA 补丁、兼容环境快照和底模配置准备工具。
- `utils/constants.py`、`utils/checkpoint_utils.py` 及包初始化文件：仍是补丁的运行依赖。
- 当前 claim-feedback 训练链路及测试。

依赖拆分：

- 共用模型加载、prompt 和图像预处理移入 `mvpg_run/llava_inference.py`。
  新训练及旧 projector 评估均使用这里；加载后默认冻结全部参数。
- 四个评估文件只改引用路径及注释，保留原逻辑、命令参数和检查点加载能力。
- 原诊断 caption 生成所需的图像分组索引移入 `data/grouped_caption_sources.py`，
  保留原详细描述提示、负样本池与图像排序语义。
- 评估启动脚本通过环境变量读取凭据，移除硬编码密钥回退；不改写 Git 历史中的旧值。

验证包括完整单元/离线集成测试、保留文件逐项存在性核对、四个评估文件的 AST 等价检查、
全部当前 Python 源码解析和本地引用存在性检查。真实模型和外部评分服务未运行。
