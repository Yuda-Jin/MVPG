#!/usr/bin/env python3
"""MVPG 训练指标可视化（读取 run_mvpg.py 生成的 train_log.csv）。

指标（不含 adv_mean，恒 0 无信息量）：
  step, gid, loss, r_mean, r_min, r_max, r_std, r_main, kl
  - loss   ：REINFORCE surrogate = -mean(advantage_norm·logprob)；advantage 已组内
             归一化（除以 std），loss 数值尺度与旧版不可直接比较；不单调，仅作参考
  - r_mean ：组内平均 RCGR 奖励
  - r_min  ：组内最差区域奖励
  - r_max  ：组内最好区域奖励
  - r_std  ：组内奖励标准差 = advantage 归一化分母的平均水平；持续收缩 = 输出趋同
            （探索度衰减/熵坍缩前兆），配合 r_mean 判断：高 r_mean + 低 r_std 为良性
            收敛，低 r_mean + 低 r_std 为模式坍缩
  - r_main ：主图行竞争奖励（把主图当作组内一个视角，与 r_mean/r_min/r_max 同口径、
            含跨区域竞争项），落在同一尺度区间内，可直接比较；不单独参与优化（主图行
            本身已随组内其他视角一起进梯度），作为主图视角的进度信号
  - kl     ：主图行对原模型（训练前 mm_projector）的 KL 散度样本估计
            = Σ_t (log π_θ - log π_ref)，期望 >= 0，单样本可正可负；偏离参考模型
            的程度，配合 reward 判断是否被 KL 压死（kl 大而 reward 不涨 = kl_weight 过大）

用法：
    python mvpg_run/plot_train_log.py --log <output_dir>/train_log.csv [--out fig.png]
    python mvpg_run/plot_train_log.py --log <output_dir>   # 目录下自动找 train_log.csv
  --log 也可缺省时默认 ./train_log.csv。

也提供可导入接口：
    from mvpg_run.plot_train_log import plot_train_log
    plot_train_log("/path/to/train_log.csv", out_png="/tmp/curves.png", window=50)
训练中可随时重复运行（读已完成的行），实现“实时”查看。
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

# 绘制指标列及英文标签（matplotlib 默认字体无 CJK 字形，避免中文变方块）
_METRIC_CN = {
    "loss": "loss (REINFORCE surrogate)",
    "r_mean": "r_mean (group avg)",
    "r_min": "r_min (worst in group)",
    "r_max": "r_max (best in group)",
    "r_main": "r_main (main-image margin, no grad)",
}


def load_log(log_csv):
    """读 CSV -> (steps: np.ndarray, metrics: dict[str, np.ndarray])。"""
    with open(log_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None, {}
    header = rows[0]
    float_cols = {name: i for i, name in enumerate(header) if name not in ("", "gid")}
    data = [r for r in rows[1:] if len(r) == len(header)]
    steps = np.array([float(r[float_cols["step"]]) for r in data])
    metrics = {
        name: np.array([float(r[i]) for r in data])
        for name, i in float_cols.items()
        if name != "step"
    }
    return steps, metrics


def _smooth(y, window):
    """等长居中滑动平均（边界处除以窗口内实际覆盖数）。window<2 时不平滑。"""
    y = np.asarray(y, dtype=float)
    if window < 2 or len(y) < 2:
        return y
    half = (int(window) - 1) // 2  # 居中窗口左右各取 half 个样本
    n = len(y)
    csum = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half, n - 1)
    return (csum[hi + 1] - csum[lo]) / (hi - lo + 1)


def plot_train_log(log_csv, out_png=None, window=50):
    """绘制训练曲线。

    log_csv : train_log.csv 路径（或含它的输出目录）
    out_png : 输出图片路径；None 时与 csv 同名 .png
    window  : 平滑窗口（步数），>=2 生效
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_csv = Path(log_csv)
    if log_csv.is_dir():
        log_csv = log_csv / "train_log.csv"
    if not log_csv.is_file():
        raise FileNotFoundError(f"找不到指标日志: {log_csv}")

    steps, metrics = load_log(log_csv)
    if steps is None or len(steps) == 0:
        print(f"[plot_train_log] {log_csv} 暂无数据行")
        return

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(11, 13), sharex=True, constrained_layout=True
    )

    # 上：loss
    loss = metrics["loss"]
    ax1.plot(steps, loss, alpha=0.35, lw=0.8, color="#d62728", label="raw")
    ax1.plot(steps, _smooth(loss, window), lw=1.8, color="#d62728", label=f"smooth({window})")
    ax1.set_ylabel(_METRIC_CN["loss"])
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title(
        f"MVPG training curves  ({log_csv.parent.name})   N={len(steps)}",
        fontsize=12,
    )

    # 下：奖励 r_mean / r_min / r_max + 主图指标 r_main
    colors = {
        "r_mean": "#1f77b4", "r_min": "#2ca02c", "r_max": "#ff7f0e",
        "r_main": "#9467bd",
    }
    for name in ("r_mean", "r_min", "r_max", "r_main"):
        if name not in metrics:
            continue
        raw = metrics[name]
        ax2.plot(steps, raw, alpha=0.22, lw=0.8, color=colors[name], label=f"{name} raw")
        if np.isnan(raw).any():  # r_main 缺主图时为 nan，只画 raw 避免平滑被污染
            ax2.plot(steps, raw, lw=1.8, color=colors[name], label=f"{name} (nan skipped)")
        else:
            ax2.plot(
                steps, _smooth(raw, window), lw=1.8, color=colors[name],
                label=f"{name} smooth({window})",
            )
    ax2.set_ylabel("RCGR reward r")
    ax2.set_xlabel("step")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    # 下：组内奖励标准差 r_std（探索度/熵坍缩前兆监控，独立面板便于观察收缩趋势）
    if "r_std" in metrics:
        std = metrics["r_std"]
        ax3.plot(steps, std, alpha=0.3, lw=0.8, color="#8c564b", label="r_std raw")
        ax3.plot(steps, _smooth(std, window), lw=1.8, color="#8c564b",
                 label=f"r_std smooth({window})")
        ax3.axhline(1e-3, color="gray", ls="--", lw=0.8, label="noise floor 1e-3")
        ax3.set_ylabel("group reward std")
        ax3.set_xlabel("step")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(alpha=0.3)
    else:  # 兼容旧 CSV（无 r_std 列）
        ax3.set_visible(False)

    # 下：主图 KL 散度（Σ_t log π_θ/π_ref；期望>=0，单样本可正可负；监控偏离参考模型程度）
    if "kl" in metrics:
        kl = metrics["kl"]
        ax4.plot(steps, kl, alpha=0.3, lw=0.8, color="#e377c2", label="kl raw")
        if np.isnan(kl).any():  # kl_weight<=0 或该组无主图时为 nan，只画 raw
            ax4.plot(steps, kl, lw=1.8, color="#e377c2", label="kl (nan skipped)")
        else:
            ax4.plot(steps, _smooth(kl, window), lw=1.8, color="#e377c2",
                     label=f"kl smooth({window})")
        ax4.axhline(0.0, color="gray", ls="--", lw=0.8, label="ref (kl=0)")
        ax4.set_ylabel("KL (main image)")
        ax4.set_xlabel("step")
        ax4.legend(loc="best", fontsize=9)
        ax4.grid(alpha=0.3)
    else:  # 兼容旧 CSV（无 kl 列）
        ax4.set_visible(False)

    if out_png is None:
        out_png = log_csv.with_suffix(".png")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_train_log] 已保存曲线: {out_png}")


def main():
    p = argparse.ArgumentParser(description="MVPG 训练指标可视化")
    p.add_argument("--log", default="train_log.csv",
                   help="train_log.csv 路径，或含它的输出目录（默认 train_log.csv）")
    p.add_argument("--out", default=None, help="输出 png 路径（默认与 log 同名 .png）")
    p.add_argument("--window", type=int, default=50, help="平滑窗口（步数）")
    args = p.parse_args()
    plot_train_log(args.log, out_png=args.out, window=args.window)


if __name__ == "__main__":
    main()
