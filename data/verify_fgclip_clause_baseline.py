"""验证「caption 内均值 baseline」能否把子句级 box 相似度变成带符号的
鼓励/抑制 reward（reward = s_box − 同 caption 子句均值）。

读取 fgclip_split_compare_results.json 里 combined（联合 `.,;` 分割）结果，
按 (gid, cap_idx) 分组做句内中心化，统计：
- 幻觉子句 reward 是否整体为负（应被抑制）、真实子句 reward 是否整体为正（应被鼓励）
- 符号判别准确率（抑制正确率 / 鼓励正确率 / 整体，以及仅含幻觉子句的 caption）
- reward 上的 Cohen's d 与组级 margin>0
"""

import json
from pathlib import Path

from statistics import mean, pstdev


def cohens_d(a, b):
    ma, mb = mean(a), mean(b)
    sa, sb = pstdev(a), pstdev(b)
    sp = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    return (mb - ma) / sp if sp > 0 else 0.0


def main():
    d = json.load(open("/root/MVPG/fgclip_split_compare_results.json"))
    rows = d["results"]["combined"]

    # 按 (gid, cap_idx) 分组
    groups = {}
    for r in rows:
        groups.setdefault((r["gid"], r["cap_idx"]), []).append(r)

    # 计算每个子句的 reward = s_box − 组内均值
    rewarded = []
    for key, gr in groups.items():
        m = mean(r["s_box"] for r in gr)
        for r in gr:
            rw = r["s_box"] - m
            rewarded.append({**r, "reward": rw})

    hall_rw = [r["reward"] for r in rewarded if r["is_hall"]]
    real_rw = [r["reward"] for r in rewarded if not r["is_hall"]]

    # 符号判别：幻觉应被抑制(reward<0)，真实应被鼓励(reward>0)
    def sign_stats(subset):
        total = len(subset)
        hall = [r for r in subset if r["is_hall"]]
        real = [r for r in subset if not r["is_hall"]]
        hall_ok = sum(1 for r in hall if r["reward"] < 0)
        real_ok = sum(1 for r in real if r["reward"] > 0)
        return {
            "n": total, "n_hall": len(hall), "n_real": len(real),
            "suppress_acc": hall_ok / len(hall) if hall else 0.0,
            "encourage_acc": real_ok / len(real) if real else 0.0,
            "overall_acc": (hall_ok + real_ok) / total if total else 0.0,
        }

    all_stats = sign_stats(rewarded)

    # 仅含幻觉子句的 caption（这些才真正需要给抑制信号）
    hall_keys = {key for key, gr in groups.items() if any(r["is_hall"] for r in gr)}
    hall_only_rows = [r for r in rewarded if (r["gid"], r["cap_idx"]) in hall_keys]
    hall_only_stats = sign_stats(hall_only_rows)

    # 组级 margin（用 reward）
    gids = sorted(set(r["gid"] for r in rewarded))
    gid_margins = []
    for gid in gids:
        mh = mean(r["reward"] for r in rewarded if r["gid"] == gid and r["is_hall"])
        mr = mean(r["reward"] for r in rewarded if r["gid"] == gid and not r["is_hall"])
        gid_margins.append(mr - mh)

    summary = {
        "n_hall": len(hall_rw), "n_real": len(real_rw),
        "hall_reward_mean": mean(hall_rw), "real_reward_mean": mean(real_rw),
        "cohens_d": cohens_d(hall_rw, real_rw),
        "suppress_acc": all_stats["suppress_acc"],
        "encourage_acc": all_stats["encourage_acc"],
        "overall_acc": all_stats["overall_acc"],
        "hall_only_suppress_acc": hall_only_stats["suppress_acc"],
        "hall_only_overall_acc": hall_only_stats["overall_acc"],
        "gid_margin_gt0": sum(1 for m in gid_margins if m > 0),
        "n_groups": len(gids),
    }

    json.dump({"summary": summary, "rewarded": rewarded},
              open("/root/MVPG/fgclip_clause_baseline_results.json", "w"),
              ensure_ascii=False, indent=2)

    L = []
    L.append("# caption 内均值 baseline：子句级 reward 符号判别验证\n")
    L.append("- 信号：联合 `.,;` 分割 → 子句 box 头 × dense patch max_patch_sim（raw）。\n")
    L.append("- baseline：`reward = s_box − 同 caption 子句均值`（句内中心化）。\n")
    L.append("- 判定：幻觉子句 reward<0 视为「正确抑制」，真实子句 reward>0 视为「正确鼓励」。\n")

    L.append("\n## 结果\n")
    L.append(f"- 幻觉子句 reward 均值：**{summary['hall_reward_mean']:+.5f}**（应为负）\n")
    L.append(f"- 真实子句 reward 均值：**{summary['real_reward_mean']:+.5f}**（应为正）\n")
    L.append(f"- reward 上 Cohen's d：**{summary['cohens_d']:+.2f}**\n")
    L.append(f"- 组级 margin>0：**{summary['gid_margin_gt0']}/{summary['n_groups']}**\n")

    L.append("\n## 符号判别准确率\n")
    L.append("| 范围 | 抑制正确率(幻觉<0) | 鼓励正确率(真实>0) | 整体 |")
    L.append("|---|---|---|---|")
    L.append(f"| 全部子句 | {summary['suppress_acc']*100:.1f}% | {summary['encourage_acc']*100:.1f}% | {summary['overall_acc']*100:.1f}% |")
    L.append(f"| 仅含幻觉子句的 caption | {summary['hall_only_suppress_acc']*100:.1f}% | - | {summary['hall_only_overall_acc']*100:.1f}% |")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_clause_baseline_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_clause_baseline_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
