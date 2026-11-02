"""Scalar training diagnostics, separate from held-out hallucination metrics."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


def scalar_metrics(row, global_step, record_index):
    result = {"attempt": row["attempt"], "global_step": global_step,
              "epoch": row["epoch"], "record_index": record_index,
              "record_id": row["record_id"], "skipped": row["skipped"],
              "reason": row["reason"], "branch": row.get("branch")}
    for key, value in row.items():
        if key not in result and (value is None or isinstance(value, (int, float, bool))):
            result[key] = value
    result.update(row.get("objectives", {}))
    return result


class TrainingMetrics:
    def __init__(self, root, *, tensorboard=False):
        self.root = Path(root)
        self.handle = (self.root / "metrics.jsonl").open("x", encoding="utf-8")
        self.writer = None
        self.branches = Counter()
        self.counts = Counter(attempted=0, updated=0, skipped=0)
        self.elapsed = 0.0
        self.conflicts = self.projections = 0
        self.positive_down = self.positive_up = 0
        self.negative_up = self.negative_down = 0
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.root / "tensorboard"))
            except ImportError as exc:
                self.handle.close()
                raise RuntimeError("TensorBoard logging needs the tensorboard package") from exc

    def record(self, row, *, record_index):
        self.counts["attempted"] += 1
        self.counts["skipped" if row["skipped"] else "updated"] += 1
        flat = scalar_metrics(row, self.counts["updated"], record_index)
        self.handle.write(json.dumps(flat, allow_nan=False) + "\n")
        self.handle.flush()
        self.elapsed += row.get("elapsed_seconds", 0.0)
        if not row["skipped"]:
            self.branches[row["branch"]] += 1
            self.conflicts += bool(row.get("conflict"))
            self.projections += bool(row.get("projection_applied"))
            for prefix in ("positive", "negative"):
                before, after = row.get(prefix + "_loss"), row.get(prefix + "_loss_after")
                if before is not None and after is not None:
                    setattr(self, prefix + "_down", getattr(self, prefix + "_down") + (after < before))
                    setattr(self, prefix + "_up", getattr(self, prefix + "_up") + (after > before))
            if self.writer:
                for key, value in flat.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        self.writer.add_scalar("train/" + key, value, flat["global_step"])
                self.writer.flush()

    def summary(self):
        return {"counts": dict(self.counts), "branches": dict(self.branches),
                "conflict_updates": self.conflicts, "projected_updates": self.projections,
                "step_seconds_sum": self.elapsed,
                "positive_nll_decreased": self.positive_down, "positive_nll_increased": self.positive_up,
                "negative_nll_increased": self.negative_up, "negative_nll_decreased": self.negative_down,
                "note": "Training diagnostics only; NLL or feedback counts are not benchmark accuracy"}

    def close(self):
        self.handle.close()
        if self.writer:
            self.writer.close()
