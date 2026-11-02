"""Masked objectives and conservative per-response SGD direction modulation.

The signs refer to *parameter updates*: error NLL is increased and supported
NLL is decreased.  Projection is an adaptation of existing gradient-conflict
methods, not an original optimizer.  It only constrains the selected average
positive loss to first order; it does not guarantee free-generation retention.

``max_update_norm`` bounds the norm of the actual representable parameter
delta, including parameter-dtype rounding.  ``correction_tolerance`` is tested
on the direction before positive norm rescaling; the post-rescaling sign and
absolute tolerance are reported separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

import torch


MODES = ("modulated", "fixed", "positive", "negative", "project_only")


def _number(name: str, value: Any, *, minimum: float = 0.0,
            strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return value


def masked_objectives(token_nll: torch.Tensor, units: Iterable[Mapping[str, Any]]):
    """Return separately normalized (positive, negative, statistics) objectives.

    Each retained unit contributes its mean token NLL, weighted by ``weight``.
    Empty, zero-weight, and explicitly unresolved units do not contribute.
    Alignment must already have resolved duplicate ownership: duplicate token
    indices, including indices shared across retained units, raise an error.
    No graph is detached from the returned objectives.
    """
    if not isinstance(token_nll, torch.Tensor) or token_nll.ndim != 1:
        raise ValueError("token_nll must be a one-dimensional torch.Tensor")
    if not token_nll.is_floating_point() or not torch.isfinite(token_nll).all().item():
        raise ValueError("token_nll must contain finite real floating-point values")
    stats = {"input_units": 0, "skipped_units": 0,
             "positive_units": 0, "negative_units": 0,
             "positive_tokens": 0, "negative_tokens": 0,
             "positive_weight": 0.0, "negative_weight": 0.0}
    groups: dict[str, list[tuple[float, torch.Tensor]]] = {"+": [], "-": []}
    used: set[int] = set()
    dtype = torch.float64 if token_nll.dtype == torch.float64 else torch.float32
    for unit in units:
        stats["input_units"] += 1
        if not isinstance(unit, Mapping):
            raise ValueError("every aligned unit must be a mapping")
        label = unit.get("label")
        if label not in ("+", "-", "?"):
            raise ValueError("aligned unit label must be '+', '-', or '?'")
        weight = _number("unit weight", unit.get("weight", 1.0))
        indices = unit.get("token_indices", [])
        if not isinstance(indices, (list, tuple)):
            raise ValueError("token_indices must be a list or tuple of integers")
        if any(isinstance(t, bool) or not isinstance(t, Integral) for t in indices):
            raise ValueError("token_indices must contain integers")
        indices = [int(t) for t in indices]
        if len(indices) != len(set(indices)):
            raise ValueError("a unit contains duplicate token indices")
        if any(t < 0 or t >= token_nll.numel() for t in indices):
            raise ValueError("token index lies outside token_nll")
        if label == "?" or weight == 0.0 or not indices:
            stats["skipped_units"] += 1
            continue
        if used.intersection(indices):
            raise ValueError("source-token ownership must be resolved before computing losses")
        used.update(indices)
        index = torch.tensor(indices, dtype=torch.long, device=token_nll.device)
        mean = token_nll.index_select(0, index).to(dtype=dtype).mean()
        groups[label].append((weight, mean))
        prefix = "positive" if label == "+" else "negative"
        stats[prefix + "_units"] += 1
        stats[prefix + "_tokens"] += len(indices)
    losses = {}
    for label, values in groups.items():
        if not values:
            losses[label] = None
            continue
        # Scale weights first to avoid overflow in an otherwise valid weighted
        # mean. Confidence weights are normally in [0, 1], but relative weights
        # remain supported by this loss helper.
        scale = max(w for w, _ in values)
        denominator = math.fsum(w / scale for w, _ in values)
        loss = sum((w / scale / denominator) * mean for w, mean in values)
        if not torch.isfinite(loss).item():
            raise ValueError("weighted objective is not finite")
        losses[label] = loss
        prefix = "positive" if label == "+" else "negative"
        try:
            total = math.fsum(w for w, _ in values)
        except OverflowError:
            total = float("inf")
        stats[prefix + "_weight"] = total
    return losses["+"], losses["-"], stats


def _parameters(model_or_parameters):
    values = (model_or_parameters.parameters() if hasattr(model_or_parameters, "parameters")
              else model_or_parameters)
    params = []
    seen = set()
    for p in values:
        if not isinstance(p, torch.Tensor):
            raise ValueError("parameters must be torch tensors")
        if not p.requires_grad:
            continue
        if not p.is_leaf or not p.is_floating_point():
            raise ValueError("trainable parameters must be real floating-point leaf tensors")
        if id(p) in seen:
            raise ValueError("duplicate trainable parameter")
        seen.add(id(p))
        params.append(p)
    return params


def _dot(left, right):
    # Per-tensor FP64 reductions avoid a giant concatenated adapter vector and
    # reduce errors in nearly orthogonal directions. Parameters may span devices.
    try:
        result = math.fsum(torch.sum(a.to(torch.float64) * b.to(torch.float64)).item()
                           for a, b in zip(left, right))
    except (OverflowError, ValueError) as exc:
        raise ValueError("nonfinite direction reduction; no parameters were changed") from exc
    if not math.isfinite(result):
        raise ValueError("nonfinite direction reduction; no parameters were changed")
    return result


def _norm(vector):
    return math.sqrt(max(0.0, _dot(vector, vector)))


def _cosine(left, right):
    denominator = _norm(left) * _norm(right)
    return max(-1.0, min(1.0, _dot(left, right) / denominator)) if denominator else None


def _finite(vector):
    return all(torch.isfinite(t).all().item() for t in vector)


def _validate_loss(loss, name):
    if loss is not None and (not isinstance(loss, torch.Tensor) or loss.ndim != 0
                             or not loss.is_floating_point()
                             or not torch.isfinite(loss.detach()).item()):
        raise ValueError(f"{name} must be a finite scalar floating-point tensor or None")


def _gradient(loss, params, retain_graph):
    if loss is None or not loss.requires_grad:
        grads = [None] * len(params)
    else:
        grads = torch.autograd.grad(loss, params, allow_unused=True,
                                    retain_graph=retain_graph)
    result = []
    for p, grad in zip(params, grads):
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        if grad is None:
            result.append(torch.zeros_like(p, dtype=dtype))
        elif grad.layout != torch.strided:
            raise ValueError("sparse gradients are not supported by response_step")
        else:
            result.append(grad.detach().to(dtype=dtype))
    return result


def response_step(model_or_parameters, positive_loss, negative_loss, *,
                  mode="modulated", lr, alpha=1.0, beta=1.0,
                  max_update_norm=1.0, protection_tolerance=1e-12,
                  correction_tolerance=0.0, beta_backoff=0.5, max_backoffs=20):
    """Apply one bounded, momentum-free SGD step for one response.

    ``model_or_parameters`` is an nn.Module or an iterable of leaf parameters.
    ``fixed`` is the unmodified weighted-sum control: no projection, coefficient
    backoff, or preservation guard. ``positive`` and ``negative`` only use their
    named objective. ``project_only`` sets the active positive coefficient to 0;
    it still uses a positive gradient as a constraint when available.

    Protected modes check correction on the pre-clipping direction and audit
    its sign and positive-loss first-order change on the representable delta.
    All candidate values are validated before any parameter is mutated. Numerical
    guards may skip otherwise useful steps; they do not establish a finite-step
    loss or capability guarantee.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    lr = _number("lr", lr, strict=True)
    alpha = _number("alpha", alpha)
    beta = _number("beta", beta)
    max_update_norm = _number("max_update_norm", max_update_norm, strict=True)
    protection_tolerance = _number("protection_tolerance", protection_tolerance)
    correction_tolerance = _number("correction_tolerance", correction_tolerance)
    beta_backoff = _number("beta_backoff", beta_backoff, strict=True)
    if beta_backoff >= 1:
        raise ValueError("beta_backoff must be strictly between 0 and 1")
    if isinstance(max_backoffs, bool) or not isinstance(max_backoffs, Integral) or max_backoffs < 0:
        raise ValueError("max_backoffs must be a nonnegative integer")
    _validate_loss(positive_loss, "positive_loss")
    _validate_loss(negative_loss, "negative_loss")
    use_positive = positive_loss is not None and mode != "negative"
    use_negative = negative_loss is not None and mode != "positive"
    protected_mode = mode in ("modulated", "project_only")
    branch = ("both" if use_positive and use_negative else
              "positive_only" if use_positive else "negative_only" if use_negative else "none")
    stats = {"mode": mode, "branch": branch, "skipped": True, "reason": None,
             "positive_loss": float(positive_loss.detach().item()) if positive_loss is not None else None,
             "negative_loss": float(negative_loss.detach().item()) if negative_loss is not None else None,
             "conflict": False, "projection_applied": False, "protection_active": False,
             "beta_used": 0.0 if mode == "project_only" else beta, "backoffs": 0,
             "positive_gradient_norm": 0.0, "negative_gradient_norm": 0.0,
             "direction_norm": 0.0, "actual_update_norm": 0.0,
             "pre_direction_cosine": None, "post_direction_cosine": None,
             "correction_dot": None, "positive_dot": None,
             "actual_correction_dot": None, "actual_positive_dot": None,
             "correction_guard_satisfied": None,
             "actual_correction_sign_ok": None,
             "actual_correction_above_tolerance": None,
             "positive_guard_satisfied": None,
             "norm_scale": 0.0, "rounding_backoffs": 0}

    def skip(reason):
        stats["reason"] = reason
        return stats

    params = _parameters(model_or_parameters)
    if not params:
        return skip("no_trainable_parameters")
    if branch == "none":
        return skip("no_selected_objective")
    g_pos = _gradient(positive_loss if use_positive else None, params,
                      retain_graph=use_negative)
    g_neg = _gradient(negative_loss if use_negative else None, params, retain_graph=False)
    if not _finite(g_pos) or not _finite(g_neg):
        raise ValueError("nonfinite objective gradient; no parameters were changed")
    d_pos = [-g for g in g_pos]
    d_neg = g_neg
    stats["positive_gradient_norm"] = _norm(g_pos)
    stats["negative_gradient_norm"] = _norm(g_neg)
    stats["pre_direction_cosine"] = _cosine(d_neg, d_pos) if branch == "both" else None
    cross = _dot(d_neg, d_pos)
    stats["conflict"] = branch == "both" and cross < 0.0
    d_correct = d_neg
    if protected_mode and branch == "both":
        positive_norm_sq = _dot(d_pos, d_pos)
        stats["protection_active"] = positive_norm_sq > protection_tolerance
        if cross < 0:
            if positive_norm_sq <= protection_tolerance:
                return skip("near_zero_protection")
            coefficient = cross / positive_norm_sq
            d_correct = [a - coefficient * b for a, b in zip(d_neg, d_pos)]
            stats["projection_applied"] = True
        stats["post_direction_cosine"] = _cosine(d_correct, d_pos)

    beta_used = stats["beta_used"] if use_positive else 0.0
    alpha_used = alpha if use_negative else 0.0
    # Scalar checks permit coefficient search without allocating a new complete
    # adapter direction for every backoff attempt.
    if protected_mode and use_negative:
        correction_base = alpha_used * _dot(g_neg, d_correct)
        positive_interaction = _dot(g_neg, d_pos)
        valid = correction_base + beta_used * positive_interaction > correction_tolerance
        for _ in range(int(max_backoffs)):
            if valid or beta_used == 0.0:
                break
            beta_used *= beta_backoff
            stats["backoffs"] += 1
            valid = correction_base + beta_used * positive_interaction > correction_tolerance
        if not valid and beta_used != 0.0:
            beta_used = 0.0
            stats["backoffs"] += 1
            valid = correction_base > correction_tolerance
        stats["beta_used"] = beta_used
        stats["correction_guard_satisfied"] = valid
        if not valid:
            return skip("no_corrective_direction")
    else:
        stats["beta_used"] = beta_used

    direction = [alpha_used * a + beta_used * b for a, b in zip(d_correct, d_pos)]
    if not _finite(direction):
        raise ValueError("nonfinite update direction; no parameters were changed")
    direction_norm = _norm(direction)
    stats["direction_norm"] = direction_norm
    if direction_norm == 0.0:
        return skip("zero_direction")
    stats["correction_dot"] = _dot(g_neg, direction) if use_negative else None
    stats["positive_dot"] = _dot(g_pos, direction) if use_positive else None
    if protected_mode and use_negative and stats["correction_dot"] <= correction_tolerance:
        stats["correction_guard_satisfied"] = False
        return skip("correction_guard_after_rounding")
    if protected_mode and use_positive and stats["positive_dot"] > 0.0:
        stats["positive_guard_satisfied"] = False
        return skip("positive_direction_guard")

    scale = min(lr, max_update_norm / direction_norm)
    values = []
    deltas = []
    # Quantization can overshoot an ideal norm cap. Reduce the common positive
    # scale and recompute complete candidate values before committing anything.
    for attempt in range(25):
        values = [p.detach() + (d * scale).to(dtype=p.dtype)
                  for p, d in zip(params, direction)]
        if not _finite(values):
            raise ValueError("nonfinite candidate parameter; no parameters were changed")
        deltas = [new.to(dtype=d.dtype) - p.detach().to(dtype=d.dtype)
                  for new, p, d in zip(values, params, direction)]
        if not _finite(deltas):
            raise ValueError("nonfinite representable delta; no parameters were changed")
        actual_norm = _norm(deltas)
        if actual_norm <= max_update_norm:
            break
        stats["rounding_backoffs"] += 1
        scale *= min(0.5, 0.99 * max_update_norm / actual_norm)
    else:
        return skip("actual_update_norm_guard")
    stats["norm_scale"] = scale / lr
    if actual_norm == 0.0:
        return skip("zero_actual_update")
    stats["actual_correction_dot"] = _dot(g_neg, deltas) if use_negative else None
    stats["actual_positive_dot"] = _dot(g_pos, deltas) if use_positive else None
    if use_negative:
        stats["actual_correction_sign_ok"] = stats["actual_correction_dot"] > 0.0
        stats["actual_correction_above_tolerance"] = stats["actual_correction_dot"] > correction_tolerance
        if protected_mode and not stats["actual_correction_sign_ok"]:
            return skip("actual_correction_guard")
    if use_positive:
        stats["positive_guard_satisfied"] = stats["actual_positive_dot"] <= 0.0
        if protected_mode and not stats["positive_guard_satisfied"]:
            return skip("actual_positive_guard")

    with torch.no_grad():
        for p, new in zip(params, values):
            p.copy_(new)
    stats["skipped"] = False
    stats["reason"] = "updated"
    stats["actual_update_norm"] = actual_norm
    return stats
