"""Bounded, equal-length evaluation batches for the older LLaVA interface.

No prompt padding is introduced. Only small prompt metadata and decoded answers
are retained in each window; decoded image tensors live in a bounded CPU cache.
The batch scheduler and cache do not import model frameworks.
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import islice


def validate_batch_size(batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def ordered_batch_map(items, predict_batch, *, batch_size, batch_key,
                      window_batches=32):
    """Group a bounded window by key, then emit predictions in input order.

    ``batch_key`` must include prompt length and every generation setting that
    varies by example. ``predict_batch`` returns one decoded answer per input.
    Reordering affects execution only: scorers receive the original row order.
    """
    validate_batch_size(batch_size)
    validate_batch_size(window_batches)
    iterator = iter(items)
    # Singleton mode retains execution order, including sampling RNG order and
    # per-response flushing, as well as the original output order.
    window_size = 1 if batch_size == 1 else batch_size * window_batches
    while window := list(islice(iterator, window_size)):
        buckets = OrderedDict()
        for index, item in enumerate(window):
            buckets.setdefault(batch_key(item), []).append(index)
        answers = [None] * len(window)
        for indices in buckets.values():
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size]
                predictions = list(predict_batch([window[i] for i in selected]))
                if len(predictions) != len(selected):
                    raise ValueError("Batch prediction count does not match input count")
                for index, prediction in zip(selected, predictions):
                    answers[index] = prediction
        yield from zip(window, answers)


class BoundedImageCache:
    """LRU of preprocessed CPU image tensors; no all-dataset image allocation."""
    def __init__(self, loader, max_size=32):
        validate_batch_size(max_size)
        self.loader, self.max_size = loader, max_size
        self._values = OrderedDict()

    def get(self, path):
        if path in self._values:
            value = self._values.pop(path)
        else:
            value = self.loader(path)
        self._values[path] = value
        if len(self._values) > self.max_size:
            self._values.popitem(last=False)
        return value


def generate_equal_length_batch(model, tokenizer, items, image_cache, *,
                                device, compute_dtype, temperature,
                                top_p=None, num_beams=1):
    """Generate raw text with unchanged prompts and strict prefix validation.

    Each item contains a 1-D ``input_ids`` tensor, ``image_path``, ``stop_str``
    and ``max_new_tokens``. One image per prompt is supported. Equal-length
    prompts avoid the old multimodal implementation's padding-mask ambiguity.
    """
    import torch
    if not items:
        return []
    lengths = {item["input_ids"].numel() for item in items}
    limits = {item["max_new_tokens"] for item in items}
    if len(lengths) != 1 or any(item["input_ids"].ndim != 1 for item in items):
        raise ValueError("Evaluation batches require equal-length, unpadded 1-D input IDs")
    if len(limits) != 1:
        raise ValueError("Evaluation batches require the same generation limit")
    input_ids = torch.stack([item["input_ids"] for item in items]).to(device)
    images = torch.stack([image_cache.get(item["image_path"]) for item in items]).to(
        device=device, dtype=compute_dtype)
    model.config.use_cache = True
    model.config.cache_shape = (2048,)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids, images=images,
            attention_mask=torch.ones_like(input_ids),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p, num_beams=num_beams,
            max_new_tokens=items[0]["max_new_tokens"], use_cache=True)
    prompt_length = input_ids.shape[1]
    if (output_ids.ndim != 2 or output_ids.shape[0] != len(items)
            or output_ids.shape[1] < prompt_length
            or not torch.equal(output_ids[:, :prompt_length], input_ids)):
        raise ValueError("Unexpected LLaVA generation output: original prompt prefix was changed or omitted")
    decoded = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
    answers = []
    for item, text in zip(items, decoded):
        text = text.strip()
        stop = item["stop_str"]
        if stop and text.endswith(stop):
            text = text[:-len(stop)]
        answers.append(text.strip())
    return answers
