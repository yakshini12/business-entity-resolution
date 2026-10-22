"""Macro F0.5 exactly as described in the challenge README."""
import numpy as np


def f05_macro(pred: dict, truth: dict, s1_ids) -> float:
    """pred / truth: {s1_id: set(ids)}; s1_ids: all Source-1 ids evaluated."""
    tot = 0.0
    for s in s1_ids:
        p = pred.get(s, set())
        t = truth.get(s, set())
        if not t:
            tot += 1.0 if not p else 0.0
            continue
        if not p:
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr = tp / len(p)
        rc = tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / max(1, len(s1_ids))
