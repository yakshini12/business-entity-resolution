"""Shared stages used by train.py and predict.py."""
import csv
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from blocking import BlockIndex, prepare
from features import add_derived, pair_features

CFG = {
    "retrieve_k": 20,      # top-k Source-1 records retrieved per query by the inverted index
    "model_k": 8,          # top-k kept after cheap re-ranking; this is candidate_pairs.tsv
    "cap_frac": 0.0025,    # max document frequency for name keys (fraction of S1 in country)
    "cap_abs": 300,        # max document frequency for address / pair keys
}

csv.field_size_limit(10 ** 8)


def read_tsv(path, **kw):
    """Always tab-separated, everything as string, empty stays empty."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, **kw)


def candidates_and_features(index: BlockIndex, s1p: pd.DataFrame, qp: pd.DataFrame, cfg=CFG):
    """Blocking + re-ranking + pairwise features for one chunk of queries.

    s1p: prepared Source-1 frame of the country the index was built on.
    qp:  prepared query frame (same country).
    Returns (cand, X): cand has positional qi (into qp) and si (into s1p).
    """
    cand = index.search(qp, k=cfg["retrieve_k"])
    if len(cand) == 0:
        return cand, None
    # derived strings only for the S1 rows we actually touch
    uniq_si, local_si = np.unique(cand["si"].values, return_inverse=True)
    s1_sub = add_derived(s1p.iloc[uniq_si].reset_index(drop=True))
    q_d = add_derived(qp.reset_index(drop=True))
    qi, ls = cand["qi"].values, local_si
    # cheap re-ranking (name + address token-set similarity + blocking score)
    nts = cpdist(q_d["core"].values[qi], s1_sub["core"].values[ls],
                 scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    ats = cpdist(q_d["addr_full"].values[qi], s1_sub["addr_full"].values[ls],
                 scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    bn = cand["bscore"].values / cand.groupby("qi")["bscore"].transform("max").values
    pre = bn + 10 * nts + 5 * ats
    order = np.lexsort((-pre, qi))
    qs = qi[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(qs)), 0))
    prank = np.empty(len(qi), np.int32)
    prank[order] = np.arange(len(qs)) - grp_start + 1
    keep = prank <= cfg["model_k"]
    cand = cand[keep].reset_index(drop=True)
    cand["pre"] = pre[keep].astype(np.float32)
    cand["prank"] = prank[keep].astype(np.float32)
    local = cand.copy()
    local["si"] = ls[keep]
    X = pair_features(q_d, s1_sub, local)
    X["pre"] = cand["pre"].values
    X["prank"] = cand["prank"].values
    return cand, X


def best_per_query(qi, prob):
    """Index (into the pair arrays) of the highest-probability pair of each query."""
    order = np.lexsort((-prob, qi))
    qs = qi[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    return order[first]
