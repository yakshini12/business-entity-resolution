"""Pairwise similarity features for (query record, Source-1 record) pairs.

All string similarities come from rapidfuzz (MIT licence), computed in
vectorised C++ batches (`cpdist`).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from normalize import skeleton, LEGAL, GENERIC


def _skel_str(s: str) -> str:
    return " ".join(skeleton(t) for t in s.split())


def add_derived(p: pd.DataFrame) -> pd.DataFrame:
    """Extra per-record string columns (computed once per record)."""
    p = p.copy()
    p["core_nosp"] = p["core"].str.replace(" ", "", regex=False)
    p["core_sk"] = [_skel_str(s) for s in p["core"].values]
    p["distinct"] = [" ".join(t for t in s.split() if t not in GENERIC) for s in p["core"].values]
    p["addr_full"] = (p["addr"] + " " + p["nums"]).str.strip()
    return p


def _set_feats(a_list, b_list):
    """Jaccard / overlap for space-separated token strings."""
    n = len(a_list)
    jac = np.zeros(n, np.float32)
    inter = np.zeros(n, np.float32)
    cov_a = np.zeros(n, np.float32)
    cov_b = np.zeros(n, np.float32)
    for i in range(n):
        A = set(a_list[i].split())
        B = set(b_list[i].split())
        if not A or not B:
            jac[i] = -1
            cov_a[i] = -1
            cov_b[i] = -1
            continue
        k = len(A & B)
        inter[i] = k
        jac[i] = k / len(A | B)
        cov_a[i] = k / len(A)
        cov_b[i] = k / len(B)
    return jac, inter, cov_a, cov_b


def _first_num_eq(a_list, b_list):
    out = np.zeros(len(a_list), np.float32)
    for i in range(len(a_list)):
        a = a_list[i].split()
        b = b_list[i].split()
        if not a or not b:
            out[i] = -1
        else:
            out[i] = float(a[0] == b[0] or a[0] in b)
    return out


def _group_top2(g, v):
    """Per-row max and second max of v within group g (vectorised)."""
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    start = np.r_[True, gs[1:] != gs[:-1]]
    first_pos = np.maximum.accumulate(np.where(start, np.arange(len(gs)), 0))
    top = vs[first_pos]
    sec_pos = first_pos + 1
    has2 = (sec_pos < len(gs)) & (gs[np.minimum(sec_pos, len(gs) - 1)] == gs)
    sec = np.where(has2, vs[np.minimum(sec_pos, len(gs) - 1)], -1.0)
    mx = np.empty_like(v); sc = np.empty_like(v)
    mx[order] = top; sc[order] = sec
    return mx.astype(np.float32), sc.astype(np.float32)


def pair_features(q: pd.DataFrame, s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """q, s1: frames from add_derived(prepare(..)); cand: qi, si, bscore, brank, bgap, b2."""
    qi = cand["qi"].values
    si = cand["si"].values
    F = {}

    def col(df, name, idx):
        return df[name].values[idx]

    qa, sa = col(q, "core", qi), col(s1, "core", si)
    F["n_ratio"] = cpdist(qa, sa, scorer=fuzz.ratio, workers=-1)
    F["n_tset"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)
    F["n_tsort"] = cpdist(qa, sa, scorer=fuzz.token_sort_ratio, workers=-1)
    F["n_partial"] = cpdist(qa, sa, scorer=fuzz.partial_ratio, workers=-1)
    F["n_jw"] = cpdist(qa, sa, scorer=JaroWinkler.normalized_similarity, workers=-1)
    j, k, ca, cb = _set_feats(qa, sa)
    F["n_jac"], F["n_inter"], F["n_cov_q"], F["n_cov_s"] = j, k, ca, cb
    F["n_len_q"] = np.fromiter((len(x) for x in qa), np.float32, len(qa))
    F["n_len_s"] = np.fromiter((len(x) for x in sa), np.float32, len(sa))
    F["n_ntok_q"] = np.fromiter((len(x.split()) for x in qa), np.float32, len(qa))
    F["n_ntok_s"] = np.fromiter((len(x.split()) for x in sa), np.float32, len(sa))

    qa, sa = col(q, "name", qi), col(s1, "name", si)
    F["fn_ratio"] = cpdist(qa, sa, scorer=fuzz.ratio, workers=-1)
    F["fn_tset"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)

    qa, sa = col(q, "core_nosp", qi), col(s1, "core_nosp", si)
    F["ns_ratio"] = cpdist(qa, sa, scorer=fuzz.ratio, workers=-1)
    F["ns_partial"] = cpdist(qa, sa, scorer=fuzz.partial_ratio, workers=-1)

    qa, sa = col(q, "core_sk", qi), col(s1, "core_sk", si)
    F["sk_ratio"] = cpdist(qa, sa, scorer=fuzz.ratio, workers=-1)
    F["sk_tset"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)
    j, k, ca, cb = _set_feats(qa, sa)
    F["sk_jac"], F["sk_cov_q"], F["sk_cov_s"] = j, ca, cb

    qa, sa = col(q, "distinct", qi), col(s1, "distinct", si)
    F["d_tset"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)
    F["d_ratio"] = cpdist(qa, sa, scorer=fuzz.ratio, workers=-1)

    qa, sa = col(q, "addr", qi), col(s1, "addr", si)
    F["a_ratio"] = cpdist(qa, sa, scorer=fuzz.ratio, workers=-1)
    F["a_tset"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)
    F["a_tsort"] = cpdist(qa, sa, scorer=fuzz.token_sort_ratio, workers=-1)
    F["a_partial"] = cpdist(qa, sa, scorer=fuzz.partial_ratio, workers=-1)
    j, k, ca, cb = _set_feats(qa, sa)
    F["a_jac"], F["a_inter"], F["a_cov_q"], F["a_cov_s"] = j, k, ca, cb
    F["a_empty_q"] = np.fromiter((len(x) == 0 for x in qa), np.float32, len(qa))
    F["a_ntok_q"] = np.fromiter((len(x.split()) for x in qa), np.float32, len(qa))
    F["a_ntok_s"] = np.fromiter((len(x.split()) for x in sa), np.float32, len(sa))

    qa, sa = col(q, "nums", qi), col(s1, "nums", si)
    j, k, ca, cb = _set_feats(qa, sa)
    F["d_jac"], F["d_inter"], F["d_cov_q"], F["d_cov_s"] = j, k, ca, cb
    F["d_first"] = _first_num_eq(qa, sa)
    F["d_num_ratio"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)

    qa, sa = col(q, "addr_full", qi), col(s1, "addr_full", si)
    F["af_tset"] = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=-1)

    F["deva_q"] = q["deva"].values[qi].astype(np.float32)
    F["src3"] = np.fromiter((x.startswith("S3") for x in q["entity_id"].values[qi]), np.float32, len(qi))
    for c in ("bscore", "brank", "bgap", "b2"):
        F[c] = cand[c].values.astype(np.float32)

    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})

    # context features: how does this candidate compare with the query's other candidates?
    for c in ("n_tset", "a_tset", "sk_ratio", "af_tset", "bscore"):
        mx, sec = _group_top2(qi, X[c].values)
        v = X[c].values
        X[c + "_gap"] = mx - v                       # 0 for the best candidate
        X[c + "_2nd"] = np.where(v >= mx, sec, mx)   # best competitor's value
    _, inv, cnt = np.unique(qi, return_inverse=True, return_counts=True)
    X["ncand"] = cnt[inv].astype(np.float32)
    return X
