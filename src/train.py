"""Train the pair classifier and choose the decision threshold.

Validation design (no leakage, realistic density):
  * validation Source-1 entities = every S1 record whose address falls in a set of
    held-out regions (default: Iowa + Rajasthan);
  * validation queries = all their true S2/S3 matches + every S2/S3 record whose
    address mentions those regions (these include the unmatched "decoys");
  * training queries = a random sample of all other S2/S3 records;
  * blocking always searches the FULL training Source-1 index of the country,
    so candidate competition looks like it will at test time.
The threshold maximising macro F0.5 on the validation entities is saved, then the
model is refit on train + validation pairs.

Usage:
  python src/train.py --data-dir dataset --model-dir models
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import BlockIndex, prepare  # noqa: E402
from evaluate import f05_macro  # noqa: E402
from pipeline import CFG, best_per_query, candidates_and_features, read_tsv  # noqa: E402

LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=42)
N_ROUNDS = 250


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stable_frac(ids, salt="x"):
    """Deterministic pseudo-random number in [0,1) per id."""
    return np.array([zlib.crc32((salt + i).encode()) / 2 ** 32 for i in ids])


def safe(c):
    return "".join(ch if ch.isalnum() else "_" for ch in c) or "EMPTY"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--train-query-frac", type=float, default=0.03,
                    help="fraction of non-validation S2/S3 records used as training queries")
    ap.add_argument("--retrieve-k", type=int, default=CFG["retrieve_k"],
                    help="number of blocking candidates retrieved before reranking")
    ap.add_argument("--model-k", type=int, default=CFG["model_k"],
                    help="number of candidates kept for feature scoring")
    ap.add_argument("--val-regex", default=r"(?:, IA(?:$|,)|Iowa|Rajasthan|, RJ(?:$|,)|राजस्थान)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    CFG["retrieve_k"] = args.retrieve_k
    CFG["model_k"] = args.model_k
    os.makedirs(args.model_dir, exist_ok=True)
    tr = os.path.join(args.data_dir, "train")
    val_re = re.compile(args.val_regex)

    s1_path = os.path.join(tr, "train_source1.tsv")
    log("scanning Source 1 in chunks")
    val_s1_ids = set()
    s1_counts = {}
    n_s1 = 0
    for ch in read_tsv(s1_path, chunksize=200_000):
        counts = ch["country"].value_counts()
        for country, count in counts.items():
            s1_counts[country] = s1_counts.get(country, 0) + int(count)
        is_val = ch["business_address"].str.contains(val_re).values
        val_s1_ids.update(ch["entity_id"].values[is_val])
        n_s1 += len(ch)
        del ch
    log("S1", n_s1, "by country", s1_counts,
        "validation S1", len(val_s1_ids))

    log("reading ground truth")
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv"))
    val_truth, val_matched = {}, set()
    for s, v in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        if s in val_s1_ids:
            ids = set(x for x in v.split(",") if x)
            val_truth[s] = ids
            val_matched |= ids

    log("selecting queries")
    parts = []
    for src in ("train_source2.tsv", "train_source3.tsv"):
        for ch in read_tsv(os.path.join(tr, src), chunksize=500_000):
            ids = ch["entity_id"].values
            in_val = ch["entity_id"].isin(val_matched).values | ch["business_address"].str.contains(val_re).values
            samp = (~in_val) & (stable_frac(ids) < args.train_query_frac)
            ch = ch.assign(split=np.where(in_val, "val", "train"))
            parts.append(ch[in_val | samp])
    q = pd.concat(parts, ignore_index=True)
    del parts
    need = set(q["entity_id"].values)
    q_truth = {}
    for s, v in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        if v:
            for x in v.split(","):
                if x in need:
                    q_truth[x] = s
    del gt
    split_counts = q["split"].value_counts().to_dict()
    log("queries: train", split_counts.get("train", 0),
        "val", split_counts.get("val", 0))

    ckpt_dir = os.path.join(args.model_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    done_path = os.path.join(ckpt_dir, "done.json")
    done = json.load(open(done_path)) if os.path.exists(done_path) else []
    for country in sorted(set(q["country"].unique())):
        if country in done:
            log("country", country, "already checkpointed, skipping")
            q = q[q["country"] != country].reset_index(drop=True)
            continue
        if s1_counts.get(country, 0) == 0:
            log("country", country, "has queries but no Source 1 records")
            done.append(country)
            json.dump(done, open(done_path, "w"))
            continue
        log("loading Source 1 for", country)
        s1_data = {
            name: np.empty(s1_counts[country], dtype=object)
            for name in ("entity_id", "country", "name", "core", "addr", "nums")
        }
        s1_data["deva"] = np.empty(s1_counts[country], dtype=np.int8)
        s1_pos = 0
        for ch in read_tsv(s1_path, chunksize=100_000):
            sub = ch[ch["country"] == country]
            if len(sub):
                part = prepare(sub)
                end = s1_pos + len(part)
                for name in s1_data:
                    s1_data[name][s1_pos:end] = part[name].values
                s1_pos = end
                del part
            del sub, ch
        if s1_pos != s1_counts[country]:
            raise RuntimeError(f"Source 1 row count changed while loading {country}")
        s1c = pd.DataFrame(s1_data, copy=False)
        del s1_data
        qraw = q[q["country"] == country].reset_index(drop=True)
        split = qraw["split"].values
        qp = prepare(qraw)
        del qraw
        q_idx = np.arange(len(qp), dtype=np.int64)
        log("country", country, "S1", len(s1c), "queries", len(q_idx))
        index = BlockIndex(s1c, cap_frac=CFG["cap_frac"], cap_abs=CFG["cap_abs"])
        n_chunks = 0
        for ci, start in enumerate(range(0, len(q_idx), 50_000)):
            cpath = os.path.join(ckpt_dir, f"X_{safe(country)}_{ci:03d}.pkl")
            mpath = os.path.join(ckpt_dir, f"M_{safe(country)}_{ci:03d}.pkl")
            if os.path.exists(cpath):
                n_chunks += 1
                continue
            sub = q_idx[start:start + 50_000]
            cand, X = candidates_and_features(index, s1c, qp.iloc[sub].reset_index(drop=True))
            if X is None:
                continue
            qid = qp["entity_id"].values[sub[cand["qi"].values]]
            sid = s1c["entity_id"].values[cand["si"].values]
            meta = pd.DataFrame({"qid": qid, "sid": sid,
                                 "split": split[sub[cand["qi"].values]]})
            meta["y"] = (pd.Series(qid).map(q_truth).values == sid).astype(np.int8)
            X.to_pickle(cpath)
            meta.to_pickle(mpath)
            n_chunks += 1
            log(f"  {country} chunk {ci}: {len(meta)} pairs written")
        del index
        del qp, q_idx, split, s1c
        q = q[q["country"] != country].reset_index(drop=True)
        done.append(country)
        json.dump(done, open(done_path, "w"))
        log("country", country, f"checkpointed ({n_chunks} chunks)")

    del q

    Xs = [pd.read_pickle(os.path.join(ckpt_dir, f)) for f in sorted(os.listdir(ckpt_dir)) if f.startswith("X_")]
    metas = [pd.read_pickle(os.path.join(ckpt_dir, f)) for f in sorted(os.listdir(ckpt_dir)) if f.startswith("M_")]
    X = pd.concat(Xs, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    del Xs, metas
    feats = list(X.columns)
    params = dict(LGB_PARAMS)
    if args.threads:
        params["num_threads"] = args.threads

    trm = (meta["split"] == "train").values
    log("training on", trm.sum(), "pairs, positives", meta["y"].values[trm].sum())
    model = lgb.train(params, lgb.Dataset(X[trm], meta["y"].values[trm]), N_ROUNDS)

    log("validating")
    vm = ~trm
    pv = model.predict(X[vm])
    mv = meta[vm].reset_index(drop=True)
    b = best_per_query(pd.factorize(mv["qid"])[0], pv)
    best = mv.iloc[b].assign(p=pv[b])
    val_ids = sorted(val_s1_ids)
    results = {}
    for tau in np.arange(0.30, 0.96, 0.05):
        sel = best[best["p"] >= tau]
        pred = {}
        for s, x in zip(sel["sid"].values, sel["qid"].values):
            pred.setdefault(s, set()).add(x)
        results[round(float(tau), 2)] = f05_macro(pred, val_truth, val_ids)
        log(f"  tau={tau:.2f}  macro F0.5={results[round(float(tau), 2)]:.4f}")
    tau = max(results, key=results.get)
    # blocking recall ceiling on validation queries
    hit = mv.groupby("qid")["y"].max()
    log(f"best tau={tau}  F0.5={results[tau]:.4f}  candidate recall="
        f"{hit.sum() / max(1, len(val_matched)):.4f}")

    log("refitting on train + validation")
    model = lgb.train(params, lgb.Dataset(X, meta["y"].values), N_ROUNDS)
    model.save_model(os.path.join(args.model_dir, "lgbm.txt"))
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump({"threshold": tau, "features": feats, "cfg": CFG,
                   "val_f05": results, "n_rounds": N_ROUNDS}, f, indent=2)
    imp = pd.Series(model.feature_importance("gain"), feats).sort_values(ascending=False)
    log("top features", imp.head(15).round(0).to_dict())
    log("done")


if __name__ == "__main__":
    main()
