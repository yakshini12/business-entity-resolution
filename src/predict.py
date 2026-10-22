"""Run the trained pipeline on the test set and write the two submission files.

  output/candidate_pairs.tsv  - every (S1, S2/S3) pair fed to the model
  output/matching_results.tsv - final matches

Each S2/S3 record is assigned to at most ONE Source-1 entity: its highest-scoring
candidate, and only if the model probability clears the tuned threshold.

Memory-lean: queries are split by country into temporary files first, then
processed in chunks against a per-country Source-1 index.

Usage:
  python src/predict.py --data-dir dataset --model-dir models --out-dir output
"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import BlockIndex, prepare  # noqa: E402
from pipeline import CFG, best_per_query, candidates_and_features, read_tsv  # noqa: E402

DEFAULT_QUERY_CHUNK = 50_000
EDGE_SHARD_SIZE = 25_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def safe_name(c):
    return "".join(ch if ch.isalnum() else "_" for ch in c) or "EMPTY"


def split_queries_by_country(paths, tmp):
    """Streams S2/S3 files once, writing one TSV per country."""
    os.makedirs(tmp, exist_ok=True)
    writers, handles, files = {}, {}, {}
    max_id_bytes = 0
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            header = f.readline().rstrip("\n").rstrip("\r")
            cols = header.split("\t")
            ci = cols.index("country")
            ei = cols.index("entity_id")
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                parts = line.split("\t")
                c = parts[ci] if ci < len(parts) else ""
                if ei < len(parts):
                    max_id_bytes = max(max_id_bytes, len(parts[ei].encode("utf-8")))
                if c not in handles:
                    fn = os.path.join(tmp, f"q_{safe_name(c)}.tsv")
                    handles[c] = open(fn, "w", encoding="utf-8")
                    handles[c].write(header + "\n")
                    files[c] = fn
                handles[c].write(line + "\n")
    for h in handles.values():
        h.close()
    return files, max(1, max_id_bytes)


def split_source1_by_country(path, tmp):
    """Streams Source 1 into country files and records their row counts."""
    os.makedirs(tmp, exist_ok=True)
    handles, files, counts = {}, {}, {}
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").rstrip("\r")
        cols = header.split("\t")
        ci = cols.index("country")
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            country = parts[ci] if ci < len(parts) else ""
            if country not in handles:
                fn = os.path.join(tmp, f"s1_{safe_name(country)}.tsv")
                handles[country] = open(fn, "w", encoding="utf-8")
                handles[country].write(header + "\n")
                files[country] = fn
                counts[country] = 0
            handles[country].write(line + "\n")
            counts[country] += 1
    for h in handles.values():
        h.close()
    return files, counts


def write_sharded_edges(sipos, qids, edge_dir, prefix, dtype, shard_size, handles):
    """Writes one query-to-Source1 edge batch into bounded disk shards."""
    sipos = np.asarray(sipos, dtype=np.int32)
    if len(sipos) == 0:
        return
    qids = np.asarray(qids, dtype=dtype["qid"])
    shard = sipos // shard_size
    order = np.argsort(shard, kind="stable")
    sorted_shard = shard[order]
    bounds = np.flatnonzero(np.r_[True, sorted_shard[1:] != sorted_shard[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        shard_id = int(sorted_shard[a])
        h = handles.get(shard_id)
        if h is None:
            h = open(os.path.join(edge_dir, f"{prefix}_{shard_id:05d}.bin"), "wb")
            handles[shard_id] = h
        rec = np.empty(b - a, dtype=dtype)
        rec["si"] = sipos[order[a:b]]
        rec["qid"] = qids[order[a:b]]
        h.write(rec.tobytes())


def aggregate_edge_shard(path, dtype):
    """Reads one small edge shard and builds id lists grouped by Source 1."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    rec = np.fromfile(path, dtype=dtype)
    order = np.argsort(rec["si"], kind="stable")
    si = rec["si"][order]
    qid = rec["qid"][order]
    del rec, order
    bounds = np.flatnonzero(np.r_[True, si[1:] != si[:-1], True])
    grouped = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        ids = dict.fromkeys(x.decode("utf-8") for x in qid[a:b] if x)
        grouped[int(si[a])] = ",".join(ids)
    return grouped


def load_prepared_country(path, expected_rows):
    """Prepares one Source1 country from chunks without retaining raw rows."""
    data = {
        name: np.empty(expected_rows, dtype=object)
        for name in ("entity_id", "country", "name", "core", "addr", "nums")
    }
    data["deva"] = np.empty(expected_rows, dtype=np.int8)
    pos = 0
    for ch in read_tsv(path, chunksize=100_000):
        part = prepare(ch)
        end = pos + len(part)
        for name in data:
            data[name][pos:end] = part[name].values
        pos = end
        del part, ch
    if pos != expected_rows:
        raise RuntimeError(f"Source 1 row count changed while reading {path}")
    return pd.DataFrame(data, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tmp-dir", default="tmp_predict")
    ap.add_argument("--query-chunk-size", type=int, default=DEFAULT_QUERY_CHUNK,
                    help="queries prepared at a time; lower values use less RAM")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    args = ap.parse_args()
    te = os.path.join(args.data_dir, "test")
    os.makedirs(args.out_dir, exist_ok=True)

    conf = json.load(open(os.path.join(args.model_dir, "config.json")))
    model = lgb.Booster(model_file=os.path.join(args.model_dir, "lgbm.txt"))
    feats = conf["features"]
    tau = args.threshold if args.threshold is not None else conf["threshold"]
    cfg = dict(CFG)
    cfg.update(conf.get("cfg", {}))
    log("threshold", tau)

    os.makedirs(args.tmp_dir, exist_ok=True)
    work_tmp = tempfile.mkdtemp(prefix="ber_", dir=args.tmp_dir)
    log("temporary workspace", work_tmp)
    s1_files, s1_counts = split_source1_by_country(
        os.path.join(te, "test_source1.tsv"), os.path.join(work_tmp, "source1"))
    log("Source 1 records", sum(s1_counts.values()), "by country", s1_counts)

    log("splitting queries by country")
    qfiles, max_id_bytes = split_queries_by_country(
        [os.path.join(te, "test_source2.tsv"), os.path.join(te, "test_source3.tsv")],
        os.path.join(work_tmp, "queries"))
    log("query countries:", {c: os.path.getsize(f) // 1_000_000 for c, f in qfiles.items()}, "(MB)")

    edge_dtype = np.dtype([("si", "<i4"), ("qid", f"S{max_id_bytes}")], align=False)
    candidate_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(args.out_dir, "matching_results.tsv")
    candidate_partial = candidate_path + ".partial"
    matching_partial = matching_path + ".partial"
    n_s1 = sum(s1_counts.values())
    n_candidate_rows = n_matched_rows = n_candidate_ids = n_matched_ids = 0

    log("scoring and writing country partitions")
    with open(candidate_partial, "w", encoding="utf-8", newline="") as cf, \
            open(matching_partial, "w", encoding="utf-8", newline="") as mf:
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        for country in sorted(s1_files):
            s1c = load_prepared_country(s1_files[country], s1_counts[country])
            ids = s1c["entity_id"].values
            log("country", country, "S1", len(ids))
            if country not in qfiles:
                log("country", country, "has no queries")
                for sid in ids:
                    cf.write(f"{sid}\t\n")
                    mf.write(f"{sid}\t\n")
                del ids, s1c
                continue

            edge_dir = os.path.join(work_tmp, "edges_" + safe_name(country))
            os.makedirs(edge_dir, exist_ok=True)
            index = BlockIndex(s1c, cap_frac=cfg["cap_frac"], cap_abs=cfg["cap_abs"])
            candidate_handles, matching_handles = {}, {}
            n_done = n_country_matches = 0
            for ch in read_tsv(qfiles[country], chunksize=args.query_chunk_size):
                qp = prepare(ch)
                cand, X = candidates_and_features(index, s1c, qp, cfg)
                n_done += len(ch)
                if X is not None:
                    prob = model.predict(X[feats])
                    qi, si = cand["qi"].values, cand["si"].values
                    qids = qp["entity_id"].values[qi]
                    write_sharded_edges(si, qids, edge_dir, "candidate", edge_dtype,
                                        EDGE_SHARD_SIZE, candidate_handles)
                    best = best_per_query(qi, prob)
                    best = best[prob[best] >= tau]
                    if len(best):
                        write_sharded_edges(si[best], qids[best], edge_dir, "matching",
                                            edge_dtype, EDGE_SHARD_SIZE, matching_handles)
                        n_country_matches += len(best)
                del ch
                log(f"  {country}: {n_done} queries, {n_country_matches} matched queries")
            del index
            for handle in candidate_handles.values():
                handle.close()
            for handle in matching_handles.values():
                handle.close()

            for shard_start in range(0, len(ids), EDGE_SHARD_SIZE):
                shard_id = shard_start // EDGE_SHARD_SIZE
                shard_end = min(shard_start + EDGE_SHARD_SIZE, len(ids))
                cpath = os.path.join(edge_dir, f"candidate_{shard_id:05d}.bin")
                mpath = os.path.join(edge_dir, f"matching_{shard_id:05d}.bin")
                candidates = aggregate_edge_shard(cpath, edge_dtype)
                matches = aggregate_edge_shard(mpath, edge_dtype)
                for si in range(shard_start, shard_end):
                    cids = candidates.get(si, "")
                    mids = matches.get(si, "")
                    sid = ids[si]
                    cf.write(f"{sid}\t{cids}\n")
                    mf.write(f"{sid}\t{mids}\n")
                    if cids:
                        n_candidate_rows += 1
                        n_candidate_ids += cids.count(",") + 1
                    if mids:
                        n_matched_rows += 1
                        n_matched_ids += mids.count(",") + 1
                for path in (cpath, mpath):
                    if os.path.exists(path):
                        os.remove(path)
                del candidates, matches
            del ids, s1c

    os.replace(candidate_partial, candidate_path)
    os.replace(matching_partial, matching_path)
    shutil.rmtree(work_tmp, ignore_errors=True)
    log(f"done: {n_s1} S1 rows; {n_candidate_rows} rows with candidates "
        f"({n_candidate_ids} candidate ids); {n_matched_rows} rows with matches "
        f"({n_matched_ids} matched ids)")


if __name__ == "__main__":
    main()
