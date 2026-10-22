"""Candidate generation (blocking).

Every Source-2 / Source-3 record is a *query*; every Source-1 record is in the
*index*.  Each record is turned into a bag of blocking tokens:

  n:<skeleton>   consonant skeleton of each core-name token
  c:<skeleton>   skeleton of the whole core name with spaces removed
                 (catches 'grantinterests.com' vs 'Grant Interests')
  a:<skeleton>   skeleton of each address word
  d:<number>     each number in the address (leading zeros stripped)

Tokens are weighted by IDF (computed on Source 1, per country) and very frequent
tokens are dropped (they carry little signal and blow up the join).  The score
of a (query, S1) pair is the IDF-weighted overlap, computed as a sparse matrix
product in chunks, and we keep the top-K S1 records for each query.
Blocking is always done inside the same country string, so unseen countries
(e.g. France) are handled exactly like the others.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zlib
from array import array

from normalize import (clean_name, core_name, address_tokens, skeleton,
                       has_devanagari)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalised columns used by blocking + features."""
    out = pd.DataFrame({"entity_id": df["entity_id"].values,
                        "country": df["country"].fillna("").astype(str).values})
    names = df["business_name"].fillna("").astype(str).values
    addrs = df["business_address"].fillna("").astype(str).values
    ctry = out["country"].values
    nc, nk, aw, an, dv = [], [], [], [], []
    for n, a, c in zip(names, addrs, ctry):
        nc.append(clean_name(n))
        nk.append(core_name(n))
        w, d = address_tokens(a, c)
        aw.append(" ".join(w))
        an.append(" ".join(d))
        dv.append(has_devanagari(n))
    out["name"] = nc
    out["core"] = nk
    out["addr"] = aw
    out["nums"] = an
    out["deva"] = np.array(dv, dtype=np.int8)
    return out


def record_tokens(core: str, addr: str, nums: str):
    """Blocking keys of one record.  Returns (name_keys, other_keys)."""
    nk, ok = set(), set()
    ct = core.split()
    nsk = [sk for sk in (skeleton(t) for t in ct) if len(sk) >= 2]
    for sk in nsk:
        nk.add("n:" + sk)
    if ct:
        cc = skeleton("".join(ct))
        if len(cc) >= 4:
            nk.add("c:" + cc)
    ask = {sk for sk in (skeleton(t) for t in addr.split()) if len(sk) >= 2}
    nums_s = set(nums.split())
    for sk in ask:
        ok.add("a:" + sk)
    for d in nums_s:
        ok.add("d:" + d)
    # name x address pair keys: selective even when both parts are common
    for sk in nsk[:3]:
        for a in ask:
            ok.add("p:" + sk + "|" + a)
        for d in nums_s:
            ok.add("q:" + sk + "|" + d)
    return nk, ok


NBUCKETS = 1 << 22  # fixed-size hashed vocabulary: memory does NOT grow with
                    # the number of distinct blocking keys, only ever a few
                    # arrays of this length (~tens of MB), regardless of how
                    # many records / countries we index.


def _bucket(tok: str) -> int:
    return zlib.crc32(tok.encode()) % NBUCKETS


def _hash_keys(frame):
    """Hashes blocking keys of every record into fixed buckets.
    Returns (buckets, is_name_key, indptr)."""
    H = array("i")
    K = array("b")
    ptr = array("q", [0])
    for c, a, d in zip(frame["core"].values, frame["addr"].values, frame["nums"].values):
        nk, ok = record_tokens(c, a, d)
        H.extend(_bucket(t) for t in nk)
        K.extend(b"\x01" * len(nk))
        H.extend(_bucket(t) for t in ok)
        K.extend(b"\x00" * len(ok))
        ptr.append(len(H))
    return (np.frombuffer(H, dtype=np.int32).copy(),
            np.frombuffer(K, dtype=np.int8).copy(),
            np.frombuffer(ptr, dtype=np.int64).copy())


class BlockIndex:
    """Inverted index over the Source-1 records of ONE country.

    Uses the hashing trick (fixed NBUCKETS columns) so memory stays bounded
    however many distinct name/address tokens a country's records contain.
    Name keys may be more frequent (cap_frac * n) than address / pair keys
    (cap_abs) before they are dropped."""

    def __init__(self, s1: pd.DataFrame, cap_frac=0.0025, cap_min=50, cap_abs=300):
        self.n1 = len(s1)
        H, K, ptr = _hash_keys(s1)
        cnt = np.bincount(H, minlength=NBUCKETS)
        is_name = np.zeros(NBUCKETS, bool)
        is_name[H[K == 1]] = True
        cap_n = max(cap_min, int(cap_frac * self.n1))
        keep = (cnt > 0) & np.where(is_name, cnt <= cap_n, cnt <= cap_abs)
        self.idf = np.zeros(NBUCKETS, np.float32)
        self.idf[keep] = np.log(self.n1 / cnt[keep])
        rows = np.repeat(np.arange(self.n1, dtype=np.int32), np.diff(ptr))
        m = keep[H]
        # index side is binary; the query side carries the IDF weight
        self.I = sp.csr_matrix((np.ones(int(m.sum()), np.float32), (H[m], rows[m])),
                               shape=(NBUCKETS, self.n1), dtype=np.float32)
        del H, K, ptr, rows, m, cnt, is_name, keep

    def query_matrix(self, q: pd.DataFrame):
        H, _, ptr = _hash_keys(q)
        rows = np.repeat(np.arange(len(q), dtype=np.int32), np.diff(ptr))
        w = self.idf[H]
        m = w > 0
        Q = sp.csr_matrix((w[m], (rows[m], H[m])), shape=(len(q), NBUCKETS), dtype=np.float32)
        Q.sum_duplicates()
        return Q

    def search(self, q: pd.DataFrame, k=20, chunk=4000):
        """Top-k Source-1 rows (positions in the index frame) for each query row."""
        Q = self.query_matrix(q)
        out_q, out_s, out_v, out_r = [], [], [], []
        for start in range(0, Q.shape[0], chunk):
            S = (Q[start:start + chunk] @ self.I).tocsr()
            nnz_per = np.diff(S.indptr)
            rows = np.repeat(np.arange(S.shape[0]), nnz_per)
            order = np.lexsort((-S.data, rows))
            rank = np.arange(len(order)) - np.repeat(S.indptr[:-1], nnz_per)
            keep = rank < k
            sel = order[keep]
            out_q.append(rows[sel] + start)
            out_s.append(S.indices[sel])
            out_v.append(S.data[sel])
            out_r.append(rank[keep])
        cand = pd.DataFrame({"qi": np.concatenate(out_q).astype(np.int64),
                             "si": np.concatenate(out_s).astype(np.int64),
                             "bscore": np.concatenate(out_v).astype(np.float32),
                             "brank": np.concatenate(out_r).astype(np.int16)})
        best = cand.groupby("qi")["bscore"].transform("max")
        cand["bgap"] = (best - cand["bscore"]).astype(np.float32)
        second = cand[cand.brank == 1].set_index("qi")["bscore"]
        cand["b2"] = cand["qi"].map(second).fillna(0).astype(np.float32)
        return cand
