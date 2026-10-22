"""Memory-conscious structural checker for the two output TSV files.

Checks Source1 row coverage, paired-row alignment, ID prefixes, duplicate IDs
within lists, and that every reported match is also a candidate. It deliberately
does not load all test Source2/3 IDs; use the official challenge validator for
its optional exact ID-existence check when the machine has enough RAM.

    python src/check_outputs.py --out-dir output --test-dir dataset/test
"""
import argparse
import os
import sys

EXPECTED = {
    "candidate_pairs.tsv": "candidate_entity_ids",
    "matching_results.tsv": "matched_entity_ids",
}


def source1_ids(path):
    """Read the required Source1 IDs; this is much smaller than all query IDs."""
    with open(path, encoding="utf-8") as f:
        f.readline()
        return {line.split("\t", 1)[0].rstrip("\r\n") for line in f if line.strip()}


def read_header(f, path, expected, errors):
    line = f.readline()
    header = line.rstrip("\r\n").split("\t") if line else []
    if header != ["source1_entity_id", expected]:
        errors.append(f"{path}: expected header ['source1_entity_id', '{expected}'], got {header}")
        return False
    return True


def parse_row(line, path, row_num, errors):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2:
        errors.append(f"{path}:{row_num}: expected exactly 2 tab-separated columns")
        return None
    sid, raw_ids = fields
    ids = raw_ids.split(",") if raw_ids else []
    if len(ids) != len(set(ids)):
        errors.append(f"{path}:{row_num}: repeated ID in list for {sid}")
    bad_prefix = [x for x in ids if not x.startswith(("S2-", "S3-"))]
    if bad_prefix:
        errors.append(f"{path}:{row_num}: non-Source2/3 ID(s) {bad_prefix[:3]}")
    return sid, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--test-dir", default="dataset/test")
    args = ap.parse_args()

    expected_path = os.path.join(args.test_dir, "test_source1.tsv")
    if not os.path.isfile(expected_path):
        print(f"Missing Source1 test file: {expected_path}")
        sys.exit(1)
    required = source1_ids(expected_path)
    candidate_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(args.out_dir, "matching_results.tsv")
    errors = []
    seen = set()
    rows = candidate_rows = candidate_ids = matched_rows = matched_ids = 0

    try:
        with open(candidate_path, encoding="utf-8") as cf, open(matching_path, encoding="utf-8") as mf:
            c_ok = read_header(cf, candidate_path, EXPECTED["candidate_pairs.tsv"], errors)
            m_ok = read_header(mf, matching_path, EXPECTED["matching_results.tsv"], errors)
            if not (c_ok and m_ok):
                sys.exit(1)

            row_num = 2
            while True:
                c_line, m_line = cf.readline(), mf.readline()
                if not c_line and not m_line:
                    break
                if not c_line or not m_line:
                    errors.append("candidate and matching files have different row counts")
                    break
                c = parse_row(c_line, candidate_path, row_num, errors)
                m = parse_row(m_line, matching_path, row_num, errors)
                row_num += 1
                if c is None or m is None:
                    continue
                c_sid, c_ids = c
                m_sid, m_ids = m
                rows += 1
                if c_sid != m_sid:
                    errors.append(f"row {row_num - 1}: Source1 IDs differ between files ({c_sid}, {m_sid})")
                if c_sid in seen:
                    errors.append(f"duplicate Source1 row: {c_sid}")
                seen.add(c_sid)
                if c_sid not in required:
                    errors.append(f"Source1 ID not present in test Source1: {c_sid}")
                if not set(m_ids).issubset(c_ids):
                    errors.append(f"row {row_num - 1}: match is missing from candidate list for {m_sid}")
                if c_ids:
                    candidate_rows += 1
                    candidate_ids += len(c_ids)
                if m_ids:
                    matched_rows += 1
                    matched_ids += len(m_ids)
                if len(errors) >= 25:
                    break
    except OSError as exc:
        print(f"Could not read output files: {exc}")
        sys.exit(1)

    missing = required - seen
    extra = seen - required
    if missing:
        errors.append(f"{len(missing)} Source1 rows missing from outputs")
    if extra:
        errors.append(f"{len(extra)} extra Source1 rows in outputs")

    print(f"Source1 rows: {rows}  rows with candidates: {candidate_rows}  candidate IDs: {candidate_ids}")
    print(f"Rows with matches: {matched_rows}  matched IDs: {matched_ids}")
    print("NOTE: exact Source2/3 ID existence is not checked by this memory-light pass.")
    if errors:
        print("FAIL")
        print("\n".join(errors[:25]))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
