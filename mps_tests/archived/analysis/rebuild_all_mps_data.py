"""
Rebuilds all_mps_data.jsonl by consolidating data from:
- quantum_volume_scaling.jsonl (GPU data from the original QV scaling experiment)
- cpu/*.jsonl files
- gpu/*.jsonl files

GPU and CPU data are separated with comment headers, like the existing file already does.
Old data in the 'old' subdirectory is intentionally excluded.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # analysis/
MPS_ROOT = os.path.dirname(BASE_DIR)                    # mps_tests/
DATA_DIR = os.path.join(MPS_ROOT, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "all_mps_data.jsonl")

def read_jsonl(filepath, source_label=None):
    """Read a .jsonl file and return list of dicts, adding source_file field."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                if source_label:
                    record["source_file"] = source_label
                records.append(record)
            except json.JSONDecodeError:
                print(f"  Warning: skipping invalid JSON in {filepath}: {line[:60]}")
    return records

def collect_files(subdir):
    """Return sorted list of .jsonl files in a subdirectory."""
    dirpath = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(dirpath):
        return []
    return sorted([
        f for f in os.listdir(dirpath)
        if f.endswith(".jsonl")
    ])

print("Collecting data from all .jsonl files in mps_tests...")

# ── GPU section ────────────────────────────────────────────────────────────────
gpu_sections = []

# gpu/*.jsonl files
gpu_files = collect_files("gpu")
for fname in gpu_files:
    fpath = os.path.join(DATA_DIR, "gpu", fname)
    label = f"gpu\\{fname}"
    recs = read_jsonl(fpath, source_label=label)
    print(f"  gpu/{fname}: {len(recs)} records")
    gpu_sections.append((label, recs))

# ── CPU section ────────────────────────────────────────────────────────────────
cpu_sections = []

# quantum_volume_scaling.jsonl (CPU data)
qvs_path = os.path.join(DATA_DIR, "quantum_volume_scaling.jsonl")
if os.path.exists(qvs_path):
    recs = read_jsonl(qvs_path, source_label="quantum_volume_scaling.jsonl")
    print(f"  quantum_volume_scaling.jsonl: {len(recs)} records")
    cpu_sections.append(("quantum_volume_scaling.jsonl", recs))

cpu_files = collect_files("cpu")
for fname in cpu_files:
    fpath = os.path.join(DATA_DIR, "cpu", fname)
    label = f"cpu\\{fname}"
    recs = read_jsonl(fpath, source_label=label)
    print(f"  cpu/{fname}: {len(recs)} records")
    cpu_sections.append((label, recs))

# ── Write consolidated file ───────────────────────────────────────────────────
gpu_total = sum(len(r) for _, r in gpu_sections)
cpu_total = sum(len(r) for _, r in cpu_sections)
print(f"\nWriting {gpu_total} GPU records and {cpu_total} CPU records to all_mps_data.jsonl...")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as out:
    # GPU section
    out.write("# ── GPU Data ─────────────────────────────────────────────────────────────────\n")
    for label, recs in gpu_sections:
        out.write(f"# Source: {label}\n")
        for rec in recs:
            out.write(json.dumps(rec) + "\n")

    out.write("\n")

    # CPU section
    out.write("# ── CPU Data ─────────────────────────────────────────────────────────────────\n")
    for label, recs in cpu_sections:
        out.write(f"# Source: {label}\n")
        for rec in recs:
            out.write(json.dumps(rec) + "\n")

print(f"Done! Wrote {gpu_total + cpu_total} total records to all_mps_data.jsonl")
