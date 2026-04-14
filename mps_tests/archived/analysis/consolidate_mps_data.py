import json
import glob
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_MPS_ROOT = os.path.dirname(_BASE)
_DATA_DIR = os.path.join(_MPS_ROOT, "data")
OUTPUT_FILE = os.path.join(_DATA_DIR, "all_mps_data.jsonl")

def consolidate():
    print(f"Consolidating JSONL files into {OUTPUT_FILE}...")
    
    # Clean output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    all_records = []
    
    # Find all jsonl files
    # recursive glob might need **
    files = glob.glob(os.path.join(_DATA_DIR, "**", "*.jsonl"), recursive=True)
    
    count = 0
    for fpath in files:
        # Skip output file itself if it exists (though we deleted it)
        if fpath == OUTPUT_FILE: continue
        # Skip "old" directories or "archive"
        if "old" in fpath or "archive" in fpath: continue
        
        print(f"  Reading {fpath}...")
        
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                try:
                    data = json.loads(line)
                    # Add default shots if missing
                    if 'shots' not in data:
                        data['shots'] = 1000
                    
                    # Add source file for traceability
                    data['source_file'] = fpath
                    
                    all_records.append(data)
                    count += 1
                except:
                    pass

    print(f"Found {count} records.")
    
    # Write to output
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Consolidated MPS Data\n")
        f.write("# Fields: ... plus 'shots' (default 1000)\n")
        for r in all_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    consolidate()
