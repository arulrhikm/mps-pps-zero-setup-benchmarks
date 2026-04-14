#!/usr/bin/env python3
"""
Correction script for Experiment 3: Add num_cx_gates to existing data
Reads existing experiment3 JSONL file and adds CX gate counts by regenerating circuits
Does NOT run any jobs - just adds the missing field
"""

import json
import sys
import os
from qiskit.circuit.library import quantum_volume

def main():
    if len(sys.argv) < 2:
        print("Usage: python add_cx_gates_experiment3.py <input_file.jsonl>")
        print("Example: python add_cx_gates_experiment3.py experiment3_depth_scaling.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_with_cx.jsonl"
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    # Read existing data
    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # Keep header comments
                entries.append(('comment', line))
            elif line:
                entries.append(('data', json.loads(line)))
    
    print(f"Found {sum(1 for t, _ in entries if t == 'data')} data entries")
    
    # Process and add CX gate counts
    processed = 0
    with open(output_file, 'w') as f:
        for entry_type, entry in entries:
            if entry_type == 'comment':
                # Write comment as-is, but update header if needed
                if 'num_gates' in entry and 'num_cx_gates' not in entry:
                    entry = entry.replace('num_gates', 'num_gates, num_cx_gates')
                f.write(entry + '\n')
            else:
                # Data entry - add num_cx_gates if missing
                if 'num_cx_gates' not in entry:
                    # Regenerate circuit to count CX gates
                    num_qubits = entry['num_qubits']
                    depth = entry['depth']
                    trial = entry['trial']
                    
                    qc = quantum_volume(num_qubits, depth, seed=42 + trial)
                    decomposed_qc = qc.decompose()
                    num_cx_gates = decomposed_qc.count_ops()['cx']
                    
                    # Add the field
                    entry['num_cx_gates'] = num_cx_gates
                    processed += 1
                
                # Write updated entry
                f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {processed} entries")
    print(f"Output written to: {output_file}")

if __name__ == '__main__':
    main()
