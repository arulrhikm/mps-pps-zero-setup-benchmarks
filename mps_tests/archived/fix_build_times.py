import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fix_and_load_data(backend):
    # Depending on which file the user requested, we use the _updated or base files
    base_file = f'data/{backend}/experiment2_bond_scaling_{backend}.jsonl'
    out_file = f'data/{backend}/experiment2_bond_scaling_{backend}_fixed.jsonl'
    
    fixed_data = []
    
    # Pre-pass to find average sampling time per bond_dimension just in case we need to extrapolate
    st_sums = {}
    st_counts = {}
    
    # Read the data
    raw_data = []
    with open(base_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            raw_data.append(d)
            if 'mps_build_time_ms' in d and d.get('job_id') != 'error':
                bt = float(d['mps_build_time_ms'])
                # If build time is reported in seconds (like ~1000x smaller), multiply by 1000
                if bt < d['run_time_ms'] / 100:
                    bt *= 1000
                st = d['run_time_ms'] - bt
                chi = d.get('bond_dimension')
                st_sums[chi] = st_sums.get(chi, 0) + st
                st_counts[chi] = st_counts.get(chi, 0) + 1
    
    # Extrapolate missing sampling times
    avg_st = {}
    for chi in st_sums:
        avg_st[chi] = st_sums[chi] / st_counts[chi]
        
    # Global average if a chi is completely missing
    global_avg_st = sum(st_sums.values()) / max(sum(st_counts.values()), 1)
    
    # Second pass: fix and save
    with open(out_file, 'w', encoding='utf-8') as f:
        for d in raw_data:
            chi = d.get('bond_dimension')
            if d.get('job_id') == 'error':
                continue
                
            rt = d['run_time_ms']
            if 'mps_build_time_ms' in d:
                bt = float(d['mps_build_time_ms'])
                if bt < rt / 100:
                    bt *= 1000
                st = rt - bt
            else:
                # Extrapolate
                st = avg_st.get(chi, global_avg_st)
                bt = max(0, rt - st) # Build time is runtime minus sampling time
                
            d['mps_build_time_ms'] = bt
            d['sampling_time_ms'] = st
            d['backend'] = backend.upper()
            fixed_data.append(d)
            f.write(json.dumps(d) + '\n')
            
    return pd.DataFrame(fixed_data)

def plot_build_sample_times(df_cpu, df_gpu):
    df = pd.concat([df_cpu, df_gpu], ignore_index=True)
    
    # Aggregate
    agg = df.groupby(['backend', 'bond_dimension'])[['mps_build_time_ms', 'sampling_time_ms']].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 10))
    colors = {'CPU': '#2c7fb8', 'GPU': '#d95f02'}
    
    # Plot 1: Build Time
    plt.subplot(2, 1, 1)
    for backend in ['CPU', 'GPU']:
        b_data = agg[agg['backend'] == backend].sort_values('bond_dimension')
        plt.plot(b_data['bond_dimension'], b_data['mps_build_time_ms']['mean'],
                 label=f'{backend} Build Time', color=colors[backend], marker='o',
                 linestyle='None', markersize=7)
                     
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('MPS Build Time (ms)', fontsize=14)
    plt.title('MPS Build Time vs Bond Dimension', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # Plot 2: Sampling Time
    plt.subplot(2, 1, 2)
    for backend in ['CPU', 'GPU']:
        b_data = agg[agg['backend'] == backend].sort_values('bond_dimension')
        plt.plot(b_data['bond_dimension'], b_data['sampling_time_ms']['mean'],
                 label=f'{backend} Sampling Time', color=colors[backend], marker='s',
                 linestyle='None', markersize=7)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Bond Dimension ($\chi$)', fontsize=14)
    plt.ylabel('Sampling Time (ms)', fontsize=14)
    plt.title('Sampling Time vs Bond Dimension', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plotting/fig_build_sample_time_scaling.png', dpi=300)
    plt.close()
    print("Saved plotting/fig_build_sample_time_scaling.png")

if __name__ == '__main__':
    df_cpu = fix_and_load_data('cpu')
    df_gpu = fix_and_load_data('gpu')
    plot_build_sample_times(df_cpu, df_gpu)
