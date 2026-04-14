import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_shots_sweep_data():
    file_path = '../data/experiment2_shots_sweep.jsonl'
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            entry = json.loads(line)
            if 'error' in entry:
                continue
            # apply fix again just in case there are old logs remaining
            if 'mps_build_time_ms' in entry and entry['mps_build_time_ms'] is not None:
                bt = float(entry['mps_build_time_ms'])
                if bt < entry['run_time_ms'] / 100:
                    bt *= 1000
                st = max(0, entry['run_time_ms'] - bt)
                entry['mps_build_time_ms'] = bt
                entry['sampling_time_ms'] = st
            data.append(entry)
            
    df = pd.DataFrame(data)
    # Group by device and shots, keeping valid median/stds
    if df.empty:
        return df
        
    # We want robust representations, discard top 1.5x median outliers based on runtime
    def filter_outliers(g):
        if len(g) <= 1: return g
        med = g['run_time_ms'].median()
        return g[g['run_time_ms'] <= 1.5 * med]
        
    df = df.groupby(['device', 'shots'], group_keys=False).apply(filter_outliers)
    return df

def plot_shots_sweep(df):
    if df.empty:
        print("No data available to plot!")
        return

    # Aggregate
    agg = df.groupby(['device', 'shots']).agg({
        'run_time_ms': ['mean', 'std'],
        'mps_build_time_ms': ['mean', 'std'],
        'sampling_time_ms': ['mean', 'std']
    }).reset_index()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'mps.cpu': 'blue', 'mps.gpu': 'red'}
    markers_run = 'o'
    markers_build = 's'
    markers_sample = '^'

    for device in df['device'].unique():
        d_data = agg[agg['device'] == device].sort_values('shots')
        if d_data.empty: continue
        
        shots = d_data['shots']
        
        # CPU Plot
        if device == 'mps.cpu':
            ax1.plot(shots, d_data['run_time_ms']['mean'],
                     color=colors[device], marker=markers_run, linestyle='None', markersize=7, label='Total Runtime')
            ax1.plot(shots, d_data['mps_build_time_ms']['mean'],
                     color='orange', marker=markers_build, linestyle='None', markersize=7, label='Build Time')
            ax1.plot(shots, d_data['sampling_time_ms']['mean'],
                     color='green', marker=markers_sample, linestyle='None', markersize=7, label='Sampling Time')
                         
        # GPU Plot
        if device == 'mps.gpu':
            ax2.plot(shots, d_data['run_time_ms']['mean'],
                     color=colors[device], marker=markers_run, linestyle='None', markersize=7, label='Total Runtime')
            ax2.plot(shots, d_data['mps_build_time_ms']['mean'],
                     color='orange', marker=markers_build, linestyle='None', markersize=7, label='Build Time')
            ax2.plot(shots, d_data['sampling_time_ms']['mean'],
                     color='green', marker=markers_sample, linestyle='None', markersize=7, label='Sampling Time')

    # Formatting CPU
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('CPU Shots Sweep (Build vs Sampling)', fontsize=14)
    ax1.set_xlabel('Number of Shots', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)
    
    # Formatting GPU
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('GPU Shots Sweep (Build vs Sampling)', fontsize=14)
    ax2.set_xlabel('Number of Shots', fontsize=12)
    ax2.set_ylabel('Execution Time (ms)', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plot_path = 'fig_experiment2_shots_sweep.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")

if __name__ == '__main__':
    df = load_shots_sweep_data()
    plot_shots_sweep(df)
