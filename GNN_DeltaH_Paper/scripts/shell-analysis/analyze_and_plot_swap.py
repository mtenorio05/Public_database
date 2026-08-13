"""
analyze_swap.py
-----------------
For each of the 4 swap pairs, computes per-atom ΔS (stoichiometric score change)
and analyzes how it decays as a function of neighbor shell distance
from the swapped atom using BFS.

Produces:
  - shell_analysis_results.pkl  : full per-atom delta data for all 4 swaps
  - shell_analysis_results.csv  : same in CSV
  - One plot per swap: shell decay of mean |ΔS|
  - One combined plot with all 4 swaps
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from ase.neighborlist import neighbor_list
from pathlib import Path
from collections import deque
import os

# ── Font ──────────────────────────────────────────────────────────────────────
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14

def get_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle_file(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def bfs_shells(atoms, swap_idx, cutoff=2.55):
    """BFS from swap_idx to assign shell number to every atom."""
    i_list, j_list = neighbor_list('ij', atoms, cutoff=cutoff)
    neighbors = {i: [] for i in range(len(atoms))}
    for i, j in zip(i_list, j_list):
        neighbors[i].append(j)

    shell = np.full(len(atoms), -1)
    shell[swap_idx] = 0
    queue = deque([swap_idx])
    while queue:
        current = queue.popleft()
        for nb in neighbors[current]:
            if shell[nb] == -1:
                shell[nb] = shell[current] + 1
                queue.append(nb)
    return shell

def compute_delta_S(ng_orig, ng_swap):
    """
    Compute per-atom ΔS = S_swap - S_orig matched by ase_index only.
    Species may differ at the swapped site.
    """
    merged = pd.merge(
        ng_orig[['ase_index', 'atom', 'ng_value']].rename(
            columns={'ng_value': 'S_orig', 'atom': 'atom_orig'}),
        ng_swap[['ase_index', 'ng_value']].rename(
            columns={'ng_value': 'S_swap'}),
        on='ase_index',
        how='inner'
    )
    merged['delta_S'] = merged['S_swap'] - merged['S_orig']
    return merged

main_path = str(Path(__file__).parent)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET          = 'ternary'   # 'binary' or 'ternary'
SHOW_ERROR_BARS  = False        # set False to plot only the mean line
CUTOFF           = 2.55        # neighbor cutoff in Å

config = {
    'binary': {
        'paired_df':    'path to matched dataframe',
        'unpaired_df':  'path to unmatched dataframe',
        'output_dir':   f'{main_path}/shell_analysis_binary',
    },
    'ternary': {
        'paired_df':    'path to matched dataframe',
        'output_dir':   f'{main_path}/shell_analysis_ternary',
    },
}

PAIRED_DF_PATH    = config[DATASET]['paired_df']
SWAP_RECORDS_PATH = f'{main_path}/swapped_structures/swap_records.pkl'
OUTPUT_DIR        = config[DATASET]['output_dir']
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# tol vibrant palette — hardcoded (no tol_colors package needed)
# Row 1 (co55, Fe↔Co swaps): orange
# Row 2 (co35, Mo↔Co swaps): teal
COLORS = {
    'co55_Fe_high_to_Co': '#EE8866',   # tol light orange
    'co55_Co_high_to_Fe': '#EE8866',   # tol light orange
    'co35_Mo_high_to_Co': '#77AADD',   # tol light blue
    'co35_Co_high_to_Mo': '#77AADD',   # tol light blue
}

# Plot order: row 1 = co55 swaps, row 2 = co35 swaps
PLOT_ORDER = [
    ['co55_Fe_high_to_Co', 'co55_Co_high_to_Fe'],
    ['co35_Mo_high_to_Co', 'co35_Co_high_to_Mo'],
]

TITLES = {
    'co55_Fe_high_to_Co': 'Fe (high) → Co',
    'co55_Co_high_to_Fe': 'Co (high) → Fe',
    'co35_Mo_high_to_Co': 'Mo (high) → Co',
    'co35_Co_high_to_Mo': 'Co (high) → Mo',
}
# ─────────────────────────────────────────────────────────────────────────────

paired_df    = get_pickle_file(PAIRED_DF_PATH)
swap_records = get_pickle_file(SWAP_RECORDS_PATH)

print(f"Paired df: {len(paired_df)} pairs")
print(f"Swap records: {len(swap_records)} records")
print(f"Swap records columns: {swap_records.columns.tolist()}")

# Build file_name -> swap indices mapping from swap_records
# output_file basename (without .vasp) = file_name
swap_records['file_name'] = swap_records['output_file'].apply(
    lambda p: os.path.basename(p).replace('.vasp', '')
)
swap_idx_map = {}
for _, rec in swap_records.iterrows():
    swap_idx_map[rec['file_name']] = {
        'atom_a_index':   int(rec['atom_a_index']),
        'atom_a_element': rec['atom_a_element'],
        'atom_a_score':   rec['atom_a_score'],
        'atom_b_index':   int(rec['atom_b_index']),
        'atom_b_element': rec['atom_b_element'],
        'atom_b_score':   rec['atom_b_score'],
    }

print(f"\nSwap index map:")
for fn, info in swap_idx_map.items():
    print(f"  {fn}: {info['atom_a_element']}[{info['atom_a_index']}] ↔ "
          f"{info['atom_b_element']}[{info['atom_b_index']}]")

# ── Shell analysis for each pair — compute stats first ───────────────────────
all_results = []
shell_stats = {}  # file_name -> (shell_means, shell_stds)

for _, pair in paired_df.iterrows():
    file_name = pair['file_name']
    print(f"\n── {file_name} ──────────────────────────────────────────────────")

    swap_info = swap_idx_map.get(file_name)
    if swap_info is None:
        print(f"  WARNING: no swap record for {file_name}")
        continue

    hea_atoms = pair['pathway_swap'][0][-1].copy()
    swap_idx  = swap_info['atom_a_index']
    print(f"  Swap atom: {swap_info['atom_a_element']}[{swap_idx}] ↔ "
          f"{swap_info['atom_b_element']}[{swap_info['atom_b_index']}]")

    shell    = bfs_shells(hea_atoms, swap_idx, cutoff=CUTOFF)
    delta_df = compute_delta_S(pair['ng_df_hea_orig'], pair['ng_df_hea_swap'])
    delta_df['shell']           = delta_df['ase_index'].map(lambda i: shell[i] if i < len(shell) else -1)
    delta_df['file_name']       = file_name
    delta_df['co_concentration']= pair['co_concentration']
    all_results.append(delta_df)

    max_shell   = min(int(delta_df['shell'].max()), 5)
    shell_means, shell_stds = [], []
    shell_mins,  shell_maxs = [], []
    for s in range(max_shell + 1):
        vals = delta_df[delta_df['shell'] == s]['delta_S'].values
        shell_means.append(vals.mean() if len(vals) > 0 else 0)
        shell_stds.append(vals.std()   if len(vals) > 0 else 0)
        shell_mins.append(vals.min()   if len(vals) > 0 else 0)
        shell_maxs.append(vals.max()   if len(vals) > 0 else 0)
    shell_stats[file_name] = (shell_means, shell_mins, shell_maxs)
    print(f"  Shell means: {[f'{m:.6f}' for m in shell_means]}")

# ── Combined 2×2 plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row', sharex='col')

for r, row_files in enumerate(PLOT_ORDER):
    for c, file_name in enumerate(row_files):
        if file_name not in shell_stats:
            continue
        ax = axes[r][c]
        shell_means, shell_mins, shell_maxs = shell_stats[file_name]
        color = COLORS[file_name]
        means  = np.array(shell_means)
        err_lo = means - np.array(shell_mins)
        err_hi = np.array(shell_maxs) - means

        if SHOW_ERROR_BARS:
            ax.errorbar(range(len(shell_means)), shell_means,
                        yerr=[err_lo, err_hi],
                        fmt='o-', color=color, linewidth=2,
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        markeredgecolor='black', markeredgewidth=0.8,
                        markersize=7, label='Mean ΔS [min, max]')
        else:
            ax.plot(range(len(shell_means)), shell_means,
                    'o-', color=color, linewidth=2,
                    markeredgecolor='black', markeredgewidth=0.8,
                    markersize=7, label='Mean ΔS')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if r == 1:
            ax.set_xlabel('Neighbor Shell', fontsize=14)
        if c == 0:
            ax.set_ylabel('Mean ΔS', fontsize=14)

fig.suptitle('Score Change by Neighbor Shell', fontsize=14, y=1.01)
plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, 'shell_decay_combined.pdf')
fig.savefig(combined_path, dpi=300, bbox_inches='tight')
fig.savefig(combined_path.replace('.pdf', '.png'), dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved combined plot: {combined_path}")

# Save full results
all_results_df = pd.concat(all_results, ignore_index=True)
save_pickle_file(os.path.join(OUTPUT_DIR, 'shell_analysis_results.pkl'), all_results_df)
all_results_df.drop(columns=['atom_orig']).to_csv(
    os.path.join(OUTPUT_DIR, 'shell_analysis_results.csv'), index=False, sep=';')
print(f"Saved results: {OUTPUT_DIR}/shell_analysis_results.pkl")
print(f"\nColumns: {all_results_df.columns.tolist()}")
print(all_results_df.head())
