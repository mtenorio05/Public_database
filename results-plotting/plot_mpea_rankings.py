'''
plot_mpea_rankings.py
===================================================================================================================
Plot stoichiometric, 2 body interaction, and 3 Body interaction features belonging only to the parent MPEA in three
different plots
'''

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from pathlib import Path


TOP5_COLORS = ["#44BB99", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF"]

# ── Font ──────────────────────────────────────────────────────────────────────
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 16

# ── Config ────────────────────────────────────────────────────────────────────
DATASET       = 'binary'      # 'binary' or 'ternary'
MODEL         = 0             # which model folder (0, 1, 2, ...)
LOG_SCALE     = False         # set True to plot y-axis in log scale
MAIN_PATH     = str(Path(__file__).parent)

# Model trained on P_x, evaluated on P_y ('binary' -> P1, 'ternary' -> P2)
TRAINED_ON    = 'ternary'
EVALUATED_ON  = DATASET
P_LABEL = {'binary': '1', 'ternary': '2'}
YLABEL = (f'($P_{{{P_LABEL[TRAINED_ON]}}} \\rightarrow '
          f'P_{{{P_LABEL[EVALUATED_ON]}}}$) Contributions')

INPUT_DF = {
    'binary':  'matched_binary_df.pkl',
    'ternary': 'matched_ternary_df.pkl',
}
OUTPUT_PREFIX = f'{DATASET}_model{MODEL}{"_log" if LOG_SCALE else ""}'

# Atomic number -> element symbol
ATOM_MAP = {26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 42: 'Mo'}

# tol light palette colors for top-5 ranked interactions
CONCENTRATIONS = ['Co Poor', 'Co Mid', 'Co Rich']
# ─────────────────────────────────────────────────────────────────────────────

def get_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def label_na(key):
    return '-'.join(ATOM_MAP.get(k, str(k)) for k in key)

def label_ea(key):
    return '-'.join(ATOM_MAP.get(k, str(k)) for k in key)

# Load comprehensinve df
matched_df = get_pickle_file(f'{MAIN_PATH}/{INPUT_DF[DATASET]}')
print(f"Loaded {len(matched_df)} graphs, concentrations: "
      f"{matched_df['co_concentration'].value_counts().to_dict()}")

# ── Compute top-5 per feature type across all graphs (by mean score) ──────────
def top5_na(df):
    """Average the per-graph mean score for each interaction type."""
    agg = {}  # key -> list of per-graph means
    for _, row in df.iterrows():
        for key, vals in row['na_hea'].items():
            agg.setdefault(key, []).append(np.mean(vals))
    means = {k: np.mean(v) for k, v in agg.items()}
    return sorted(means, key=means.get, reverse=True)[:5]

def top5_ea(df):
    """Average the per-graph mean score for each interaction type."""
    agg = {}
    for _, row in df.iterrows():
        for key, vals in row['ea_hea'].items():
            agg.setdefault(key, []).append(np.mean(vals))
    means = {k: np.mean(v) for k, v in agg.items()}
    return sorted(means, key=means.get, reverse=True)[:5]

print("NA top-5 (global):", [label_na(k) for k in top5_na(matched_df)])
print("EA top-5 (global):", [label_ea(k) for k in top5_ea(matched_df)])

# ── Helper: make a 1x3 box plot ───────────────────────────────────────────────
def make_boxplot(box_data_per_conc, labels, title, output_path, ylabel=None, color=True, layout='1x3'):
    if ylabel is None:
        ylabel = YLABEL
    if layout == '3x1':
        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharey=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for c, (concentration, box_data) in enumerate(zip(CONCENTRATIONS, box_data_per_conc)):
        ax = axes[c]

        # Compute top-5 for this concentration based on median score
        medians = {lbl: np.median(vals) if len(vals) > 0 else 0
                   for lbl, vals in zip(labels, box_data)}
        top5_local = sorted(medians, key=medians.get, reverse=True)[:5]
        local_color_map = {lbl: TOP5_COLORS[i] for i, lbl in enumerate(top5_local)} \
                          if color else {}

        bp = ax.boxplot(box_data,
                        patch_artist=True,
                        widths=0.6,
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.0),
                        capprops=dict(color='black', linewidth=1.0),
                        flierprops=dict(marker='o', markersize=2,
                                        color='black', alpha=0.4, linestyle='none'))
        for patch, lbl in zip(bp['boxes'], labels):
            face_color = local_color_map.get(lbl, 'white')
            patch.set_facecolor(face_color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_alpha(0.85 if lbl in local_color_map else 1.0)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 5 else 90, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if LOG_SCALE:
            ax.set_yscale('log')
        if layout == '3x1':
            ax.set_ylabel(f'{concentration}\n{ylabel}', fontsize=18)
        else:
            ax.set_title(concentration, fontsize=16, pad=8)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=18)

    fig.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    plt.close()
    print(f"Saved {output_path} and {output_path.replace('.png', '.pdf')}")

# ── Stoichiometric Features plot ───────────────────────────────────────────────────────────────────
atomic_order = ['Co', 'Mo', 'Fe', 'Ni', 'Cu']

ng_box_data_per_conc_sum = []
ng_box_data_per_conc_ind = []
for conc in CONCENTRATIONS:
    subset = matched_df[matched_df['co_concentration'] == conc]
    atom_scores_sum = {a: [] for a in atomic_order}
    atom_scores_ind = {a: [] for a in atomic_order}
    for _, row in subset.iterrows():
        ng_df = row['ng_df_hea'].reset_index()
        total_ng = ng_df['ng_value'].sum()   # total HEA ng score for this structure
        for atom in atomic_order:
            vals = ng_df[ng_df['atom'] == atom]['ng_value']
            atom_scores_sum[atom].append(vals.sum() / total_ng if total_ng > 0 else 0)
            atom_scores_ind[atom].extend(vals.tolist())
    ng_box_data_per_conc_sum.append([atom_scores_sum[a] for a in atomic_order])
    ng_box_data_per_conc_ind.append([atom_scores_ind[a] for a in atomic_order])

make_boxplot(
    ng_box_data_per_conc_sum,
    labels=atomic_order,
    title='Node-Grouped (NG) Scores — HEA Only (Normalized fractional contribution)',
    output_path=f'{MAIN_PATH}/{OUTPUT_PREFIX}_ng_scores_summed.png',
    color=False
)

make_boxplot(
    ng_box_data_per_conc_ind,
    labels=atomic_order,
    title='Node-Grouped (NG) Scores — HEA Only (Individual atomic scores)',
    output_path=f'{MAIN_PATH}/{OUTPUT_PREFIX}_ng_scores_individual.png',
    color=False
)

# ── 2 Body interaction features plot ───────────────────────────────────────────────────────────────────
all_na_keys = set()
for _, row in matched_df.iterrows():
    all_na_keys.update(row['na_hea'].keys())

# Global top-5 used only for ordering (coloring is per-concentration)
global_top5_na = top5_na(matched_df)
na_order = global_top5_na + sorted([k for k in all_na_keys if k not in global_top5_na],
                                    key=label_na)
na_labels = [label_na(k) for k in na_order]

na_box_data_per_conc = []
for conc in CONCENTRATIONS:
    subset = matched_df[matched_df['co_concentration'] == conc]
    key_scores = {k: [] for k in na_order}
    for _, row in subset.iterrows():
        total_na = sum(sum(v) for v in row['na_hea'].values())
        for k in na_order:
            vals = row['na_hea'].get(k, [])
            key_scores[k].append(sum(vals) / total_na if total_na > 0 else 0)
    na_box_data_per_conc.append([key_scores[k] for k in na_order])

# Build color maps using the finalized label lists — top-5 are first in na_order/ea_order
make_boxplot(
    na_box_data_per_conc,
    labels=na_labels,
    title='Two-Body (NA) Scores — HEA Only',
    output_path=f'{MAIN_PATH}/{OUTPUT_PREFIX}_na_scores.png',
    layout='3x1'
)

# ── 3 Body interaction features plot ───────────────────────────────────────────────────────────────────
all_ea_keys = set()
for _, row in matched_df.iterrows():
    all_ea_keys.update(row['ea_hea'].keys())

# Global top-15 by per-graph mean, used only for ordering
global_top5_ea = top5_ea(matched_df)
agg_ea = {}
for _, row in matched_df.iterrows():
    for k, v in row['ea_hea'].items():
        agg_ea.setdefault(k, []).append(np.mean(v))
ea_order = sorted(agg_ea, key=lambda k: np.mean(agg_ea[k]), reverse=True)[:15]
ea_labels = [label_ea(k) for k in ea_order]

ea_box_data_per_conc = []
for conc in CONCENTRATIONS:
    subset = matched_df[matched_df['co_concentration'] == conc]
    key_scores = {k: [] for k in ea_order}
    for _, row in subset.iterrows():
        total_ea = sum(sum(v) for v in row['ea_hea'].values())
        for k in ea_order:
            vals = row['ea_hea'].get(k, [])
            key_scores[k].append(sum(vals) / total_ea if total_ea > 0 else 0)
    ea_box_data_per_conc.append([key_scores[k] for k in ea_order])

make_boxplot(
    ea_box_data_per_conc,
    labels=ea_labels,
    title='Three-Body (EA) Scores — HEA Only',
    output_path=f'{MAIN_PATH}/{OUTPUT_PREFIX}_ea_scores.png',
    layout='3x1'
)
