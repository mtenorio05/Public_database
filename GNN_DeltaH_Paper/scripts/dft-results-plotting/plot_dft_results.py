"""
plot_deft_results.py
-----------------------
Plots ΔH_mix distributions grouped by Co concentration:
    (a) binary pathways
    (b) ternary pathways

Concentration is determined from the Co count in the HEA structure
(pathway[0][-1]) of each row.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter
from pathlib import Path

# ── Font ──────────────────────────────────────────────────────────────────────
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14

def get_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
main_path = str(Path(__file__).parent)
BINARY_DF_PATH  = f'{main_path}/binary_df.pkl'
TERNARY_DF_PATH = f'{main_path}/ternary_df.pkl'
OUTPUT_PATH     = f'{main_path}/dft_results_boxplot.png'
Y_SCALE         = 10.0   # shift applied to H_mix when graphs were created

# Co concentration thresholds (Co atom count out of 864 HEA atoms)
CO_MID_COUNT = 304   # exact count for Co_35
PANEL_LABELS = ['(a)', '(b)']
CONC_LABELS  = [r'$Co_{15}$', r'$Co_{35}$', r'$Co_{55}$']
# ─────────────────────────────────────────────────────────────────────────────

def get_co_concentration(pathway):
    """Determine Co concentration group from the HEA structure (structure 0)."""
    hea_atoms = pathway[0][-1]
    co_count  = Counter(hea_atoms.get_chemical_symbols())['Co']
    if co_count == CO_MID_COUNT:
        return 'Co35'
    elif co_count > CO_MID_COUNT:
        return 'Co55'
    else:
        return 'Co15'

def load_delta_hmix_by_concentration(df_path):
    """Load a pathway df and return {conc: [delta_H_mix, ...]}."""
    df = get_pickle_file(df_path)
    data = {'Co15': [], 'Co35': [], 'Co55': []}
    for _, row in df.iterrows():
        conc = get_co_concentration(row['pathway'])
        y = row['y']
        y = y.item() if hasattr(y, 'item') else float(y)
        data[conc].append(y - Y_SCALE)
    return data

print("Loading binary_df.pkl...")
panel_a_data = load_delta_hmix_by_concentration(BINARY_DF_PATH)
print(f"  Co15: {len(panel_a_data['Co15'])}, "
      f"Co35: {len(panel_a_data['Co35'])}, "
      f"Co55: {len(panel_a_data['Co55'])}")

print("Loading ternary_df.pkl...")
panel_b_data = load_delta_hmix_by_concentration(TERNARY_DF_PATH)
print(f"  Co15: {len(panel_b_data['Co15'])}, "
      f"Co35: {len(panel_b_data['Co35'])}, "
      f"Co55: {len(panel_b_data['Co55'])}")

# ── Plot ──────────────────────────────────────────────────────────────────────
def make_panel(ax, data_dict, panel_label):
    box_data = [data_dict[k] for k in ['Co15', 'Co35', 'Co55']]
    bp = ax.boxplot(box_data,
                    patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4,
                                    color='black', linestyle='none'))
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_xticks(range(1, len(CONC_LABELS) + 1))
    ax.set_xticklabels(CONC_LABELS, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(panel_label, loc='left', fontsize=14, pad=8)

fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

make_panel(axes[0], panel_a_data, PANEL_LABELS[0])
make_panel(axes[1], panel_b_data, PANEL_LABELS[1])

axes[0].set_ylabel(r'$\Delta H_{mix}$ (eV/atom)', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_PATH.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved {OUTPUT_PATH.replace('.png', '.pdf')} and {OUTPUT_PATH}")
