"""
match_swap_to_original.py
-------------------------
Re-runs the atom_swap selection to identify exactly which original
pathways were used, then finds their counterparts in matched_swapped_df
via graph fingerprint.
"""

import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

def get_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle_file(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def decomp_fingerprint(pathway, precision=4):
    """Fingerprint using only decomposition structures (skip HEA at index 0)."""
    h = hashlib.md5()
    for struct_list in pathway[1:]:
        atoms = struct_list[-1]
        numbers   = np.sort(atoms.get_atomic_numbers())
        positions = np.sort(np.round(atoms.get_positions(), precision))
        h.update(numbers.tobytes())
        h.update(positions.tobytes())
    return h.hexdigest()

def pick_pathway(matched_df, concentration):
    subset = matched_df[matched_df['co_concentration'] == concentration]
    return subset.iloc[0]

main_path = str(Path(__file__).parent)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET = 'binary'   # 'binary' or 'ternary'

config = {
    'binary': {
        'matched_orig': 'path to comprehensive dataframe',
        'matched_swap': 'path to comprehensive dataframe for swapped structures',
        'output':       f'{main_path}/swap_to_original_match_binary.pkl',
    },
    'ternary': {
        'matched_orig': 'path to comprehensive dataframe',
        'matched_swap': 'path to comprehensive dataframe for swapped structures',
        'output':       f'{main_path}/swap_to_original_match_ternary.pkl',
    },
}

MATCHED_ORIG_PATH = config[DATASET]['matched_orig']
MATCHED_SWAP_PATH = config[DATASET]['matched_swap']
OUTPUT_PATH       = config[DATASET]['output']
# ─────────────────────────────────────────────────────────────────────────────

matched_orig = get_pickle_file(MATCHED_ORIG_PATH)
matched_swap = get_pickle_file(MATCHED_SWAP_PATH)

print(f"Original df:  {len(matched_orig)} rows")
print(f"Swapped df:   {len(matched_swap)} rows")

# ── Re-run atom_swap selection to get exact original rows ────────────────────
row_55 = pick_pathway(matched_orig, 'Co Rich')
row_35 = pick_pathway(matched_orig, 'Co Mid')

print(f"\nSelected original pathways:")
print(f"  Co_55 ranking_file: {row_55['ranking_file']}, y={row_55['y']:.4f}")
print(f"  Co_35 ranking_file: {row_35['ranking_file']}, y={row_35['y']:.4f}")

# ── Compute decomp fingerprints for selected original pathways ───────────────────────
fp_55 = decomp_fingerprint(row_55['pathway'])
fp_35 = decomp_fingerprint(row_35['pathway'])
print(f"\n  Co_55 decomp fp: {fp_55}")
print(f"  Co_35 decomp fp: {fp_35}")

# ── Build decomp fingerprint for all swapped rows ────────────────────────────
print("\nComputing decomp fingerprints for swapped df...")
matched_swap['decomp_fp'] = matched_swap['pathway'].apply(decomp_fingerprint)

# ── Match each swap type to its original ─────────────────────────────────────
# Co_55 swaps
co55_file_names = ['co55_Fe_high_to_Co', 'co55_Co_high_to_Fe']
# Co_35 swaps
co35_file_names = ['co35_Mo_high_to_Co', 'co35_Co_high_to_Mo']

pairs = []

for file_name in co55_file_names + co35_file_names:
    is_co55 = file_name.startswith('co55')
    orig_row = row_55 if is_co55 else row_35
    target_fp = fp_55 if is_co55 else fp_35

    # Find matching swapped row: same file_name AND same decomp fingerprint
    swap_candidates = matched_swap[
        (matched_swap['file_name'] == file_name) &
        (matched_swap['decomp_fp'] == target_fp)
    ]

    if len(swap_candidates) == 0:
        print(f"  WARNING: no match found for file_name={file_name}")
        continue
    if len(swap_candidates) > 1:
        print(f"  WARNING: {len(swap_candidates)} matches for file_name={file_name}, taking first")

    swap_row = swap_candidates.iloc[0]
    print(f"\n  Matched: {file_name}")
    print(f"    orig ranking: {orig_row['ranking_file']}, y={orig_row['y']:.4f}")
    print(f"    swap ranking: {swap_row['ranking_file']}, y={swap_row['y']:.4f}")

    pairs.append({
        'file_name':           file_name,
        'co_concentration':    orig_row['co_concentration'],
        'decomp_fp':           target_fp,
        # Original
        'ranking_file_orig':   orig_row['ranking_file'],
        'y_orig':              orig_row['y'],
        'ng_df_hea_orig':      orig_row['ng_df_hea'],
        'na_hea_orig':         orig_row['na_hea'],
        'ea_hea_orig':         orig_row['ea_hea'],
        'pathway_orig':        orig_row['pathway'],
        # Swapped
        'ranking_file_swap':   swap_row['ranking_file'],
        'y_swap':              swap_row['y'],
        'ng_df_hea_swap':      swap_row['ng_df_hea'],
        'na_hea_swap':         swap_row['na_hea'],
        'ea_hea_swap':         swap_row['ea_hea'],
        'pathway_swap':        swap_row['pathway'],
    })

paired_df = pd.DataFrame(pairs)
print(f"\nTotal pairs: {len(paired_df)} / 4")
print(paired_df[['file_name', 'co_concentration', 'y_orig', 'y_swap',
                  'ranking_file_orig', 'ranking_file_swap']])

save_pickle_file(OUTPUT_PATH, paired_df)
print(f"\nSaved -> {OUTPUT_PATH}")
