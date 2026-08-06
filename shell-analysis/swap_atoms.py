"""
swap_atoms.py
------------
For Co_55 (Co Rich) and Co_35 (Co Mid) pathways:

Co_55:
  Swap 1: high-scoring Fe -> replace with Co (Fe -> Co swap)
  Swap 2: high-scoring Co -> replace with Fe (Co -> Fe swap)

Co_35:
  Swap 1: high-scoring Mo -> replace with Co (Mo -> Co swap)
  Swap 2: high-scoring Co -> replace with Mo (Co -> Mo swap)

Records everything in a summary dataframe.
"""

import pickle
import numpy as np
import pandas as pd
from ase.io import write
from pathlib import Path
from collections import Counter
import os

def get_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle_file(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_hea_atoms(row):
    """Extract HEA ASE Atoms object from pathway."""
    return row['pathway'][0][-1].copy()

def pick_pathway(matched_df, concentration):
    """Pick one representative pathway for a given concentration."""
    subset = matched_df[matched_df['co_concentration'] == concentration]
    if len(subset) == 0:
        raise ValueError(f"No pathways found for concentration: {concentration}")
    # Pick the first one (deterministic)
    return subset.iloc[0]

def get_top_scoring_atom(ng_df, element):
    """Get ASE index of the highest scoring atom of a given element."""
    sub = ng_df[ng_df['atom'] == element]
    if len(sub) == 0:
        raise ValueError(f"No atoms of element {element} found in ng_df")
    return sub.loc[sub['ng_value'].idxmax(), 'ase_index']

def get_low_scoring_atom(ng_df, element):
    """Get ASE index of the lowest scoring atom of a given element."""
    sub = ng_df[ng_df['atom'] == element]
    if len(sub) == 0:
        raise ValueError(f"No atoms of element {element} found in ng_df")
    return sub.loc[sub['ng_value'].idxmin(), 'ase_index']

def swap_atoms(atoms, idx_a, idx_b):
    """
    Swap the chemical species of two atoms at indices idx_a and idx_b.
    Returns a new ASE Atoms object with the swap applied.
    """
    new_atoms = atoms.copy()
    sym = list(new_atoms.get_chemical_symbols())
    sym[idx_a], sym[idx_b] = sym[idx_b], sym[idx_a]
    new_atoms.set_chemical_symbols(sym)
    return new_atoms

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MAIN_PATH   = str(Path(__file__).parent)
MATCHED_DF  = 'path to matched dataframe'
OUTPUT_DIR  = f'{MAIN_PATH}/swapped_structures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

matched_df = get_pickle_file(MATCHED_DF)
print(f"Loaded matched_df: {len(matched_df)} rows")
print(f"Concentrations: {matched_df['co_concentration'].value_counts().to_dict()}")

records = []

# ════════════════════════════════════════════════════════════════════════════
# Co_55 (Co Rich): Fe <-> Co swaps
# ════════════════════════════════════════════════════════════════════════════
print("\n── Co_55 (Co Rich) ──────────────────────────────────────────────────────")
row_55 = pick_pathway(matched_df, 'Co Rich')
hea_55 = get_hea_atoms(row_55)
ng_df_55 = row_55['ng_df_hea']

print(f"  Pathway ranking file: {row_55['ranking_file']}")
print(f"  HEA composition: {dict(Counter(hea_55.get_chemical_symbols()))}")
print(f"  ng_df_hea shape: {ng_df_55.shape}")

# Swap 1: high-scoring Fe -> Co
high_fe_idx = get_top_scoring_atom(ng_df_55, 'Fe')
low_co_idx  = get_low_scoring_atom(ng_df_55, 'Co')
high_fe_score = ng_df_55.loc[ng_df_55['ase_index'] == high_fe_idx, 'ng_value'].item()
low_co_score  = ng_df_55.loc[ng_df_55['ase_index'] == low_co_idx,  'ng_value'].item()

swapped_55_FeCo = swap_atoms(hea_55, high_fe_idx, low_co_idx)
out_path = os.path.join(OUTPUT_DIR, 'co55_Fe_high_to_Co.vasp')
write(out_path, swapped_55_FeCo, format='vasp')
print(f"\n  Swap 1 (Fe→Co): high Fe[{high_fe_idx}] (score={high_fe_score:.6f}) "
      f"↔ low Co[{low_co_idx}] (score={low_co_score:.6f})")
print(f"  Saved: {out_path}")
records.append({
    'concentration':   'Co Rich (Co_55)',
    'swap_type':       'Fe_high → Co',
    'atom_a_element':  'Fe',
    'atom_a_index':    high_fe_idx,
    'atom_a_score':    high_fe_score,
    'atom_b_element':  'Co',
    'atom_b_index':    low_co_idx,
    'atom_b_score':    low_co_score,
    'output_file':     out_path,
    'ranking_file':    row_55['ranking_file'],
    'co_count':        row_55['co_count'],
    'y':               row_55['y'],
})

# Swap 2: high-scoring Co -> Fe
high_co_idx = get_top_scoring_atom(ng_df_55, 'Co')
low_fe_idx  = get_low_scoring_atom(ng_df_55, 'Fe')
high_co_score = ng_df_55.loc[ng_df_55['ase_index'] == high_co_idx, 'ng_value'].item()
low_fe_score  = ng_df_55.loc[ng_df_55['ase_index'] == low_fe_idx,  'ng_value'].item()

swapped_55_CoFe = swap_atoms(hea_55, high_co_idx, low_fe_idx)
out_path = os.path.join(OUTPUT_DIR, 'co55_Co_high_to_Fe.vasp')
write(out_path, swapped_55_CoFe, format='vasp')
print(f"\n  Swap 2 (Co→Fe): high Co[{high_co_idx}] (score={high_co_score:.6f}) "
      f"↔ low Fe[{low_fe_idx}] (score={low_fe_score:.6f})")
print(f"  Saved: {out_path}")
records.append({
    'concentration':   'Co Rich (Co_55)',
    'swap_type':       'Co_high → Fe',
    'atom_a_element':  'Co',
    'atom_a_index':    high_co_idx,
    'atom_a_score':    high_co_score,
    'atom_b_element':  'Fe',
    'atom_b_index':    low_fe_idx,
    'atom_b_score':    low_fe_score,
    'output_file':     out_path,
    'ranking_file':    row_55['ranking_file'],
    'co_count':        row_55['co_count'],
    'y':               row_55['y'],
})

# Also save original Co_55 HEA for reference
out_path_orig = os.path.join(OUTPUT_DIR, 'co55_original.vasp')
write(out_path_orig, hea_55, format='vasp')
print(f"\n  Original saved: {out_path_orig}")

# ════════════════════════════════════════════════════════════════════════════
# Co_35 (Co Mid): Mo <-> Co swaps
# ════════════════════════════════════════════════════════════════════════════
print("\n── Co_35 (Co Mid) ───────────────────────────────────────────────────────")
row_35 = pick_pathway(matched_df, 'Co Mid')
hea_35 = get_hea_atoms(row_35)
ng_df_35 = row_35['ng_df_hea']

print(f"  Pathway ranking file: {row_35['ranking_file']}")
print(f"  HEA composition: {dict(Counter(hea_35.get_chemical_symbols()))}")
print(f"  ng_df_hea shape: {ng_df_35.shape}")

# Swap 1: high-scoring Mo -> Co
high_mo_idx = get_top_scoring_atom(ng_df_35, 'Mo')
low_co_idx  = get_low_scoring_atom(ng_df_35, 'Co')
high_mo_score = ng_df_35.loc[ng_df_35['ase_index'] == high_mo_idx, 'ng_value'].item()
low_co_score  = ng_df_35.loc[ng_df_35['ase_index'] == low_co_idx,  'ng_value'].item()

swapped_35_MoCo = swap_atoms(hea_35, high_mo_idx, low_co_idx)
out_path = os.path.join(OUTPUT_DIR, 'co35_Mo_high_to_Co.vasp')
write(out_path, swapped_35_MoCo, format='vasp')
print(f"\n  Swap 1 (Mo→Co): high Mo[{high_mo_idx}] (score={high_mo_score:.6f}) "
      f"↔ low Co[{low_co_idx}] (score={low_co_score:.6f})")
print(f"  Saved: {out_path}")
records.append({
    'concentration':   'Co Mid (Co_35)',
    'swap_type':       'Mo_high → Co',
    'atom_a_element':  'Mo',
    'atom_a_index':    high_mo_idx,
    'atom_a_score':    high_mo_score,
    'atom_b_element':  'Co',
    'atom_b_index':    low_co_idx,
    'atom_b_score':    low_co_score,
    'output_file':     out_path,
    'ranking_file':    row_35['ranking_file'],
    'co_count':        row_35['co_count'],
    'y':               row_35['y'],
})

# Swap 2: high-scoring Co -> Mo
high_co_idx = get_top_scoring_atom(ng_df_35, 'Co')
low_mo_idx  = get_low_scoring_atom(ng_df_35, 'Mo')
high_co_score = ng_df_35.loc[ng_df_35['ase_index'] == high_co_idx, 'ng_value'].item()
low_mo_score  = ng_df_35.loc[ng_df_35['ase_index'] == low_mo_idx,  'ng_value'].item()

swapped_35_CoMo = swap_atoms(hea_35, high_co_idx, low_mo_idx)
out_path = os.path.join(OUTPUT_DIR, 'co35_Co_high_to_Mo.vasp')
write(out_path, swapped_35_CoMo, format='vasp')
print(f"\n  Swap 2 (Co→Mo): high Co[{high_co_idx}] (score={high_co_score:.6f}) "
      f"↔ low Mo[{low_mo_idx}] (score={low_mo_score:.6f})")
print(f"  Saved: {out_path}")
records.append({
    'concentration':   'Co Mid (Co_35)',
    'swap_type':       'Co_high → Mo',
    'atom_a_element':  'Co',
    'atom_a_index':    high_co_idx,
    'atom_a_score':    high_co_score,
    'atom_b_element':  'Mo',
    'atom_b_index':    low_mo_idx,
    'atom_b_score':    low_mo_score,
    'output_file':     out_path,
    'ranking_file':    row_35['ranking_file'],
    'co_count':        row_35['co_count'],
    'y':               row_35['y'],
})

# Also save original Co_35 HEA for reference
out_path_orig = os.path.join(OUTPUT_DIR, 'co35_original.vasp')
write(out_path_orig, hea_35, format='vasp')
print(f"\n  Original saved: {out_path_orig}")

# ════════════════════════════════════════════════════════════════════════════
# Save records =)
# ════════════════════════════════════════════════════════════════════════════
records_df = pd.DataFrame(records)
records_path = os.path.join(OUTPUT_DIR, 'swap_records.csv')
records_df.to_csv(records_path, index=False, sep=';')
save_pickle_file(os.path.join(OUTPUT_DIR, 'swap_records.pkl'), records_df)

print("\n══ Summary ══════════════════════════════════════════════════════════════")
print(records_df[['concentration', 'swap_type', 'atom_a_element', 'atom_a_index',
                   'atom_a_score', 'atom_b_element', 'atom_b_index', 'atom_b_score']])
print(f"\nRecords saved -> {records_path}")
