'''
build_comprehensive_df.py
===================================================================================================================
Builds a comprehensive pandas dataframe for each graph in the test dataset with:
        'gid':              unique hexadecimal id,
        'fingerprint':      unique fingerprint,
        'ranking_file':     path to ranking file,
        'hea_bonds':        number of bonds belonging to the parent MPEA,
        'hea_angles':       number of angles belonging to the parent MPEA,,
        'co_count':         number of Co atoms,
        'co_concentration': Co concentration, either 15,35, or 55,
        'ng_df_hea':        stoichiometric features from parent MPEA,
        'na_hea':           2-Body features from parent MPEA,
        'ea_hea':           3-Body features from parent MPEA,,
        'y':                y value,
        'y_match':          y value if found,
        'pathway':          Information about the decomposition pathway,

'''

import glob, torch, pickle, os, hashlib, gzip
import numpy as np
import pandas as pd
from pathlib import Path, PurePath
from periodictable import elements

def save_pickle_file(file_name, save_object):
    with open(file_name, "wb") as file:
        pickle.dump(save_object, file)

def get_pickle_file(file_name):
    with open(file_name, "rb") as file:
        saved_object = pickle.load(file)
    return saved_object

def load_ranking(path):
    _orig = torch.storage._load_from_bytes
    def patched(b):
        import io
        return torch.load(io.BytesIO(b), map_location=torch.device('cpu'), weights_only=False)
    torch.storage._load_from_bytes = patched
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

def graph_fingerprint(g, precision=4):
    atm_bounds = [0] + g.atm_amounts.tolist()
    bnd_bounds = [0] + g.bnd_amounts.tolist()
    ang_bounds = [0] + g.ang_amounts.tolist()
    h = hashlib.md5()
    for i in range(len(atm_bounds) - 1):
        species = np.sort(g.x_atm[atm_bounds[i]:atm_bounds[i+1]].argmax(dim=1).numpy())
        bonds   = np.sort(np.round(g.x_bnd[bnd_bounds[i]:bnd_bounds[i+1]].numpy(), precision))
        angles  = np.sort(np.round(g.x_ang[ang_bounds[i]:ang_bounds[i+1]].numpy(), precision))
        h.update(species.tobytes())
        h.update(bonds.tobytes())
        h.update(angles.tobytes())
    return h.hexdigest()

def extract_hea_features(d, element_list):
    """Extract HEA-only ng, na, ea features. Scores are already normalized
    by the model prediction (pred[i]/preds) at save time in predict_interpretable."""
    hea_atoms  = d['ng_amounts'][0].item()
    hea_bonds  = d['na_amounts'][0].item()
    hea_angles = d['ea_amounts'][0].item()

    # ng: atom-level scores for HEA atoms only
    # ng_data is ordered by GNN processing (grouped by element), NOT by ASE atom index.
    # i_ng_data[n] contains the ASE atom indices for element n.
    # The position in ng_data corresponds to the flattened iteration order of i_ng_data.
    atom_species_hea = []
    ng_values_hea    = []
    ase_indices_hea  = []
    co_count         = 0
    ng_pos           = 0
    for n, ng in enumerate(d['i_ng_data']):
        if len(ng) != 0:
            for a in ng:
                score = d['ng_data'][ng_pos]
                ng_pos += 1
                if a < hea_atoms:
                    atom_species_hea.append(element_list[n])
                    ng_values_hea.append(score)
                    ase_indices_hea.append(a)          # ASE atom index
                    if element_list[n] == 'Co':
                        co_count += 1
    ng_df_hea = pd.DataFrame({'atom': atom_species_hea,
                               'ng_value': ng_values_hea,
                               'ase_index': ase_indices_hea})

    # na: two-body scores for HEA bonds only (first hea_bonds entries)
    na_hea = {}
    for a, na in enumerate(d['na_data']):
        if a >= hea_bonds:
            break
        key = tuple(d['na'][d['i_na_data'][a]])
        na_hea.setdefault(key, []).append(na)

    # ea: three-body scores for HEA angles only (first hea_angles entries)
    ea_hea = {}
    for e, ea in enumerate(d['ea_data']):
        if e >= hea_angles:
            break
        key = tuple(d['ea'][d['i_ea_data'][e]])
        ea_hea.setdefault(key, []).append(ea)

    return ng_df_hea, na_hea, ea_hea, co_count, hea_bonds, hea_angles

element_list = [str(e) for e in elements]
main_path    = str(Path(__file__).parent)

# ── CONFIGURATION — change this to switch between datasets ───────────────────
DATASET = 'ternary'   # options: 'binary' or 'ternary'

config = {
    'binary': {
        'rankings_path':      'path to ranking results',
        'testing_graph_path': 'path to testing graphs',
        'pathway_df_path':    'path to dataframe with graph info and fingerprint',
        'output_df_path':     '',
    },
    'ternary': {
        'rankings_path':      'path to ranking results',
        'testing_graph_path': 'path to testing graphs',
        'pathway_df_path':    'path to dataframe with graph info and fingerprint',
        'output_df_path':     '',
    },
}

cfg                = config[DATASET]
rankings_path      = cfg['rankings_path']
testing_graph_path = cfg['testing_graph_path']
pathway_df_path    = cfg['pathway_df_path']
output_df_path     = cfg['output_df_path']
# ─────────────────────────────────────────────────────────────────────────────

# Load pathway df (must have 'fingerprint' column)
pathway_df = get_pickle_file(pathway_df_path)
assert 'fingerprint' in pathway_df.columns, \
    f"{pathway_df_path} must have a 'fingerprint' column — rerun create_graphs_with_fingerprint.py first"
fp_to_row = {row['fingerprint']: idx for idx, row in pathway_df.iterrows()}
print(f"Pathway df rows: {len(pathway_df)}, unique fingerprints: {len(fp_to_row)}")

# Sort ranking files by numeric index (order they were computed)
ranking_files = sorted(
    glob.glob(os.path.join(rankings_path, '*.data')),
    key=os.path.getctime
)
print(f"Ranking files found: {len(ranking_files)}")

# Testing graphs in glob order (same order rankings were computed)
testing_graph_files = glob.glob(os.path.join(testing_graph_path, '*.pt'))
print(f"Testing graphs found: {len(testing_graph_files)}")

assert len(ranking_files) == len(testing_graph_files), \
    f"Mismatch: {len(ranking_files)} ranking files vs {len(testing_graph_files)} graphs"

matched, mismatches = 0, 0
records = []

for graph_path, rank_path in zip(testing_graph_files, ranking_files):
    g = torch.load(graph_path, map_location='cpu', weights_only=False)
    d = load_ranking(rank_path)

    # Validate positional match via structural amounts
    bnd_ok = g.bnd_amounts.tolist() == d['na_amounts'].tolist()
    ang_ok = g.ang_amounts.tolist() == d['ea_amounts'].tolist()
    if not bnd_ok or not ang_ok:
        mismatches += 1
        print(f"  MISMATCH at {PurePath(rank_path).parts[-1]}: "
              f"bnd_ok={bnd_ok}, ang_ok={ang_ok}")
        continue

    # Extract HEA features from ranking file
    ng_df_hea, na_hea, ea_hea, co_count, hea_bonds, hea_angles = \
        extract_hea_features(d, element_list)

    # Match to pathway df via fingerprint
    fp = graph_fingerprint(g)
    if fp not in fp_to_row:
        print(f"  No pathway match for gid={g.gid[:16]}...")
        continue
    row_idx = fp_to_row[fp]

    # Sanity check y values
    y_graph = g.y.item()
    y_df    = pathway_df.loc[row_idx, 'y']
    y_df    = y_df.item() if hasattr(y_df, 'item') else float(y_df)
    y_ok    = abs(y_graph - y_df) < 1e-4
    if not y_ok:
        print(f"  y MISMATCH: gid={g.gid[:16]}... graph={y_graph:.6f} df={y_df:.6f}")

    # Carry through file_name if it exists in pathway_df
    file_name = pathway_df.loc[row_idx, 'file_name'] \
                if 'file_name' in pathway_df.columns else None

    # Determine Co concentration
    if co_count == 304:
        co_concentration = 'Co Mid'
    elif co_count > 304:
        co_concentration = 'Co Rich'
    else:
        co_concentration = 'Co Poor'

    records.append({
        'gid':              g.gid,
        'fingerprint':      fp,
        'ranking_file':     PurePath(rank_path).parts[-1],
        'hea_bonds':        hea_bonds,
        'hea_angles':       hea_angles,
        'co_count':         co_count,
        'co_concentration': co_concentration,
        'ng_df_hea':        ng_df_hea,
        'na_hea':           na_hea,
        'ea_hea':           ea_hea,
        'y':                y_graph,
        'y_match':          y_ok,
        'pathway':          pathway_df.loc[row_idx, 'pathway'],
        'file_name':        file_name,
    })
    matched += 1

print(f"\nMatched: {matched}, Position mismatches: {mismatches}")

matched_df = pd.DataFrame(records)
save_pickle_file(output_df_path, matched_df)
print(f"Saved {output_df_path}")
print(matched_df[['gid', 'ranking_file', 'co_concentration', 'hea_bonds', 'hea_angles', 'y', 'y_match']].head())
