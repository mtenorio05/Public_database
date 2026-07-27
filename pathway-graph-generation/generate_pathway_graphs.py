"""
generate_pathway_graphs.py
============
Builds binary and ternary decomposition pathway graphs for MPEAs

For each MPEA structure, this script searches over all combinations of
decomposition products (binary and ternary reference phases) that satisfy
stoichiometric conservation, computes the mixing enthalpy for each valid
pathway, and constructs the corresponding ALIGNN-D graph.

Outputs:
    binary_pathways_summary.pkl   - all valid binary decomposition pathways (HEA + 2 products)
    ternary_pathways_summary.pkl  - all valid ternary decomposition pathways (HEA + 3 products)

Each dataframe has columns:
    'pathway' - list of ASE snapshot objects for each structure in the pathway
    'y'       - mixing enthalpy (shifted by y_scale for positivity)
"""

from pathlib import Path, PurePath
import shutil
import glob
import os
import pickle
import numpy as np
import pandas as pd
import torch
from ase.io import read

from catalyst.src.properties.chemical_properties import (
    get_structure_stoichiometry, check_stoichiometry,
    check_num_elements, calc_reaction_enthalpy,
)
from catalyst.src.graph.alignnd import realignnd


def save_pickle_file(file_name, save_object):
    """Save any picklable object to disk."""
    with open(file_name, "wb") as file:
        pickle.dump(save_object, file)


# ══════════════════════════════════════════════════════════════════════════
#  1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Path to DFT decomposition reference structures (binary/ternary/elemental)
STRUCTURES_PATH = 'path_to_dft_relaxed_reference_structures'

# Output directory for graph data
SAVING_DIR = 'path_to_directory_to_store_graph_data'


# Structure dataset labels
dataset_A = 'mpea'          # MPEA (target structures)
dataset_B = 'quaternary'   # quaternary decomposition reference
dataset_C = 'ternary'      # ternary decomposition reference
dataset_D = 'binary'       # binary decomposition reference
dataset_E = 'elemental'    # pure elemental reference

# Decomposition pathway output subdirectories
pathway_types = ['binary', 'ternary']
pathway_dirs = []
for pt in pathway_types:
    pathway_dirs.append(os.path.join(SAVING_DIR, pt))
    if os.path.isdir(pathway_dirs[-1]):
        shutil.rmtree(pathway_dirs[-1])
    os.mkdir(pathway_dirs[-1])

# Each entry in `pathways` defines a decomposition scheme:
#   MPEA -> [reference_1, reference_2, ...]
# `pathway_indexes` maps each reference to its index in `snapshots` below
#   (0=mpea, 1=quaternary, 2=ternary, 3=binary, 4=elemental)
pathways = [
    [dataset_A, dataset_B, dataset_E],                 # MPEA -> quaternary + elemental
    [dataset_A, dataset_C, dataset_D],                 # MPEA -> ternary + binary
    [dataset_A, dataset_D, dataset_D, dataset_E],       # MPEA -> binary + binary + elemental
    [dataset_A, dataset_C, dataset_E, dataset_E],       # MPEA -> ternary + elemental + elemental
]
pathway_indexes = [
    [0, 1, 4],
    [0, 2, 3],
    [0, 3, 3, 4],
    [0, 2, 4, 4],
]

y_scale = 10.0  # shift applied to mixing enthalpy to keep target values positive


# ══════════════════════════════════════════════════════════════════════════
# 2  LOAD REFERENCE STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

print("═" * 70)
print(" CONFIGURATION")
print("═" * 70)
print(f"  Enthalpy shift (y_scale): {y_scale}  "f"(targets stored as H_mix + {y_scale})")
print("═" * 70)
print(" STEP 1 — Loading reference structures")
print("═" * 70)

A_files = glob.glob(os.path.join(STRUCTURES_PATH, dataset_A, '*'))
B_files = glob.glob(os.path.join(STRUCTURES_PATH, dataset_B, '*'))
C_files = glob.glob(os.path.join(STRUCTURES_PATH, dataset_C, '*'))
D_files = glob.glob(os.path.join(STRUCTURES_PATH, dataset_D, '*'))
E_files = glob.glob(os.path.join(STRUCTURES_PATH, dataset_E, '*'))


print(f"\n  MPEA structures found:        {len(A_files):>4}")
print(f"  Quaternary references found: {len(B_files):>4}")
print(f"  Ternary references found:    {len(C_files):>4}")
print(f"  Binary references found:     {len(D_files):>4}")
print(f"  Elemental references found:  {len(E_files):>4}")

# Read the final relaxed snapshot (OUTCAR) for every structure in each dataset
snapshots = [[], [], [], [], []]
dataset_labels = [dataset_A, dataset_B, dataset_C, dataset_D, dataset_E]
file_lists = [A_files, B_files, C_files, D_files, E_files]

print("\n  Reading OUTCAR snapshots...")
for idx, (label, files) in enumerate(zip(dataset_labels, file_lists)):
    for structure in files:
        outcars = glob.glob(os.path.join(structure, 'OUTCAR*'))
        snapshots[idx].append(read(outcars[0], format='vasp-out', index='-1:'))
    print(f"    [{label:<10}] {len(snapshots[idx]):>4} snapshots loaded")


# ══════════════════════════════════════════════════════════════════════════
#  3  ENUMERATE VALID DECOMPOSITION PATHWAYS AND BUILD GRAPHS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print(" STEP 2 — Enumerating decomposition pathways and building graphs")
print("═" * 70)

binary_dict  = {'pathway': [], 'y': []}
ternary_dict = {'pathway': [], 'y': []}

for i, pathway in enumerate(pathways):
    n_terms = len(pathway)
    print(f"\n  Pathway scheme {i + 1}/{len(pathways)}: "
          f"{' + '.join(pathway)}  ({n_terms} terms)")
    n_valid = 0

    if n_terms == 3:
        energies = [0.0, 0.0, 0.0]
        stoichiometries = [[], [], []]

        for s1, structure_1 in enumerate(snapshots[pathway_indexes[i][0]]):
            stoichiometries[0] = get_structure_stoichiometry(structure_1[-1])
            energies[0] = structure_1[-1].get_potential_energy() / len(structure_1[-1])

            for s2, structure_2 in enumerate(snapshots[pathway_indexes[i][1]]):
                stoichiometries[1] = get_structure_stoichiometry(structure_2[-1])
                energies[1] = structure_2[-1].get_potential_energy() / len(structure_2[-1])

                for s3, structure_3 in enumerate(snapshots[pathway_indexes[i][2]]):
                    stoichiometries[2] = get_structure_stoichiometry(structure_3[-1])
                    energies[2] = structure_3[-1].get_potential_energy() / len(structure_3[-1])

                    # Check that the decomposition products conserve all
                    # elements present in the MPEA within the allowed tolerance
                    if check_num_elements(stoichiometries[0], [stoichiometries[1], stoichiometries[2]]):
                        if check_stoichiometry(stoichiometries[0],
                                              [stoichiometries[1], stoichiometries[2]],
                                              delta=0.15):
                            H_mix = calc_reaction_enthalpy(energies=energies, n_systems=len(energies))
                            structures = [structure_1[-1], structure_2[-1], structure_3[-1]]

                            binary_dict['pathway'].append([structure_1, structure_2, structure_3])
                            binary_dict['y'].append(torch.tensor(H_mix + y_scale))

                            # Build the ALIGNN-D graph for this pathway
                            graph_data = realignnd(structures, neighbor_params=[3.5, -1],
                                                   dihedral=False, store_atoms=False,
                                                   use_pt=True, include_angs=True)
                            graph_data.y = torch.tensor(H_mix + y_scale)

                            torch.save(graph_data, os.path.join(pathway_dirs[0], graph_data.gid + '.pt'))
                            n_valid += 1
        print(f"    -> {n_valid} valid pathway(s) found for this scheme")
        scheme_counts.append((' + '.join(pathway), output_type, n_valid))

    elif n_terms == 4:
        energies = [0.0, 0.0, 0.0, 0.0]
        stoichiometries = [[], [], [], []]

        for s1, structure_1 in enumerate(snapshots[pathway_indexes[i][0]]):
            stoichiometries[0] = get_structure_stoichiometry(structure_1[-1])
            energies[0] = structure_1[-1].get_potential_energy() / len(structure_1[-1])

            for s2, structure_2 in enumerate(snapshots[pathway_indexes[i][1]]):
                stoichiometries[1] = get_structure_stoichiometry(structure_2[-1])
                energies[1] = structure_2[-1].get_potential_energy() / len(structure_2[-1])

                for s3, structure_3 in enumerate(snapshots[pathway_indexes[i][2]]):
                    stoichiometries[2] = get_structure_stoichiometry(structure_3[-1])
                    energies[2] = structure_3[-1].get_potential_energy() / len(structure_3[-1])

                    for s4, structure_4 in enumerate(snapshots[pathway_indexes[i][3]]):
                        stoichiometries[3] = get_structure_stoichiometry(structure_4[-1])
                        energies[3] = structure_4[-1].get_potential_energy() / len(structure_4[-1])

                        if check_num_elements(stoichiometries[0],
                                              [stoichiometries[1], stoichiometries[2], stoichiometries[3]]):
                            if check_stoichiometry(stoichiometries[0],
                                                   [stoichiometries[1], stoichiometries[2], stoichiometries[3]],
                                                   delta=0.15):
                                H_mix = calc_reaction_enthalpy(energies=energies, n_systems=len(energies))
                                structures = [structure_1[-1], structure_2[-1],
                                              structure_3[-1], structure_4[-1]]

                                ternary_dict['pathway'].append(
                                    [structure_1, structure_2, structure_3, structure_4])
                                ternary_dict['y'].append(torch.tensor(H_mix + y_scale))

                                graph_data = realignnd(structures, neighbor_params=[3.5, -1],
                                                       dihedral=False, store_atoms=False,
                                                       use_pt=True, include_angs=True)
                                graph_data.y = torch.tensor(H_mix + y_scale)
                                torch.save(graph_data, os.path.join(pathway_dirs[1], graph_data.gid + '.pt'))
                                n_valid += 1

        print(f"    -> {n_valid} valid pathway(s) found for this scheme")
        scheme_counts.append((' + '.join(pathway), output_type, n_valid))

# ══════════════════════════════════════════════════════════════════════════
#  4  SUMMARY OF PATHWAYS FOUND
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print(" SUMMARY — pathways found per scheme")
print("═" * 70)
print(f"  {'Scheme':<40} {'Type':<10} {'Count':>7}")
print(f"  {'-' * 40} {'-' * 10} {'-' * 7}")
for description, output_type, n_valid in scheme_counts:
    print(f"  {description:<40} {output_type:<10} {n_valid:>7}")

# ══════════════════════════════════════════════════════════════════════════
#  5  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print(" STEP 3 — Saving results")
print("═" * 70)

binary_df  = pd.DataFrame(binary_dict)
ternary_df = pd.DataFrame(ternary_dict)

save_pickle_file(os.path.join(str(Path(__file__).parent), 'binary_pathways_summary.pkl'), binary_df)
save_pickle_file(os.path.join(str(Path(__file__).parent), 'ternary_pathways_summary.pkl'), ternary_df)

print(f"  binary_pathways_summary.pkl  saved — {len(binary_df):>5} pathways")
print(f"  ternary_pathways_summary.pkl saved — {len(ternary_df):>5} pathways")
print("\nDone.")
