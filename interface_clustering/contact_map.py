import os
from typing import List, Tuple
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser


def extract_ca_coords(cif_file: str, chain_ids: List[str]) -> np.ndarray:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('model', cif_file)
    ca_coords = []
    for model in structure:
        for chain in model:
            if chain.id in chain_ids:
                for res in chain:
                    if 'CA' in res:
                        ca_coords.append(res['CA'].get_coord())
    return np.array(ca_coords)


def compute_contact_map(coords_ab: np.ndarray, coords_ag: np.ndarray, cutoff: float = 5.0) -> np.ndarray:
    if len(coords_ab) == 0 or len(coords_ag) == 0:
        return np.zeros((len(coords_ab), len(coords_ag)), dtype=int)
    dists = np.linalg.norm(coords_ab[:, None, :] - coords_ag[None, :, :], axis=-1)
    return (dists < cutoff).astype(int)


def load_contact_maps(cif_dir: str, antibody_chain: str, antigen_chains: List[str], cutoff: float) -> Tuple[List[str], List[np.ndarray]]:
    cif_files = [os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith('.cif')]
    file_names: List[str] = []
    contact_maps: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    for path in cif_files:
        ab = extract_ca_coords(path, [antibody_chain])
        ag = extract_ca_coords(path, antigen_chains)
        cmap = compute_contact_map(ab, ag, cutoff)
        file_names.append(os.path.basename(path))
        contact_maps.append(cmap)
        shapes.append(cmap.shape)
    return file_names, contact_maps
