import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Protein structure clustering based on chain distances')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing structure files (PDB or mmCIF format)')
    parser.add_argument('--chain_pair', type=str, required=True,
                       help='Chain pair with optional residue ranges. Format: "chainID:range;chainID:range"\n'
                            'Range format options:\n'
                            '- Single range: "A:100-200;B:50-150"\n'
                            '- Multiple ranges: "A:20-100,150-200;B:10-35,200-250"\n'
                            '- All residues: "A:all;B:100-200"\n'
                            'Example: "A:1-50,70-100;B:all" means residues 1-50 and 70-100 from chain A\n'
                            'will be compared with all residues from chain B.\n'
                            'Note: the chain A is refernce structure, chain B is the target structure.')
    parser.add_argument('--distance_threshold', type=float, default=8.0,
                       help='Distance threshold for contact definition (in Angstroms)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--eps', type=float, default=0.3,
                       help='The maximum distance between two samples for one to be considered as in the \n'
                       'neighborhood of the other. This is not a maximum bound on the distances of points \n'
                       'within a cluster. This is the most important DBSCAN parameter to choose appropriately \n'
                       'for your data set and distance function.')
    parser.add_argument('--min_samples', type=int, default=3,
                       help='Minimum number of samples in a neighborhood for a point to be considered as a core point')
    
    return parser.parse_args()

def parse_structure(file_path):
    """
    Parse a PDB or CIF structure file to extract atomic coordinates.

    Parameters:
        file_path (str): Path to the structure file.

    Returns:
        structure (Structure): Parsed structure object.
    """
    if file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif file_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format: %s" % file_path)
    return parser.get_structure(os.path.basename(file_path), file_path)

def calculate_statistics(chain_a_coords, chain_b_coords, threshold):
    """
    Calculate distances and generate a binary array for residues in chain A.

    Parameters:
        chain_a_coords (List[np.ndarray]): Coordinates of chain A residues.
        chain_b_coords (List[np.ndarray]): Coordinates of chain B residues.
        threshold (float): Distance threshold.

    Returns:
        List[int]: Binary array indicating counts of residues meeting the threshold.
    """
    statistics = []
    for coord_a in chain_a_coords:
        count = sum(1 for coord_b in chain_b_coords if euclidean(coord_a, coord_b) <= threshold)
        statistics.append(1 if count > 0 else 0)
    return statistics

def normalized_hamming_distance(arr1, arr2):
    """
    Calculate the normalized Hamming distance between two arrays.
    Elements where both arrays are 0 are ignored in the calculation.
    The result is normalized by the number of elements that are not both 0.

    Parameters:
        arr1 (np.ndarray): First array.
        arr2 (np.ndarray): Second array.

    Returns:
        float: Normalized Hamming distance.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")
    
    # Create a mask where either arr1 or arr2 is not 0
    mask = (arr1 != 0) | (arr2 != 0)
    
    # Apply the mask to both arrays
    filtered_arr1 = arr1[mask]
    filtered_arr2 = arr2[mask]
    
    # Calculate the Hamming distance
    if len(filtered_arr1) == 0:
        return 1.0  # If no elements to compare, return 1.0
    
    hamming_dist = np.sum(filtered_arr1 != filtered_arr2)
    # Normalize by the number of elements that are not both 0
    normalized_dist = hamming_dist / len(filtered_arr1)

    return normalized_dist

def cosine_distance(vec_a, vec_b):
    """
    Calculate the cosine distance between two vectors.

    Parameters:
        vec_a (np.ndarray): The first vector.
        vec_b (np.ndarray): The second vector.

    Returns:
        float: The cosine distance, which ranges from [0, 2].
    """
    # Ensure inputs are NumPy arrays
    vec_a = np.asarray(vec_a)
    vec_b = np.asarray(vec_b)
    # Calculate cosine similarity
    cosine_similarity = cosine(vec_a, vec_b)

    # Calculate cosine distance
    return cosine_similarity


def cluster_statistics(statistics, eps, min_samples):
    """
    Perform clustering on statistics arrays.

    Parameters:
        statistics (List[List[int]]): List of statistics arrays for all structures.
        eps (float): DBSCAN eps parameter.
        min_samples (int): DBSCAN min_samples parameter.

    Returns:
        np.ndarray: Cluster labels.
    """
    clustering = DBSCAN(metric=cosine_distance, eps=eps, min_samples=min_samples)
    return clustering.fit_predict(statistics)


def save_cluster_results(labels, structure_files, output_dir):
    """
    Save the clustered results into separate directories, converting all files to PDB format.

    Parameters:
        labels (np.ndarray): Cluster labels.
        structure_files (List[str]): List of structure file paths.
        output_dir (str): Output directory for clustered results.
    """
    for cluster_id in set(labels):
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        for idx, label in enumerate(labels):
            if label == cluster_id:
                file_path = structure_files[idx]
                structure = parse_structure(file_path)
                io = PDBIO()
                io.set_structure(structure)
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file_path = os.path.join(cluster_dir, f"Cluster_{cluster_id}_{file_name}.pdb")
                io.save(output_file_path)

def parse_chain_and_ranges(chain_pair):
    """Parse chain pair and residue ranges
    
    Input format examples:
    - "A:100-200;B:50-150"
    - "A:20-100,150-200;B:10-35,200-250"
    - "A:all;B:100-200"
    
    Returns:
        tuple: (chain1_id, chain1_ranges, chain2_id, chain2_ranges)
        where ranges is None if 'all' is specified, otherwise a list of (start, end) tuples
    """
    chain_specs = chain_pair.split(';')
    if len(chain_specs) != 2:
        raise ValueError("Chain pair must specify exactly two chains")
    
    chains_ranges = {}
    for spec in chain_specs:
        parts = spec.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid chain specification: {spec}")
        
        chain_id = parts[0]
        range_str = parts[1].lower()  # Convert to lowercase for 'all' comparison
        
        if range_str == 'all':
            chains_ranges[chain_id] = None  # None indicates all residues
        else:
            ranges = []
            for range_part in range_str.split(','):
                try:
                    start, end = map(int, range_part.split('-'))
                    if start > end:
                        raise ValueError(f"Invalid range: {start}-{end}, start must be less than end")
                    ranges.append((start, end))
                except ValueError as e:
                    raise ValueError(f"Invalid range format in {range_part}. Expected format: start-end") from e
            chains_ranges[chain_id] = ranges
    
    chain_ids = list(chains_ranges.keys())
    return chain_ids[0], chains_ranges[chain_ids[0]], chain_ids[1], chains_ranges[chain_ids[1]]

def extract_chain_residues_by_ranges(structure, chain_id, ranges):
    """
    Extract CA atom coordinates for all residues in a specified chain based on given ranges.

    Parameters:
        structure (Structure): Parsed structure object.
        chain_id (str): Chain identifier.
        ranges (list): List of (start, end) tuples or None for all residues.

    Returns:
        List[np.ndarray]: List of CA atom coordinates.
    """
    chain = structure[0][chain_id]
    coords = []
    
    if ranges is None:  # if 'all'
        return [residue["CA"].coord for residue in chain if "CA" in residue]
    
    for start, end in ranges:
        for residue in chain:
            residue_id = residue.id[1]
            # the index of the first residue is 0
            if (start <= residue_id <= end) and ("CA" in residue):
                coords.append(residue["CA"].coord)
    
    return coords

def main():
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse chain pair and ranges
    chain_a, chain_a_ranges, chain_b, chain_b_ranges = parse_chain_and_ranges(args.chain_pair)

    # Process structure files
    structure_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".pdb") or f.endswith(".cif")
    ]

    statistics = []
    for file_path in structure_files:
        structure = parse_structure(file_path)
        
        # extract coordinates based on the parsed ranges
        chain_a_coords = extract_chain_residues_by_ranges(structure, chain_a, chain_a_ranges)
        chain_b_coords = extract_chain_residues_by_ranges(structure, chain_b, chain_b_ranges)

        # Ensure chains are not empty
        if (not chain_a_coords) or (not chain_b_coords):
            raise ValueError(f"Chains {chain_a} or {chain_b} are empty in {file_path}")

        stats = calculate_statistics(chain_a_coords, chain_b_coords, args.distance_threshold)
        statistics.append(stats)

    # Validate data consistency
    array_lengths = [len(arr) for arr in statistics]
    if not all(length == array_lengths[0] for length in array_lengths):
        raise AssertionError("Inconsistent array lengths among structures")

    # Perform clustering
    labels = cluster_statistics(statistics, args.eps, args.min_samples)

    # Sort the labels based on the number of residues in each cluster
    cluster_stats = Counter(labels)
    sorted_cluster_stats = sorted(cluster_stats.items(), key=lambda x: x[1], reverse=True)
    label_map = {}
    for index, (cluster_id, _) in enumerate(sorted_cluster_stats):
        label_map[cluster_id] = index+1
    sorted_labels = [label_map[label] for label in labels]

    # Save results
    save_cluster_results(sorted_labels, structure_files, args.output_dir)

    # Output clustering summary
    print(sorted_labels)
    print(f"Clustering results: {Counter(sorted_labels)}")


if __name__ == "__main__":
    main()

