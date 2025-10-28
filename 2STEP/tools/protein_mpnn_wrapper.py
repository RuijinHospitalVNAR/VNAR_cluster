"""
ProteinMPNN wrapper for sequence generation.

This module provides functionality to generate mutant protein sequences
using ProteinMPNN based on epitope regions and immunogenicity mode.
"""

import os
import logging
import subprocess
import tempfile
from typing import List, Dict, Optional
import pandas as pd


class ProteinMPNNWrapper:
    """Wrapper class for ProteinMPNN sequence generation."""
    
    def __init__(self, proteinmpnn_path: Optional[str] = None):
        """
        Initialize the ProteinMPNN wrapper.
        
        Args:
            proteinmpnn_path: Path to ProteinMPNN executable
        """
        self.proteinmpnn_path = proteinmpnn_path or "ProteinMPNN"
        self.logger = logging.getLogger(__name__)
    
    def prepare_input_files(self, pdb_path: str, epitope_df: pd.DataFrame, 
                          mode: str, output_dir: str) -> Dict[str, str]:
        """
        Prepare input files for ProteinMPNN.
        
        Args:
            pdb_path: Path to input PDB file
            epitope_df: DataFrame containing epitope information
            mode: Immunogenicity mode ('reduce' or 'enhance')
            output_dir: Output directory for temporary files
            
        Returns:
            Dictionary with file paths for ProteinMPNN input
        """
        # Create temporary directory for ProteinMPNN files
        temp_dir = os.path.join(output_dir, "proteinmpnn_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy PDB file
        pdb_file = os.path.join(temp_dir, "input.pdb")
        with open(pdb_path, 'r') as src, open(pdb_file, 'w') as dst:
            dst.write(src.read())
        
        # Create position list file based on epitopes
        pos_list_file = os.path.join(temp_dir, "pos_list.txt")
        with open(pos_list_file, 'w') as f:
            # Extract unique positions from epitopes
            positions = set()
            for _, row in epitope_df.iterrows():
                start_pos = row['start_position']
                end_pos = row['end_position']
                positions.update(range(start_pos, end_pos + 1))
            
            # Write positions (1-indexed)
            for pos in sorted(positions):
                f.write(f"{pos}\n")
        
        # Create fixed positions file (empty for now)
        fixed_pos_file = os.path.join(temp_dir, "fixed_positions.txt")
        with open(fixed_pos_file, 'w') as f:
            f.write("")  # Empty file for now
        
        return {
            'pdb_file': pdb_file,
            'pos_list_file': pos_list_file,
            'fixed_pos_file': fixed_pos_file,
            'temp_dir': temp_dir
        }
    
    def run_proteinmpnn(self, input_files: Dict[str, str], samples_per_temp: int,
                       temperatures: List[float]) -> List[str]:
        """
        Run ProteinMPNN to generate sequences.
        
        Args:
            input_files: Dictionary with input file paths
            samples_per_temp: Number of samples per temperature
            temperatures: List of temperature values
            
        Returns:
            List of generated sequences
        """
        self.logger.info(f"Running ProteinMPNN with {len(temperatures)} temperatures")
        
        all_sequences = []
        
        for temp in temperatures:
            self.logger.info(f"Generating sequences at temperature {temp}")
            
            # Prepare command
            cmd = [
                self.proteinmpnn_path,
                "--pdb_path", input_files['pdb_file'],
                "--pdb_path_chains", "A",  # Assuming chain A
                "--out_path", os.path.join(input_files['temp_dir'], f"output_T{temp}"),
                "--num_seq_per_target", str(samples_per_temp),
                "--sampling_temp", str(temp),
                "--seed", "42"
            ]
            
            try:
                # Run ProteinMPNN
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.logger.error(f"ProteinMPNN failed: {result.stderr}")
                    continue
                
                # Read generated sequences
                output_file = os.path.join(input_files['temp_dir'], f"output_T{temp}.fa")
                if os.path.exists(output_file):
                    sequences = self._read_fasta_sequences(output_file)
                    all_sequences.extend(sequences)
                    self.logger.info(f"Generated {len(sequences)} sequences at T={temp}")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"ProteinMPNN timed out at temperature {temp}")
            except Exception as e:
                self.logger.error(f"Error running ProteinMPNN at temperature {temp}: {e}")
        
        self.logger.info(f"Total sequences generated: {len(all_sequences)}")
        return all_sequences
    
    def _read_fasta_sequences(self, fasta_file: str) -> List[str]:
        """
        Read sequences from FASTA file.
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            List of sequences
        """
        sequences = []
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)
        
        if current_seq:
            sequences.append(''.join(current_seq))
        
        return sequences
    
    def cleanup_temp_files(self, input_files: Dict[str, str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            input_files: Dictionary with file paths to clean up
        """
        try:
            import shutil
            if os.path.exists(input_files['temp_dir']):
                shutil.rmtree(input_files['temp_dir'])
                self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary files: {e}")


def generate_mutants(pdb_path: str, epitope_df: pd.DataFrame, mode: str,
                    samples_per_temp: int = 20, temps: List[float] = None,
                    proteinmpnn_path: Optional[str] = None) -> List[str]:
    """
    Generate mutant sequences using ProteinMPNN.
    
    Args:
        pdb_path: Path to input PDB file
        epitope_df: DataFrame containing epitope information
        mode: Immunogenicity mode ('reduce' or 'enhance')
        samples_per_temp: Number of samples per temperature
        temps: List of temperature values
        proteinmpnn_path: Path to ProteinMPNN executable
        
    Returns:
        List of generated mutant sequences
    """
    if temps is None:
        temps = [0.1, 0.3, 0.5]
    
    wrapper = ProteinMPNNWrapper(proteinmpnn_path)
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Prepare input files
        input_files = wrapper.prepare_input_files(pdb_path, epitope_df, mode, output_dir)
        
        # Run ProteinMPNN
        sequences = wrapper.run_proteinmpnn(input_files, samples_per_temp, temps)
        
        return sequences
        
    finally:
        # Clean up temporary files
        if 'input_files' in locals():
            wrapper.cleanup_temp_files(input_files)
