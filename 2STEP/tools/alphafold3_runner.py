"""
AlphaFold3 runner for structure prediction and scoring.

This module provides functionality to predict protein structures
using AlphaFold3 and apply filtering based on RMSD and DOCKQ thresholds.
"""

import os
import logging
import subprocess
import tempfile
import pandas as pd
from typing import List, Dict, Optional, Tuple
import numpy as np


class AlphaFold3Runner:
    """Class for running AlphaFold3 structure predictions."""
    
    def __init__(self, alphafold3_path: Optional[str] = None):
        """
        Initialize the AlphaFold3 runner.
        
        Args:
            alphafold3_path: Path to AlphaFold3 executable
        """
        self.alphafold3_path = alphafold3_path or "alphafold3"
        self.logger = logging.getLogger(__name__)
    
    def prepare_sequences(self, candidates_df: pd.DataFrame, output_dir: str) -> List[str]:
        """
        Prepare sequence files for AlphaFold3.
        
        Args:
            candidates_df: DataFrame with candidate sequences
            output_dir: Output directory for sequence files
            
        Returns:
            List of sequence file paths
        """
        sequence_files = []
        
        for idx, row in candidates_df.iterrows():
            # Create FASTA file for each sequence
            seq_id = row.get('sequence_id', f'candidate_{idx}')
            peptide = row.get('peptide', '')
            
            if not peptide:
                self.logger.warning(f"No peptide sequence for candidate {idx}")
                continue
            
            # Create FASTA file
            fasta_file = os.path.join(output_dir, f"{seq_id}.fasta")
            with open(fasta_file, 'w') as f:
                f.write(f">{seq_id}\n{peptide}\n")
            
            sequence_files.append(fasta_file)
        
        return sequence_files
    
    def run_alphafold3(self, sequence_files: List[str], output_dir: str) -> Dict[str, Dict]:
        """
        Run AlphaFold3 on sequence files.
        
        Args:
            sequence_files: List of sequence file paths
            output_dir: Output directory for results
            
        Returns:
            Dictionary with prediction results
        """
        self.logger.info(f"Running AlphaFold3 on {len(sequence_files)} sequences")
        
        results = {}
        
        for seq_file in sequence_files:
            seq_id = os.path.basename(seq_file).replace('.fasta', '')
            self.logger.info(f"Predicting structure for {seq_id}")
            
            try:
                # Create output directory for this sequence
                seq_output_dir = os.path.join(output_dir, f"{seq_id}_af3")
                os.makedirs(seq_output_dir, exist_ok=True)
                
                # Prepare command
                cmd = [
                    self.alphafold3_path,
                    "--fasta_path", seq_file,
                    "--output_dir", seq_output_dir,
                    "--model_preset", "monomer",
                    "--max_template_date", "2023-12-31"
                ]
                
                # Run AlphaFold3
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode != 0:
                    self.logger.error(f"AlphaFold3 failed for {seq_id}: {result.stderr}")
                    results[seq_id] = {
                        'pdb_file': None,
                        'confidence_score': 0.0,
                        'rmsd': float('inf'),
                        'dockq': 0.0,
                        'status': 'failed'
                    }
                else:
                    # Parse results
                    pdb_file = os.path.join(seq_output_dir, f"{seq_id}_unrelaxed_rank_001_alphafold3.pdb")
                    confidence_score = self._extract_confidence_score(seq_output_dir)
                    
                    results[seq_id] = {
                        'pdb_file': pdb_file if os.path.exists(pdb_file) else None,
                        'confidence_score': confidence_score,
                        'rmsd': 0.0,  # Would need reference structure for RMSD
                        'dockq': 0.0,  # Would need reference structure for DOCKQ
                        'status': 'success'
                    }
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"AlphaFold3 timed out for {seq_id}")
                results[seq_id] = {
                    'pdb_file': None,
                    'confidence_score': 0.0,
                    'rmsd': float('inf'),
                    'dockq': 0.0,
                    'status': 'timeout'
                }
            except Exception as e:
                self.logger.error(f"Error running AlphaFold3 for {seq_id}: {e}")
                results[seq_id] = {
                    'pdb_file': None,
                    'confidence_score': 0.0,
                    'rmsd': float('inf'),
                    'dockq': 0.0,
                    'status': 'error'
                }
        
        return results
    
    def _extract_confidence_score(self, output_dir: str) -> float:
        """
        Extract confidence score from AlphaFold3 output.
        
        Args:
            output_dir: AlphaFold3 output directory
            
        Returns:
            Confidence score (0-100)
        """
        try:
            # Look for confidence score in JSON file
            json_file = os.path.join(output_dir, "ranking_debug.json")
            if os.path.exists(json_file):
                import json
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    return data.get('confidence', 0.0)
        except Exception as e:
            self.logger.warning(f"Could not extract confidence score: {e}")
        
        return 0.0
    
    def apply_filters(self, results: Dict[str, Dict], rmsd_threshold: float = 2.0,
                     dockq_threshold: float = 0.23) -> Dict[str, Dict]:
        """
        Apply RMSD and DOCKQ filters to results.
        
        Args:
            results: Dictionary with prediction results
            rmsd_threshold: RMSD threshold for filtering
            dockq_threshold: DOCKQ threshold for filtering
            
        Returns:
            Filtered results dictionary
        """
        filtered_results = {}
        
        for seq_id, result in results.items():
            if result['status'] != 'success':
                continue
            
            # Apply filters
            if (result['rmsd'] <= rmsd_threshold and 
                result['dockq'] >= dockq_threshold):
                filtered_results[seq_id] = result
            else:
                self.logger.info(f"Filtered out {seq_id}: RMSD={result['rmsd']:.2f}, "
                               f"DOCKQ={result['dockq']:.3f}")
        
        return filtered_results
    
    def create_results_dataframe(self, candidates_df: pd.DataFrame, 
                               prediction_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create results DataFrame with prediction information.
        
        Args:
            candidates_df: Original candidates DataFrame
            prediction_results: Dictionary with prediction results
            
        Returns:
            DataFrame with combined results
        """
        results_data = []
        
        for idx, row in candidates_df.iterrows():
            seq_id = row.get('sequence_id', f'candidate_{idx}')
            
            if seq_id in prediction_results:
                result = prediction_results[seq_id]
                
                # Combine original data with prediction results
                result_row = row.to_dict()
                result_row.update({
                    'pdb_file': result['pdb_file'],
                    'confidence_score': result['confidence_score'],
                    'rmsd': result['rmsd'],
                    'dockq': result['dockq'],
                    'prediction_status': result['status']
                })
                results_data.append(result_row)
            else:
                # No prediction results for this sequence
                result_row = row.to_dict()
                result_row.update({
                    'pdb_file': None,
                    'confidence_score': 0.0,
                    'rmsd': float('inf'),
                    'dockq': 0.0,
                    'prediction_status': 'no_prediction'
                })
                results_data.append(result_row)
        
        return pd.DataFrame(results_data)


def predict_structure_and_score(candidates_df: pd.DataFrame, 
                               rmsd_threshold: float = 2.0,
                               dockq_threshold: float = 0.23,
                               alphafold3_path: Optional[str] = None) -> pd.DataFrame:
    """
    Predict structures and apply filtering for candidate sequences.
    
    Args:
        candidates_df: DataFrame with candidate sequences
        rmsd_threshold: RMSD threshold for filtering
        dockq_threshold: DOCKQ threshold for filtering
        alphafold3_path: Path to AlphaFold3 executable
        
    Returns:
        DataFrame with structure prediction results
    """
    runner = AlphaFold3Runner(alphafold3_path)
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare sequences
    sequence_files = runner.prepare_sequences(candidates_df, output_dir)
    
    if not sequence_files:
        runner.logger.warning("No sequences to process")
        return pd.DataFrame()
    
    # Run AlphaFold3
    prediction_results = runner.run_alphafold3(sequence_files, output_dir)
    
    # Apply filters
    filtered_results = runner.apply_filters(prediction_results, rmsd_threshold, dockq_threshold)
    
    # Create results DataFrame
    results_df = runner.create_results_dataframe(candidates_df, prediction_results)
    
    runner.logger.info(f"Structure prediction completed: {len(filtered_results)} passed filters")
    
    return results_df
