"""
Epitope prediction module using IEDB API and NetMHCIIpan.

This module provides functionality to predict CD4+ T-cell epitopes
from protein sequences using various prediction methods.
"""

import os
import logging
import pandas as pd
import requests
from typing import List, Dict, Optional
from pathlib import Path


class EpitopePredictor:
    """Class for predicting CD4+ T-cell epitopes."""
    
    def __init__(self, hla_alleles: Optional[List[str]] = None):
        """
        Initialize the epitope predictor.
        
        Args:
            hla_alleles: List of HLA alleles to use for prediction
        """
        self.hla_alleles = hla_alleles or [
            "DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01",
            "DRB1*08:01", "DRB1*11:01", "DRB1*13:01", "DRB1*15:01"
        ]
        self.logger = logging.getLogger(__name__)
    
    def read_fasta(self, fasta_path: str) -> Dict[str, str]:
        """
        Read FASTA file and return sequences.
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]  # Remove '>' and take first word
                    current_seq = []
                else:
                    current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def predict_iedb_epitopes(self, sequence: str, sequence_id: str) -> pd.DataFrame:
        """
        Predict epitopes using IEDB API.
        
        Args:
            sequence: Protein sequence
            sequence_id: Sequence identifier
            
        Returns:
            DataFrame with epitope predictions
        """
        # This is a placeholder implementation
        # In practice, you would integrate with the actual IEDB API
        self.logger.info(f"Predicting epitopes for sequence {sequence_id} using IEDB API")
        
        # Mock data for demonstration
        epitopes = []
        for i in range(0, len(sequence) - 8, 10):  # 9-mer peptides every 10 positions
            peptide = sequence[i:i+9]
            if len(peptide) == 9:
                epitopes.append({
                    'sequence_id': sequence_id,
                    'peptide': peptide,
                    'start_position': i + 1,
                    'end_position': i + 9,
                    'method': 'IEDB',
                    'score': 0.5 + (i % 10) * 0.05  # Mock score
                })
        
        return pd.DataFrame(epitopes)
    
    def predict_netmhcii_epitopes(self, sequence: str, sequence_id: str) -> pd.DataFrame:
        """
        Predict epitopes using NetMHCIIpan.
        
        Args:
            sequence: Protein sequence
            sequence_id: Sequence identifier
            
        Returns:
            DataFrame with epitope predictions
        """
        # This is a placeholder implementation
        # In practice, you would integrate with NetMHCIIpan
        self.logger.info(f"Predicting epitopes for sequence {sequence_id} using NetMHCIIpan")
        
        epitopes = []
        for i in range(0, len(sequence) - 14, 5):  # 15-mer peptides every 5 positions
            peptide = sequence[i:i+15]
            if len(peptide) == 15:
                epitopes.append({
                    'sequence_id': sequence_id,
                    'peptide': peptide,
                    'start_position': i + 1,
                    'end_position': i + 15,
                    'method': 'NetMHCIIpan',
                    'score': 0.3 + (i % 15) * 0.03  # Mock score
                })
        
        return pd.DataFrame(epitopes)
    
    def combine_predictions(self, predictions: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine predictions from different methods.
        
        Args:
            predictions: List of prediction DataFrames
            
        Returns:
            Combined DataFrame with all predictions
        """
        if not predictions:
            return pd.DataFrame()
        
        combined = pd.concat(predictions, ignore_index=True)
        
        # Remove duplicates based on peptide and position
        combined = combined.drop_duplicates(subset=['peptide', 'start_position'])
        
        # Sort by score (higher is better)
        combined = combined.sort_values('score', ascending=False)
        
        return combined


def identify_epitopes(fasta_path: str, hla_alleles: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Identify CD4+ T-cell epitopes from FASTA file.
    
    Args:
        fasta_path: Path to input FASTA file
        hla_alleles: List of HLA alleles for prediction
        
    Returns:
        DataFrame containing epitope predictions
    """
    predictor = EpitopePredictor(hla_alleles)
    sequences = predictor.read_fasta(fasta_path)
    
    all_predictions = []
    
    for seq_id, sequence in sequences.items():
        predictor.logger.info(f"Processing sequence {seq_id} (length: {len(sequence)})")
        
        # Get predictions from different methods
        iedb_preds = predictor.predict_iedb_epitopes(sequence, seq_id)
        netmhcii_preds = predictor.predict_netmhcii_epitopes(sequence, seq_id)
        
        # Combine predictions
        combined = predictor.combine_predictions([iedb_preds, netmhcii_preds])
        all_predictions.append(combined)
    
    # Combine all sequence predictions
    final_predictions = predictor.combine_predictions(all_predictions)
    
    predictor.logger.info(f"Total epitopes predicted: {len(final_predictions)}")
    
    return final_predictions
