"""
NetMHCIIpan runner for MHC-II binding affinity evaluation.

This module provides functionality to evaluate MHC-II binding affinity
using NetMHCIIpan for generated mutant sequences.
"""

import os
import logging
import subprocess
import tempfile
import pandas as pd
from typing import List, Dict, Optional


class NetMHCIIRunner:
    """Class for running NetMHCIIpan predictions."""
    
    def __init__(self, netmhcii_path: Optional[str] = None, hla_alleles: Optional[List[str]] = None):
        """
        Initialize the NetMHCIIpan runner.
        
        Args:
            netmhcii_path: Path to NetMHCIIpan executable
            hla_alleles: List of HLA alleles to evaluate
        """
        self.netmhcii_path = netmhcii_path or "netMHCIIpan"
        self.hla_alleles = hla_alleles or [
            "DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01",
            "DRB1*08:01", "DRB1*11:01", "DRB1*13:01", "DRB1*15:01"
        ]
        self.logger = logging.getLogger(__name__)
    
    def read_fasta_sequences(self, fasta_path: str) -> Dict[str, str]:
        """
        Read sequences from FASTA file.
        
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
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def run_netmhcii(self, fasta_path: str, output_dir: str) -> pd.DataFrame:
        """
        Run NetMHCIIpan on FASTA sequences.
        
        Args:
            fasta_path: Path to input FASTA file
            output_dir: Output directory for results
            
        Returns:
            DataFrame with binding predictions
        """
        self.logger.info(f"Running NetMHCIIpan on {len(self.hla_alleles)} alleles")
        
        # Create output file
        output_file = os.path.join(output_dir, "netmhcii_results.txt")
        
        # Prepare command
        hla_string = ",".join(self.hla_alleles)
        cmd = [
            self.netmhcii_path,
            "-f", fasta_path,
            "-a", hla_string,
            "-xls",
            "-xlsfile", output_file
        ]
        
        try:
            # Run NetMHCIIpan
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self.logger.error(f"NetMHCIIpan failed: {result.stderr}")
                return pd.DataFrame()
            
            # Parse results
            results_df = self._parse_netmhcii_output(output_file)
            
            self.logger.info(f"NetMHCIIpan completed: {len(results_df)} predictions")
            return results_df
            
        except subprocess.TimeoutExpired:
            self.logger.error("NetMHCIIpan timed out")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error running NetMHCIIpan: {e}")
            return pd.DataFrame()
    
    def _parse_netmhcii_output(self, output_file: str) -> pd.DataFrame:
        """
        Parse NetMHCIIpan output file.
        
        Args:
            output_file: Path to NetMHCIIpan output file
            
        Returns:
            DataFrame with parsed results
        """
        results = []
        
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and parse data
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Pos'):
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        try:
                            result = {
                                'sequence_id': parts[0],
                                'peptide': parts[1],
                                'allele': parts[2],
                                'affinity': float(parts[3]),
                                'rank': float(parts[4]),
                                'core': parts[5] if len(parts) > 5 else ''
                            }
                            results.append(result)
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Failed to parse line: {line} - {e}")
                            continue
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            if not df.empty:
                # Pivot to have alleles as columns
                pivot_df = df.pivot_table(
                    index=['sequence_id', 'peptide'],
                    columns='allele',
                    values='rank',
                    aggfunc='first'
                ).reset_index()
                
                # Rename columns to include Rank_ prefix
                rank_columns = {col: f'Rank_{col}' for col in pivot_df.columns 
                              if col not in ['sequence_id', 'peptide']}
                pivot_df = pivot_df.rename(columns=rank_columns)
                
                return pivot_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error parsing NetMHCIIpan output: {e}")
            return pd.DataFrame()


def evaluate_mhc_affinity(fasta_path: str, hla_alleles: Optional[List[str]] = None,
                         netmhcii_path: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate MHC-II binding affinity for sequences in FASTA file.
    
    Args:
        fasta_path: Path to input FASTA file
        hla_alleles: List of HLA alleles to evaluate
        netmhcii_path: Path to NetMHCIIpan executable
        
    Returns:
        DataFrame with MHC-II binding predictions
    """
    runner = NetMHCIIRunner(netmhcii_path, hla_alleles)
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run NetMHCIIpan
    results_df = runner.run_netmhcii(fasta_path, output_dir)
    
    return results_df
