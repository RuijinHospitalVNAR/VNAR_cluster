#!/usr/bin/env python3
"""
Immunogenicity Optimization Pipeline

A comprehensive computational workflow for modulating protein immunogenicity through:
1. Epitope identification using IEDB API and NetMHCIIpan
2. Sequence generation with ProteinMPNN from selected epitope regions
3. MHC-II affinity evaluation across HLA-DRB1 alleles using NetMHCIIpan
4. Scoring based on immunogenicity adjustment direction
5. Structure simulation with AlphaFold3 and filtering based on RMSD/DOCKQ thresholds

Author: [Your Name]
Version: 2.0
Date: 2025
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

# Import custom tools (assuming they exist in tools/ directory)
try:
    from tools.epitope_predictor import identify_epitopes
    from tools.protein_mpnn_wrapper import generate_mutants
    from tools.netmhcii_runner import evaluate_mhc_affinity
    from tools.alphafold3_runner import predict_structure_and_score
except ImportError as e:
    logging.error(f"Failed to import required tools: {e}")
    logging.error("Please ensure all required tools are available in the tools/ directory")
    sys.exit(1)


class ImmunogenicityMode(Enum):
    """Enumeration for immunogenicity modulation modes."""
    REDUCE = "reduce"
    ENHANCE = "enhance"


@dataclass
class PipelineConfig:
    """Configuration class for the immunogenicity optimization pipeline."""
    # Input parameters
    fasta_path: str
    pdb_path: str
    mode: ImmunogenicityMode
    
    # Output parameters
    output_dir: str = "results"
    log_level: str = "INFO"
    
    # ProteinMPNN parameters
    samples_per_temp: int = 20
    temperatures: List[float] = None
    
    # MHC-II parameters
    hla_alleles: List[str] = None
    
    # Structure prediction parameters
    max_candidates: int = 10
    rmsd_threshold: float = 2.0
    dockq_threshold: float = 0.23
    
    # Scoring parameters
    neutral_score: float = 50.0
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.temperatures is None:
            self.temperatures = [0.1, 0.3, 0.5]
        if self.hla_alleles is None:
            self.hla_alleles = [
                "DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01",
                "DRB1*08:01", "DRB1*11:01", "DRB1*13:01", "DRB1*15:01"
            ]


class ImmunogenicityOptimizer:
    """Main class for immunogenicity optimization pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the optimizer with configuration."""
        self.config = config
        self.logger = self._setup_logging()
        self.start_time = time.time()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Configure logging
        log_file = os.path.join(self.config.output_dir, "pipeline.log")
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _validate_inputs(self) -> None:
        """Validate input files and parameters."""
        self.logger.info("Validating input parameters...")
        
        # Check FASTA file
        if not os.path.exists(self.config.fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {self.config.fasta_path}")
        
        # Check PDB file
        if not os.path.exists(self.config.pdb_path):
            raise FileNotFoundError(f"PDB file not found: {self.config.pdb_path}")
        
        # Validate file extensions
        if not self.config.fasta_path.lower().endswith(('.fasta', '.fa')):
            raise ValueError(f"Invalid FASTA file extension: {self.config.fasta_path}")
        
        if not self.config.pdb_path.lower().endswith('.pdb'):
            raise ValueError(f"Invalid PDB file extension: {self.config.pdb_path}")
        
        # Validate mode
        if not isinstance(self.config.mode, ImmunogenicityMode):
            if isinstance(self.config.mode, str):
                self.config.mode = ImmunogenicityMode(self.config.mode.lower())
            else:
                raise ValueError(f"Invalid mode: {self.config.mode}")
        
        self.logger.info("Input validation completed successfully")
    
    def _save_config(self) -> None:
        """Save configuration to file for reproducibility."""
        config_file = os.path.join(self.config.output_dir, "config.json")
        config_dict = asdict(self.config)
        # Convert enum to string for JSON serialization
        config_dict['mode'] = self.config.mode.value
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def compute_immunogenicity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute immunogenicity scores based on MHC-II binding ranks.
        
        Args:
            df: DataFrame containing MHC-II binding rank columns
            
        Returns:
            DataFrame with added score columns
        """
        self.logger.info("Computing immunogenicity scores...")
        
        if df.empty:
            self.logger.warning("Empty DataFrame provided for scoring")
            return df
        
        score_columns = []
        mode = self.config.mode.value
        
        for col in df.columns:
            if col.startswith('Rank_'):
                allele = col.replace('Rank_', '')
                min_rank, max_rank = df[col].min(), df[col].max()
                
                # Avoid division by zero
                if max_rank == min_rank:
                    df[f'Score_{allele}'] = self.config.neutral_score
                    self.logger.warning(f"Identical ranks for {allele}, using neutral score")
                else:
                    if mode == 'reduce':
                        # Lower ranks (stronger binding) get lower scores for reduction
                        df[f'Score_{allele}'] = 100 - ((df[col] - min_rank) / (max_rank - min_rank)) * 100
                    else:  # enhance
                        # Lower ranks (stronger binding) get higher scores for enhancement
                        df[f'Score_{allele}'] = ((df[col] - min_rank) / (max_rank - min_rank)) * 100
                
                score_columns.append(f'Score_{allele}')
        
        # Add overall immunogenicity score
        if score_columns:
            df['Overall_Immunogenicity_Score'] = df[score_columns].mean(axis=1)
            self.logger.info(f"Computed scores for {len(score_columns)} alleles")
        else:
            self.logger.warning("No rank columns found for scoring")
        
        return df
    
    def predict_epitopes(self) -> pd.DataFrame:
        """Predict CD4+ T-cell epitopes."""
        self.logger.info("Step 1: Predicting CD4+ T-cell epitopes...")
        
        try:
            epitope_df = identify_epitopes(self.config.fasta_path)
            
            if epitope_df.empty:
                raise ValueError("No epitopes predicted")
            
            # Save results
            epitope_file = os.path.join(self.config.output_dir, "epitope_predictions.csv")
            epitope_df.to_csv(epitope_file, index=False)
            
            self.logger.info(f"Found {len(epitope_df)} epitope predictions")
            self.logger.info(f"Epitope predictions saved to {epitope_file}")
            
            return epitope_df
            
        except Exception as e:
            self.logger.error(f"Epitope prediction failed: {e}")
            raise
    
    def generate_mutant_sequences(self, epitope_df: pd.DataFrame) -> List[str]:
        """Generate mutant sequences using ProteinMPNN."""
        self.logger.info("Step 2: Generating mutant sequences with ProteinMPNN...")
        
        try:
            mutants = generate_mutants(
                self.config.pdb_path,
                epitope_df,
                self.config.mode.value,
                samples_per_temp=self.config.samples_per_temp,
                temps=self.config.temperatures
            )
            
            if not mutants:
                raise ValueError("No mutant sequences generated")
            
            # Save mutant sequences
            mutant_file = os.path.join(self.config.output_dir, "mutant_sequences.fasta")
            with open(mutant_file, 'w') as f:
                for i, seq in enumerate(mutants):
                    f.write(f">mutant_{i:04d}\n{seq}\n")
            
            self.logger.info(f"Generated {len(mutants)} mutant sequences")
            self.logger.info(f"Mutant sequences saved to {mutant_file}")
            
            return mutants
            
        except Exception as e:
            self.logger.error(f"Mutant generation failed: {e}")
            raise
    
    def evaluate_mhc_binding(self, mutant_file: str) -> pd.DataFrame:
        """Evaluate MHC-II binding affinity."""
        self.logger.info("Step 3: Evaluating MHC-II binding affinity...")
        
        try:
            affinity_df = evaluate_mhc_affinity(mutant_file)
            
            if affinity_df.empty:
                raise ValueError("No MHC-II binding results obtained")
            
            # Compute immunogenicity scores
            affinity_df = self.compute_immunogenicity_scores(affinity_df)
            
            # Save results
            affinity_file = os.path.join(self.config.output_dir, "mhc_binding_scores.csv")
            affinity_df.to_csv(affinity_file, index=False)
            
            self.logger.info(f"Evaluated {len(affinity_df)} sequences for MHC-II binding")
            self.logger.info(f"MHC binding scores saved to {affinity_file}")
            
            return affinity_df
            
        except Exception as e:
            self.logger.error(f"MHC-II evaluation failed: {e}")
            raise
    
    def predict_structures_and_rank(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
        """Predict structures and rank final candidates."""
        self.logger.info("Step 4: Predicting structures and filtering final candidates...")
        
        try:
            # Select top candidates based on overall immunogenicity score
            if self.config.mode == ImmunogenicityMode.ENHANCE:
                top_candidates = affinity_df.nlargest(
                    self.config.max_candidates, 
                    'Overall_Immunogenicity_Score'
                )
            else:
                top_candidates = affinity_df.nsmallest(
                    self.config.max_candidates, 
                    'Overall_Immunogenicity_Score'
                )
            
            if len(top_candidates) == 0:
                raise ValueError("No candidates selected for structure prediction")
            
            self.logger.info(f"Selected {len(top_candidates)} top candidates for structure prediction")
            
            # Predict structures and apply filtering
            final_results = predict_structure_and_score(
                top_candidates,
                rmsd_threshold=self.config.rmsd_threshold,
                dockq_threshold=self.config.dockq_threshold
            )
            
            # Save final results
            final_file = os.path.join(self.config.output_dir, "final_ranked_candidates.csv")
            final_results.to_csv(final_file, index=False)
            
            self.logger.info(f"Final candidates: {len(final_results)}")
            self.logger.info(f"Final results saved to {final_file}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Structure prediction failed: {e}")
            raise
    
    def run_pipeline(self) -> Dict[str, Union[pd.DataFrame, List[str]]]:
        """
        Run the complete immunogenicity optimization pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        self.logger.info(f"Starting immunogenicity {self.config.mode.value} pipeline...")
        self.logger.info(f"Input FASTA: {self.config.fasta_path}")
        self.logger.info(f"Input PDB: {self.config.pdb_path}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        
        try:
            # Validate inputs
            self._validate_inputs()
            
            # Save configuration
            self._save_config()
            
            # Run pipeline steps
            epitope_df = self.predict_epitopes()
            mutants = self.generate_mutant_sequences(epitope_df)
            mutant_file = os.path.join(self.config.output_dir, "mutant_sequences.fasta")
            affinity_df = self.evaluate_mhc_binding(mutant_file)
            final_results = self.predict_structures_and_rank(affinity_df)
            
            # Calculate runtime
            runtime = time.time() - self.start_time
            
            # Log completion
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total runtime: {runtime:.2f} seconds")
            self.logger.info("Results saved in ./results directory:")
            self.logger.info("  - epitope_predictions.csv")
            self.logger.info("  - mutant_sequences.fasta")
            self.logger.info("  - mhc_binding_scores.csv")
            self.logger.info("  - final_ranked_candidates.csv")
            self.logger.info("  - config.json")
            self.logger.info("  - pipeline.log")
            self.logger.info("=" * 60)
            
            return {
                'epitopes': epitope_df,
                'mutants': mutants,
                'mhc_binding': affinity_df,
                'final_candidates': final_results,
                'runtime': runtime
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error("Check the log file for detailed error information")
            raise


def create_config_from_args(args) -> PipelineConfig:
    """Create configuration from command line arguments."""
    return PipelineConfig(
        fasta_path=args.fasta,
        pdb_path=args.pdb,
        mode=ImmunogenicityMode(args.mode),
        output_dir=args.output_dir,
        log_level=args.log_level,
        samples_per_temp=args.samples_per_temp,
        temperatures=args.temperatures,
        max_candidates=args.max_candidates,
        rmsd_threshold=args.rmsd_threshold,
        dockq_threshold=args.dockq_threshold
    )


def main():
    """Main entry point for the immunogenicity optimization pipeline."""
    parser = argparse.ArgumentParser(
        description="Immunogenicity Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - reduce immunogenicity
  python immunogenicity_optimization_pipeline.py --fasta protein.fasta --pdb protein.pdb --mode reduce
  
  # Enhance immunogenicity with custom parameters
  python immunogenicity_optimization_pipeline.py --fasta protein.fasta --pdb protein.pdb --mode enhance --max-candidates 20 --samples-per-temp 50
  
  # Use custom output directory and log level
  python immunogenicity_optimization_pipeline.py --fasta protein.fasta --pdb protein.pdb --mode reduce --output-dir custom_results --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file path')
    parser.add_argument('--pdb', type=str, required=True,
                       help='Input PDB file path')
    parser.add_argument('--mode', type=str, choices=['reduce', 'enhance'],
                       default='reduce',
                       help='Modulate immunogenicity: reduce or enhance (default: reduce)')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    # ProteinMPNN parameters
    parser.add_argument('--samples-per-temp', type=int, default=20,
                       help='Number of samples per temperature (default: 20)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.1, 0.3, 0.5],
                       help='Temperature values for ProteinMPNN (default: 0.1 0.3 0.5)')
    
    # Structure prediction parameters
    parser.add_argument('--max-candidates', type=int, default=10,
                       help='Maximum number of candidates for structure prediction (default: 10)')
    parser.add_argument('--rmsd-threshold', type=float, default=2.0,
                       help='RMSD threshold for structure filtering (default: 2.0)')
    parser.add_argument('--dockq-threshold', type=float, default=0.23,
                       help='DOCKQ threshold for structure filtering (default: 0.23)')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Initialize and run pipeline
        optimizer = ImmunogenicityOptimizer(config)
        results = optimizer.run_pipeline()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total runtime: {results['runtime']:.2f} seconds")
        print(f"Results saved in: {config.output_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
