#!/usr/bin/env python3
"""
Example usage of the Immunogenicity Optimization Pipeline.

This script demonstrates how to use the pipeline programmatically
with custom configuration and parameters.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from immunogenicity_optimization_pipeline import ImmunogenicityOptimizer, PipelineConfig, ImmunogenicityMode


def example_basic_usage():
    """Example of basic pipeline usage."""
    print("=== Basic Usage Example ===")
    
    # Create configuration
    config = PipelineConfig(
        fasta_path="examples/protein.fasta",
        pdb_path="examples/protein.pdb",
        mode=ImmunogenicityMode.REDUCE,
        output_dir="results_basic",
        log_level="INFO"
    )
    
    # Initialize optimizer
    optimizer = ImmunogenicityOptimizer(config)
    
    try:
        # Run pipeline
        results = optimizer.run_pipeline()
        
        print(f"Pipeline completed successfully!")
        print(f"Generated {len(results['mutants'])} mutant sequences")
        print(f"Found {len(results['epitopes'])} epitopes")
        print(f"Final candidates: {len(results['final_candidates'])}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")


def example_advanced_usage():
    """Example of advanced pipeline usage with custom parameters."""
    print("\n=== Advanced Usage Example ===")
    
    # Create advanced configuration
    config = PipelineConfig(
        fasta_path="examples/protein.fasta",
        pdb_path="examples/protein.pdb",
        mode=ImmunogenicityMode.ENHANCE,
        output_dir="results_advanced",
        log_level="DEBUG",
        samples_per_temp=50,
        temperatures=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        max_candidates=20,
        rmsd_threshold=1.5,
        dockq_threshold=0.3
    )
    
    # Initialize optimizer
    optimizer = ImmunogenicityOptimizer(config)
    
    try:
        # Run pipeline
        results = optimizer.run_pipeline()
        
        print(f"Advanced pipeline completed successfully!")
        print(f"Generated {len(results['mutants'])} mutant sequences")
        print(f"Found {len(results['epitopes'])} epitopes")
        print(f"Final candidates: {len(results['final_candidates'])}")
        print(f"Runtime: {results['runtime']:.2f} seconds")
        
    except Exception as e:
        print(f"Advanced pipeline failed: {e}")


def example_batch_processing():
    """Example of batch processing multiple proteins."""
    print("\n=== Batch Processing Example ===")
    
    # List of proteins to process
    proteins = [
        {"fasta": "examples/protein1.fasta", "pdb": "examples/protein1.pdb", "mode": "reduce"},
        {"fasta": "examples/protein2.fasta", "pdb": "examples/protein2.pdb", "mode": "enhance"},
        {"fasta": "examples/protein3.fasta", "pdb": "examples/protein3.pdb", "mode": "reduce"},
    ]
    
    results_summary = []
    
    for i, protein in enumerate(proteins):
        print(f"\nProcessing protein {i+1}/{len(proteins)}: {protein['fasta']}")
        
        # Create configuration for this protein
        config = PipelineConfig(
            fasta_path=protein["fasta"],
            pdb_path=protein["pdb"],
            mode=ImmunogenicityMode(protein["mode"]),
            output_dir=f"results_protein_{i+1}",
            log_level="INFO"
        )
        
        # Initialize optimizer
        optimizer = ImmunogenicityOptimizer(config)
        
        try:
            # Run pipeline
            results = optimizer.run_pipeline()
            
            # Store summary
            results_summary.append({
                "protein": protein["fasta"],
                "mode": protein["mode"],
                "mutants": len(results["mutants"]),
                "epitopes": len(results["epitopes"]),
                "final_candidates": len(results["final_candidates"]),
                "runtime": results["runtime"]
            })
            
            print(f"✓ Protein {i+1} completed successfully")
            
        except Exception as e:
            print(f"✗ Protein {i+1} failed: {e}")
            results_summary.append({
                "protein": protein["fasta"],
                "mode": protein["mode"],
                "mutants": 0,
                "epitopes": 0,
                "final_candidates": 0,
                "runtime": 0,
                "error": str(e)
            })
    
    # Print summary
    print("\n=== Batch Processing Summary ===")
    for result in results_summary:
        if "error" in result:
            print(f"❌ {result['protein']}: FAILED - {result['error']}")
        else:
            print(f"✅ {result['protein']}: {result['mutants']} mutants, "
                  f"{result['epitopes']} epitopes, {result['final_candidates']} candidates "
                  f"({result['runtime']:.1f}s)")


def example_custom_scoring():
    """Example of custom scoring parameters."""
    print("\n=== Custom Scoring Example ===")
    
    # Create configuration with custom scoring
    config = PipelineConfig(
        fasta_path="examples/protein.fasta",
        pdb_path="examples/protein.pdb",
        mode=ImmunogenicityMode.REDUCE,
        output_dir="results_custom_scoring",
        log_level="INFO",
        neutral_score=75.0,  # Custom neutral score
        rmsd_threshold=1.0,  # Stricter RMSD threshold
        dockq_threshold=0.4  # Stricter DOCKQ threshold
    )
    
    # Initialize optimizer
    optimizer = ImmunogenicityOptimizer(config)
    
    try:
        # Run pipeline
        results = optimizer.run_pipeline()
        
        print(f"Custom scoring pipeline completed successfully!")
        print(f"Using neutral score: {config.neutral_score}")
        print(f"Using RMSD threshold: {config.rmsd_threshold}")
        print(f"Using DOCKQ threshold: {config.dockq_threshold}")
        
    except Exception as e:
        print(f"Custom scoring pipeline failed: {e}")


if __name__ == "__main__":
    print("Immunogenicity Optimization Pipeline - Usage Examples")
    print("=" * 60)
    
    # Check if example files exist
    example_files = [
        "examples/protein.fasta",
        "examples/protein.pdb"
    ]
    
    missing_files = [f for f in example_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing example files: {missing_files}")
        print("Please create example files or modify the paths in this script")
        print()
    
    # Run examples
    try:
        example_basic_usage()
        example_advanced_usage()
        example_batch_processing()
        example_custom_scoring()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nExamples failed: {e}")
