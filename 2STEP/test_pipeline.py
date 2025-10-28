#!/usr/bin/env python3
"""
Test script for the Immunogenicity Optimization Pipeline.

This script tests the pipeline components and validates the code structure.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from immunogenicity_optimization_pipeline import (
    ImmunogenicityOptimizer, 
    PipelineConfig, 
    ImmunogenicityMode
)


class TestPipelineConfig(unittest.TestCase):
    """Test cases for PipelineConfig class."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = PipelineConfig(
            fasta_path="test.fasta",
            pdb_path="test.pdb",
            mode=ImmunogenicityMode.REDUCE
        )
        
        self.assertEqual(config.fasta_path, "test.fasta")
        self.assertEqual(config.pdb_path, "test.pdb")
        self.assertEqual(config.mode, ImmunogenicityMode.REDUCE)
        self.assertEqual(config.output_dir, "results")
        self.assertEqual(config.log_level, "INFO")
    
    def test_config_defaults(self):
        """Test config default values."""
        config = PipelineConfig(
            fasta_path="test.fasta",
            pdb_path="test.pdb",
            mode=ImmunogenicityMode.ENHANCE
        )
        
        self.assertEqual(config.samples_per_temp, 20)
        self.assertEqual(config.temperatures, [0.1, 0.3, 0.5])
        self.assertEqual(config.max_candidates, 10)
        self.assertEqual(config.rmsd_threshold, 2.0)
        self.assertEqual(config.dockq_threshold, 0.23)
        self.assertEqual(config.neutral_score, 50.0)


class TestImmunogenicityOptimizer(unittest.TestCase):
    """Test cases for ImmunogenicityOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PipelineConfig(
            fasta_path="test.fasta",
            pdb_path="test.pdb",
            mode=ImmunogenicityMode.REDUCE,
            output_dir="test_results"
        )
        self.optimizer = ImmunogenicityOptimizer(self.config)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.config, self.config)
        self.assertIsNotNone(self.optimizer.logger)
        self.assertIsNotNone(self.optimizer.start_time)
    
    def test_compute_immunogenicity_scores_reduce(self):
        """Test immunogenicity score computation for reduction mode."""
        import pandas as pd
        
        # Create test data
        df = pd.DataFrame({
            'sequence_id': ['seq1', 'seq2', 'seq3'],
            'Rank_DRB1*01:01': [1, 5, 10],
            'Rank_DRB1*03:01': [2, 4, 8]
        })
        
        # Test reduce mode
        result_df = self.optimizer.compute_immunogenicity_scores(df)
        
        # Check that score columns were added
        self.assertIn('Score_DRB1*01:01', result_df.columns)
        self.assertIn('Score_DRB1*03:01', result_df.columns)
        self.assertIn('Overall_Immunogenicity_Score', result_df.columns)
        
        # Check score values (lower ranks should get lower scores in reduce mode)
        self.assertLess(result_df.loc[0, 'Score_DRB1*01:01'], result_df.loc[2, 'Score_DRB1*01:01'])
    
    def test_compute_immunogenicity_scores_enhance(self):
        """Test immunogenicity score computation for enhancement mode."""
        import pandas as pd
        
        # Change mode to enhance
        self.optimizer.config.mode = ImmunogenicityMode.ENHANCE
        
        # Create test data
        df = pd.DataFrame({
            'sequence_id': ['seq1', 'seq2', 'seq3'],
            'Rank_DRB1*01:01': [1, 5, 10],
            'Rank_DRB1*03:01': [2, 4, 8]
        })
        
        # Test enhance mode
        result_df = self.optimizer.compute_immunogenicity_scores(df)
        
        # Check that score columns were added
        self.assertIn('Score_DRB1*01:01', result_df.columns)
        self.assertIn('Score_DRB1*03:01', result_df.columns)
        self.assertIn('Overall_Immunogenicity_Score', result_df.columns)
        
        # Check score values (lower ranks should get higher scores in enhance mode)
        self.assertGreater(result_df.loc[0, 'Score_DRB1*01:01'], result_df.loc[2, 'Score_DRB1*01:01'])
    
    def test_compute_immunogenicity_scores_empty_df(self):
        """Test immunogenicity score computation with empty DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame()
        result_df = self.optimizer.compute_immunogenicity_scores(df)
        
        self.assertTrue(result_df.empty)
    
    def test_compute_immunogenicity_scores_identical_ranks(self):
        """Test immunogenicity score computation with identical ranks."""
        import pandas as pd
        
        # Create test data with identical ranks
        df = pd.DataFrame({
            'sequence_id': ['seq1', 'seq2', 'seq3'],
            'Rank_DRB1*01:01': [5, 5, 5],
            'Rank_DRB1*03:01': [3, 3, 3]
        })
        
        result_df = self.optimizer.compute_immunogenicity_scores(df)
        
        # Check that neutral scores were assigned
        self.assertEqual(result_df.loc[0, 'Score_DRB1*01:01'], 50.0)
        self.assertEqual(result_df.loc[0, 'Score_DRB1*03:01'], 50.0)


class TestInputValidation(unittest.TestCase):
    """Test cases for input validation."""
    
    def test_validate_inputs_success(self):
        """Test successful input validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            fasta_file = os.path.join(temp_dir, "test.fasta")
            pdb_file = os.path.join(temp_dir, "test.pdb")
            
            with open(fasta_file, 'w') as f:
                f.write(">test_sequence\nATCGATCGATCG\n")
            
            with open(pdb_file, 'w') as f:
                f.write("ATOM      1  N   ALA A   1      20.154  16.967  23.862  1.00 11.18           N\n")
            
            config = PipelineConfig(
                fasta_path=fasta_file,
                pdb_path=pdb_file,
                mode=ImmunogenicityMode.REDUCE
            )
            
            optimizer = ImmunogenicityOptimizer(config)
            
            # Should not raise an exception
            optimizer._validate_inputs()
    
    def test_validate_inputs_missing_fasta(self):
        """Test input validation with missing FASTA file."""
        config = PipelineConfig(
            fasta_path="nonexistent.fasta",
            pdb_path="test.pdb",
            mode=ImmunogenicityMode.REDUCE
        )
        
        optimizer = ImmunogenicityOptimizer(config)
        
        with self.assertRaises(FileNotFoundError):
            optimizer._validate_inputs()
    
    def test_validate_inputs_missing_pdb(self):
        """Test input validation with missing PDB file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_file = os.path.join(temp_dir, "test.fasta")
            
            with open(fasta_file, 'w') as f:
                f.write(">test_sequence\nATCGATCGATCG\n")
            
            config = PipelineConfig(
                fasta_path=fasta_file,
                pdb_path="nonexistent.pdb",
                mode=ImmunogenicityMode.REDUCE
            )
            
            optimizer = ImmunogenicityOptimizer(config)
            
            with self.assertRaises(FileNotFoundError):
                optimizer._validate_inputs()
    
    def test_validate_inputs_invalid_extension(self):
        """Test input validation with invalid file extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with wrong extensions
            fasta_file = os.path.join(temp_dir, "test.txt")
            pdb_file = os.path.join(temp_dir, "test.txt")
            
            with open(fasta_file, 'w') as f:
                f.write(">test_sequence\nATCGATCGATCG\n")
            
            with open(pdb_file, 'w') as f:
                f.write("ATOM      1  N   ALA A   1      20.154  16.967  23.862  1.00 11.18           N\n")
            
            config = PipelineConfig(
                fasta_path=fasta_file,
                pdb_path=pdb_file,
                mode=ImmunogenicityMode.REDUCE
            )
            
            optimizer = ImmunogenicityOptimizer(config)
            
            with self.assertRaises(ValueError):
                optimizer._validate_inputs()


class TestMockPipeline(unittest.TestCase):
    """Test cases with mocked external dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PipelineConfig(
            fasta_path="test.fasta",
            pdb_path="test.pdb",
            mode=ImmunogenicityMode.REDUCE,
            output_dir="test_results"
        )
    
    @patch('tools.epitope_predictor.identify_epitopes')
    @patch('tools.protein_mpnn_wrapper.generate_mutants')
    @patch('tools.netmhcii_runner.evaluate_mhc_affinity')
    @patch('tools.alphafold3_runner.predict_structure_and_score')
    def test_mock_pipeline_run(self, mock_af3, mock_mhc, mock_mpnn, mock_epitopes):
        """Test pipeline run with mocked dependencies."""
        import pandas as pd
        
        # Mock return values
        mock_epitopes.return_value = pd.DataFrame({
            'peptide': ['ATCGATCG', 'GCTAGCTA'],
            'start_position': [1, 10],
            'end_position': [8, 17],
            'score': [0.8, 0.6]
        })
        
        mock_mpnn.return_value = ['ATCGATCGATCG', 'GCTAGCTAGCTA']
        
        mock_mhc.return_value = pd.DataFrame({
            'sequence_id': ['mutant_0000', 'mutant_0001'],
            'Rank_DRB1*01:01': [1, 5],
            'Rank_DRB1*03:01': [2, 4]
        })
        
        mock_af3.return_value = pd.DataFrame({
            'sequence_id': ['mutant_0000', 'mutant_0001'],
            'pdb_file': ['path1.pdb', 'path2.pdb'],
            'confidence_score': [85.0, 75.0],
            'rmsd': [1.5, 2.1],
            'dockq': [0.3, 0.25]
        })
        
        # Create optimizer
        optimizer = ImmunogenicityOptimizer(self.config)
        
        # Run pipeline
        results = optimizer.run_pipeline()
        
        # Verify results
        self.assertIn('epitopes', results)
        self.assertIn('mutants', results)
        self.assertIn('mhc_binding', results)
        self.assertIn('final_candidates', results)
        self.assertIn('runtime', results)
        
        # Verify mock calls
        mock_epitopes.assert_called_once()
        mock_mpnn.assert_called_once()
        mock_mhc.assert_called_once()
        mock_af3.assert_called_once()


def run_tests():
    """Run all tests."""
    print("Running Immunogenicity Optimization Pipeline Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPipelineConfig))
    test_suite.addTest(unittest.makeSuite(TestImmunogenicityOptimizer))
    test_suite.addTest(unittest.makeSuite(TestInputValidation))
    test_suite.addTest(unittest.makeSuite(TestMockPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
            print(f"      {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"       {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
