# Immunogenicity Optimization Pipeline

A comprehensive computational workflow for modulating protein immunogenicity through epitope identification, sequence generation, MHC-II binding evaluation, and structure prediction.

## Features

- **Epitope Prediction**: Uses IEDB API and NetMHCIIpan for CD4+ T-cell epitope identification
- **Sequence Generation**: Leverages ProteinMPNN for generating mutant sequences
- **MHC-II Evaluation**: Evaluates binding affinity across multiple HLA-DRB1 alleles
- **Structure Prediction**: Uses AlphaFold3 for structure prediction and filtering
- **Flexible Configuration**: Supports both immunogenicity reduction and enhancement modes
- **Comprehensive Logging**: Detailed logging and error handling throughout the pipeline

## Installation

### Prerequisites

- Python 3.8+
- Required tools:
  - ProteinMPNN
  - NetMHCIIpan
  - AlphaFold3
  - IEDB API access

### Dependencies

```bash
pip install pandas numpy requests
```

## Quick Start

### Basic Usage

```bash
# Reduce immunogenicity
python immunogenicity_optimization_pipeline.py --fasta protein.fasta --pdb protein.pdb --mode reduce

# Enhance immunogenicity
python immunogenicity_optimization_pipeline.py --fasta protein.fasta --pdb protein.pdb --mode enhance
```

### Advanced Usage

```bash
# Custom parameters
python immunogenicity_optimization_pipeline.py \
    --fasta protein.fasta \
    --pdb protein.pdb \
    --mode reduce \
    --output-dir custom_results \
    --samples-per-temp 50 \
    --temperatures 0.1 0.2 0.3 0.4 0.5 \
    --max-candidates 20 \
    --log-level DEBUG
```

## Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fasta` | str | Required | Input FASTA file path |
| `--pdb` | str | Required | Input PDB file path |
| `--mode` | str | reduce | Immunogenicity mode: reduce or enhance |
| `--output-dir` | str | results | Output directory for results |
| `--log-level` | str | INFO | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--samples-per-temp` | int | 20 | Number of samples per temperature |
| `--temperatures` | float[] | [0.1, 0.3, 0.5] | Temperature values for ProteinMPNN |
| `--max-candidates` | int | 10 | Maximum candidates for structure prediction |
| `--rmsd-threshold` | float | 2.0 | RMSD threshold for structure filtering |
| `--dockq-threshold` | float | 0.23 | DOCKQ threshold for structure filtering |

### Configuration File

You can also use a JSON configuration file:

```json
{
  "fasta_path": "input/protein.fasta",
  "pdb_path": "input/protein.pdb",
  "mode": "reduce",
  "output_dir": "results",
  "log_level": "INFO",
  "samples_per_temp": 20,
  "temperatures": [0.1, 0.3, 0.5],
  "hla_alleles": [
    "DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01",
    "DRB1*08:01", "DRB1*11:01", "DRB1*13:01", "DRB1*15:01"
  ],
  "max_candidates": 10,
  "rmsd_threshold": 2.0,
  "dockq_threshold": 0.23,
  "neutral_score": 50.0
}
```

## Pipeline Workflow

1. **Input Validation**: Validates FASTA and PDB files
2. **Epitope Prediction**: Identifies CD4+ T-cell epitopes using IEDB API and NetMHCIIpan
3. **Sequence Generation**: Generates mutant sequences using ProteinMPNN
4. **MHC-II Evaluation**: Evaluates binding affinity across HLA alleles
5. **Scoring**: Computes immunogenicity scores based on binding ranks
6. **Structure Prediction**: Predicts structures using AlphaFold3
7. **Filtering**: Applies RMSD and DOCKQ thresholds
8. **Output**: Saves all results to specified directory

## Output Files

The pipeline generates the following output files:

- `epitope_predictions.csv`: Predicted epitopes with scores
- `mutant_sequences.fasta`: Generated mutant sequences
- `mhc_binding_scores.csv`: MHC-II binding affinity scores
- `final_ranked_candidates.csv`: Final ranked candidates with structure predictions
- `config.json`: Configuration used for the run
- `pipeline.log`: Detailed log file

## Scoring System

### Immunogenicity Reduction Mode
- Lower binding ranks (stronger binding) receive lower scores
- Formula: `Score = 100 - ((rank - min_rank) / (max_rank - min_rank)) * 100`

### Immunogenicity Enhancement Mode
- Lower binding ranks (stronger binding) receive higher scores
- Formula: `Score = ((rank - min_rank) / (max_rank - min_rank)) * 100`

## Error Handling

The pipeline includes comprehensive error handling:

- Input file validation
- Tool availability checks
- Timeout handling for long-running processes
- Graceful failure recovery
- Detailed error logging

## Logging

Logs are written to both console and file (`pipeline.log`). Log levels:

- `DEBUG`: Detailed debugging information
- `INFO`: General information about pipeline progress
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors that may cause pipeline failure

## Examples

### Example 1: Basic Immunogenicity Reduction

```bash
python immunogenicity_optimization_pipeline.py \
    --fasta examples/protein.fasta \
    --pdb examples/protein.pdb \
    --mode reduce
```

### Example 2: High-Throughput Enhancement

```bash
python immunogenicity_optimization_pipeline.py \
    --fasta examples/protein.fasta \
    --pdb examples/protein.pdb \
    --mode enhance \
    --samples-per-temp 100 \
    --temperatures 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --max-candidates 50
```

### Example 3: Debug Mode

```bash
python immunogenicity_optimization_pipeline.py \
    --fasta examples/protein.fasta \
    --pdb examples/protein.pdb \
    --mode reduce \
    --log-level DEBUG \
    --output-dir debug_results
```

## Troubleshooting

### Common Issues

1. **Tool Not Found**: Ensure ProteinMPNN, NetMHCIIpan, and AlphaFold3 are installed and in PATH
2. **Memory Issues**: Reduce `samples_per_temp` or `max_candidates` for large proteins
3. **Timeout Errors**: Increase timeout values or reduce batch sizes
4. **File Permissions**: Ensure write permissions for output directory

### Debug Mode

Use `--log-level DEBUG` to get detailed information about pipeline execution.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{immunogenicity_pipeline,
  title={Immunogenicity Optimization Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/immunogenicity-pipeline}
}
```

## Support

For questions and support, please open an issue on GitHub or contact [your-email@domain.com].