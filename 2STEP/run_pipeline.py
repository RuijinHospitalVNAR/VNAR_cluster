#!/usr/bin/env python3
"""
Simple runner script for the Immunogenicity Optimization Pipeline.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from immunogenicity_optimization_pipeline import main

if __name__ == "__main__":
    main()
