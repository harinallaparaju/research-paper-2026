#!/usr/bin/env python3
"""
Re-run ONLY Experiment 4 (IBFV + Unimodal Bs sweep) from the full sweep script.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_bs_fhd_sweep import experiment_ibfv_bs_sweep

if __name__ == "__main__":
    experiment_ibfv_bs_sweep()
