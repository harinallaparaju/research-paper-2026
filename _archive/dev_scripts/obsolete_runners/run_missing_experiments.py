#!/usr/bin/env python3
"""
Run ONLY the missing experiments: n_bonus ablation + Arch B/C/D.
These were not completed in the previous run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_remaining_experiments import experiment_nbonus_ablation, experiment_arch_bcd
from datetime import datetime

print("=" * 80)
print(f"MISSING EXPERIMENTS — Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\n>>> Running n_bonus ablation...")
experiment_nbonus_ablation()

print("\n>>> Running Architectures B/C/D...")
experiment_arch_bcd()

print(f"\n>>> ALL DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
