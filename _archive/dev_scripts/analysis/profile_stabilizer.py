"""Profile iris stabilizer recovery time."""
import time, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import load_chimeric_data
from iris_stabilizer import IrisStabilizer

subjects = load_chimeric_data('fvc2002_1', min_iris=2)
s0, s1 = subjects[0], subjects[1]
iris0 = s0['iris'][0]
iris1 = s1['iris'][1]  # impostor

for B in [255, 511, 1023]:
    stab = IrisStabilizer(block_size=B)
    commit = stab.enroll(iris0['code'], iris0['mask'])
    
    t0 = time.time()
    result = stab.recover(iris1['code'], iris1['mask'], commit)
    dt = time.time() - t0
    print(f'B={B}: {dt:.3f}s per impostor recover, success={result.success}')
    print(f'  9900 impostor attempts: {dt*9900/60:.0f} min')
    print(f'  36 combos × 9900: {dt*9900*36/3600:.0f} hours')
