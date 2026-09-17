"""
Targeted follow-up: explore edges beyond the greedy optimum.
  - Narrower bandwidths (0.20–0.35)
  - Higher frag thresholds (0.35–0.55)
  - Joint frag × bandwidth grid
  - v8_adaptive_thresh with best params
  - Wider MAX_SHIFT
"""
import sys, time, json, gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_pipeline_experiments import (
    load_all_images, presegment_all, evaluate_config, print_result,
    PipelineConfig,
)

# Best from previous run
BEST_MASK = 'v2_intensity'
BEST_WL = [18, 36]
BEST_ROWS = (0, 64)
BEST_CLAHE = 2.0
BEST_BLUR = 1.0


def main():
    print("=" * 100)
    print(f"TARGETED FOLLOW-UP EXPERIMENTS — {datetime.now().isoformat()}")
    print("=" * 100)

    subjects = load_all_images()
    cached = presegment_all(subjects)
    del subjects
    gc.collect()

    results = {}

    # ─── 1. Joint frag × bandwidth grid ──────────────────────────────
    print("\n" + "=" * 100)
    print("[1/3] JOINT FRAG × BANDWIDTH GRID")
    print("=" * 100)

    frags = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    bws = [0.20, 0.25, 0.30, 0.35, 0.40]

    grid_results = {}
    for frag in frags:
        for bw in bws:
            name = f"frag_{frag:.2f}_bw_{bw:.2f}"
            cfg = PipelineConfig(
                name=name,
                eyelid_fn_name=BEST_MASK,
                wavelengths=BEST_WL,
                code_rows=BEST_ROWS,
                clahe_clip=BEST_CLAHE,
                blur_sigma=BEST_BLUR,
                frag_threshold_factor=frag,
                bandwidth=bw,
            )
            t0 = time.time()
            r = evaluate_config(cached, cfg, verbose=True)
            elapsed = time.time() - t0
            grid_results[name] = r
            print_result(name, r, elapsed)
            gc.collect()

    best_grid = max(grid_results, key=lambda k: grid_results[k]['d_prime'])
    print(f"\n  >>> BEST GRID: {best_grid} (d'={grid_results[best_grid]['d_prime']:.3f})")
    results['frag_bw_grid'] = grid_results

    # Parse best frag/bw from winner
    parts = best_grid.split('_')
    best_frag = float(parts[1])
    best_bw = float(parts[3])

    # ─── 2. v8_adaptive_thresh with best params ──────────────────────
    print("\n" + "=" * 100)
    print("[2/3] v8_adaptive_thresh WITH BEST PARAMS")
    print("=" * 100)

    v8_cfg = PipelineConfig(
        name="v8_best_params",
        eyelid_fn_name='v8_adaptive_thresh',
        wavelengths=BEST_WL,
        code_rows=BEST_ROWS,
        clahe_clip=BEST_CLAHE,
        blur_sigma=BEST_BLUR,
        frag_threshold_factor=best_frag,
        bandwidth=best_bw,
    )
    t0 = time.time()
    v8_r = evaluate_config(cached, v8_cfg, verbose=True)
    elapsed = time.time() - t0
    print_result("v8_best_params", v8_r, elapsed)
    results['v8_best_params'] = v8_r

    # Also test v6_flat25 and v7_none with best params
    for mask_name in ['v6_flat25', 'v7_none']:
        cfg = PipelineConfig(
            name=f"{mask_name}_best_params",
            eyelid_fn_name=mask_name,
            wavelengths=BEST_WL,
            code_rows=BEST_ROWS,
            clahe_clip=BEST_CLAHE,
            blur_sigma=BEST_BLUR,
            frag_threshold_factor=best_frag,
            bandwidth=best_bw,
        )
        t0 = time.time()
        r = evaluate_config(cached, cfg, verbose=True)
        elapsed = time.time() - t0
        print_result(cfg.name, r, elapsed)
        results[cfg.name] = r
        gc.collect()

    # ─── 3. MAX_SHIFT sweep ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("[3/3] MAX_SHIFT SWEEP (with best config)")
    print("=" * 100)

    for shift in [8, 12, 16, 20, 24, 32]:
        name = f"shift_{shift}"
        cfg = PipelineConfig(
            name=name,
            eyelid_fn_name=BEST_MASK,
            wavelengths=BEST_WL,
            code_rows=BEST_ROWS,
            clahe_clip=BEST_CLAHE,
            blur_sigma=BEST_BLUR,
            frag_threshold_factor=best_frag,
            bandwidth=best_bw,
        )
        t0 = time.time()
        r = evaluate_config(cached, cfg, max_shift=shift, verbose=True)
        elapsed = time.time() - t0
        print_result(name, r, elapsed)
        results[name] = r
        gc.collect()

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("OVERALL BEST FROM ALL FOLLOW-UP EXPERIMENTS")
    print("=" * 100)

    # Flatten all d' values
    all_dp = {}
    for k, v in results.items():
        if isinstance(v, dict) and 'd_prime' in v:
            all_dp[k] = v['d_prime']
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict) and 'd_prime' in v2:
                    all_dp[k2] = v2['d_prime']

    for name in sorted(all_dp, key=all_dp.get, reverse=True)[:10]:
        print(f"  {name:>40s} | d' = {all_dp[name]:.3f}")

    out = Path(__file__).resolve().parent / "results" / "iris_followup_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
