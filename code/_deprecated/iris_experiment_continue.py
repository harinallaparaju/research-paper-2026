"""
Continuation: Experiments 4-5 + Final/Baseline.

Uses best results from completed Experiments 1-3:
  Masking:     v2_intensity  (d'=1.892)
  Wavelengths: [18, 36]     (d'=1.892)
  Row Band:    (0, 64)      (d'=2.056)
"""
import sys, time, json, gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_pipeline_experiments import (
    load_all_images, presegment_all, evaluate_config, print_result,
    get_preprocessing_experiments, get_bandwidth_experiments,
    PipelineConfig,
)

# ── Best results from Experiments 1-3 ──
BEST_MASK = 'v2_intensity'
BEST_WL = [18, 36]
BEST_ROWS = (0, 64)

def main():
    print("=" * 100)
    print(f"IRIS EXPERIMENT CONTINUATION (Exp 4-5 + Final) — {datetime.now().isoformat()}")
    print("=" * 100)

    # Load and presegment
    print("\n[1/5] Loading and pre-segmenting...")
    subjects = load_all_images()
    cached_subjects = presegment_all(subjects)
    del subjects
    gc.collect()

    results_all = {}

    # ─── Experiment 4: Pre-processing ─────────────────────────────────
    print("\n" + "=" * 100)
    print(f"[2/5] EXPERIMENT 4: PRE-PROCESSING (mask={BEST_MASK}, wl={BEST_WL}, rows={BEST_ROWS})")
    print("=" * 100)

    pp_configs = get_preprocessing_experiments(BEST_MASK, BEST_WL, BEST_ROWS)
    pp_results = {}

    for cfg in pp_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        pp_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
        gc.collect()

    best_pp_name = max(pp_results, key=lambda k: pp_results[k]['d_prime'])
    best_pp = [cfg for cfg in pp_configs if cfg.name == best_pp_name][0]
    print(f"\n  >>> BEST PRE-PROCESSING: {best_pp_name} (d'={pp_results[best_pp_name]['d_prime']:.3f})")
    results_all['preprocessing'] = pp_results
    gc.collect()

    # ─── Experiment 5: Bandwidth ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("[3/5] EXPERIMENT 5: LOG-GABOR BANDWIDTH")
    print("=" * 100)

    bw_configs = get_bandwidth_experiments(
        BEST_MASK, BEST_WL, BEST_ROWS,
        best_pp.clahe_clip, best_pp.blur_sigma, best_pp.frag_threshold_factor
    )
    bw_results = {}

    for cfg in bw_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        bw_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
        gc.collect()

    best_bw_name = max(bw_results, key=lambda k: bw_results[k]['d_prime'])
    best_bw = [cfg.bandwidth for cfg in bw_configs if cfg.name == best_bw_name][0]
    print(f"\n  >>> BEST BANDWIDTH: {best_bw_name} = {best_bw} (d'={bw_results[best_bw_name]['d_prime']:.3f})")
    results_all['bandwidth'] = bw_results
    gc.collect()

    # ─── Final: Best Combined Config ──────────────────────────────────
    print("\n" + "=" * 100)
    print("[4/5] FINAL: BEST COMBINED CONFIGURATION")
    print("=" * 100)

    final_config = PipelineConfig(
        name="BEST_COMBINED",
        eyelid_fn_name=BEST_MASK,
        wavelengths=BEST_WL,
        bandwidth=best_bw,
        code_rows=BEST_ROWS,
        clahe_clip=best_pp.clahe_clip,
        blur_sigma=best_pp.blur_sigma,
        frag_threshold_factor=best_pp.frag_threshold_factor,
    )

    t0 = time.time()
    final_result = evaluate_config(cached_subjects, final_config, verbose=True)
    elapsed = time.time() - t0
    print_result("BEST_COMBINED", final_result, elapsed)
    results_all['final'] = {'config': str(final_config), 'result': final_result}

    # ─── Baseline comparison ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("[5/5] BASELINE COMPARISON")
    print("=" * 100)

    baseline_config = PipelineConfig(
        name="BASELINE",
        eyelid_fn_name='v1_original',
        wavelengths=[18, 36],
        code_rows=(20, 44),
    )

    t0 = time.time()
    baseline_result = evaluate_config(cached_subjects, baseline_config, verbose=True)
    elapsed = time.time() - t0
    print_result("BASELINE", baseline_result, elapsed)
    results_all['baseline'] = {'config': str(baseline_config), 'result': baseline_result}

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  BASELINE:      d' = {baseline_result['d_prime']:.3f}")
    print(f"  BEST COMBINED: d' = {final_result['d_prime']:.3f}")
    improvement = final_result['d_prime'] - baseline_result['d_prime']
    pct = improvement / baseline_result['d_prime'] * 100 if baseline_result['d_prime'] > 0 else 0
    print(f"  IMPROVEMENT:   +{improvement:.3f} ({pct:.1f}%)")
    print(f"\n  Best config: mask={BEST_MASK}, wl={BEST_WL}, rows={BEST_ROWS}")
    print(f"               clahe={best_pp.clahe_clip}, blur={best_pp.blur_sigma}, "
          f"frag={best_pp.frag_threshold_factor}, bw={best_bw}")

    # Save JSON
    out_path = Path(__file__).resolve().parent / "results" / "iris_continuation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
