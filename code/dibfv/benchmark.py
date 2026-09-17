"""
D-IBFV Benchmarking Harness — Multi-Platform Edition.

Runs all system experiments for the paper's tables:
  - Table 5: Vault Integrity (Shamir round-trip, 10K trials)
  - Table 7: Enrollment Latency (per-phase, per-platform)
  - Table 8: Verification Latency (per-phase, per-platform)
  - Table 9: Throughput (TPS per platform)
  - Table 10: Storage Costs
  - Table 11: Node Failure Resilience
  - Table 12: Shamir Parameter Sensitivity
  - Table 13: Platform Comparison (derived summary)

Platforms tested:
  - Ganache (local EVM)
  - Ethereum Sepolia (public L1 testnet)
  - Polygon Amoy (public L2 testnet)
  - Hyperledger Fabric (local test-network)

All results are exported to CSV files in code/dibfv/results/.
"""

import sys
import os
import csv
import time
import hashlib
import itertools
import numpy as np
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

code_dir = str(Path(__file__).resolve().parent.parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dibfv import shamir, vault_serial
from dibfv.ipfs_client import IPFSClient
from dibfv.blockchain_client import (
    GanacheClient, EthereumClient, PolygonClient, _compile_contract
)
from dibfv import enrollment
from iris_stabilizer import IrisCommitment
from architectures.architecture_ibfv import MultimodalVaultIBFV

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Platform Discovery ───

def _get_available_platforms(bytecode=None):
    """
    Discover and connect to all available blockchain platforms.
    Returns list of (name, client) tuples.
    """
    platforms = []

    # 1. Ganache (local)
    try:
        gc = GanacheClient()
        if gc.w3.is_connected():
            if bytecode is None:
                bytecode = _compile_contract()
            gc.deploy(bytecode)
            platforms.append(("Ganache", gc))
            print("  ✓ Ganache connected")
    except Exception as e:
        print(f"  ✗ Ganache: {e}")

    # 2. Ethereum Sepolia
    try:
        from dibfv.config import SEPOLIA
        if SEPOLIA.private_key:
            ec = EthereumClient(
                rpc_url=SEPOLIA.rpc_url,
                private_key=SEPOLIA.private_key,
            )
            if ec.w3.is_connected():
                bal = ec.w3.eth.get_balance(ec.sender)
                if bal > 0.005 * 1e18:
                    # Check if contract already deployed
                    from dibfv.config import _load_env
                    contract_addr = os.environ.get("SEPOLIA_CONTRACT")
                    if contract_addr:
                        ec.contract_address = contract_addr
                        from web3 import Web3
                        from dibfv.blockchain_client import VAULT_REGISTRY_ABI
                        ec.contract = ec.w3.eth.contract(
                            address=Web3.to_checksum_address(contract_addr),
                            abi=VAULT_REGISTRY_ABI
                        )
                    else:
                        if bytecode is None:
                            bytecode = _compile_contract()
                        ec.deploy(bytecode)
                    platforms.append(("Sepolia", ec))
                    print(f"  ✓ Sepolia connected (balance: {bal/1e18:.4f} ETH)")
                else:
                    print(f"  ✗ Sepolia: insufficient balance ({bal/1e18:.6f} ETH)")
    except Exception as e:
        print(f"  ✗ Sepolia: {e}")

    # 3. Polygon Amoy
    try:
        from dibfv.config import AMOY
        if AMOY.private_key:
            pc = PolygonClient(
                rpc_url=AMOY.rpc_url,
                private_key=AMOY.private_key,
            )
            if pc.w3.is_connected():
                bal = pc.w3.eth.get_balance(pc.sender)
                if bal > 0.05 * 1e18:
                    contract_addr = os.environ.get("AMOY_CONTRACT")
                    if contract_addr:
                        pc.contract_address = contract_addr
                        from web3 import Web3
                        from dibfv.blockchain_client import VAULT_REGISTRY_ABI
                        pc.contract = pc.w3.eth.contract(
                            address=Web3.to_checksum_address(contract_addr),
                            abi=VAULT_REGISTRY_ABI
                        )
                    else:
                        if bytecode is None:
                            bytecode = _compile_contract()
                        pc.deploy(bytecode)
                    platforms.append(("Amoy", pc))
                    print(f"  ✓ Amoy connected (balance: {bal/1e18:.4f} MATIC)")
                else:
                    print(f"  ✗ Amoy: insufficient balance ({bal/1e18:.6f} MATIC)")
    except Exception as e:
        print(f"  ✗ Amoy: {e}")

    # 4. Hyperledger Fabric
    try:
        from dibfv.fabric_client import FabricClient
        fc = FabricClient()
        # Quick health check (avoid underscore-prefixed keys)
        fc.vault_status("healthcheck000")
        platforms.append(("Fabric", fc))
        print("  ✓ Fabric connected")
    except Exception as e:
        print(f"  ✗ Fabric: {e}")

    return platforms, bytecode


def _make_vault(n_points=70, seed=42):
    rng = np.random.RandomState(seed)
    fp = 0xA9FB57DBA1EEA9BC3E660A909D838D726E3BF623D52620282013481D1F6E5377
    pts = [(int.from_bytes(os.urandom(32), 'big'),
            int.from_bytes(os.urandom(32), 'big')) for _ in range(n_points)]
    n_bits = 49152
    ic = IrisCommitment(
        helper_data=rng.randint(0, 2, size=n_bits).astype(np.uint8),
        enrollment_mask=rng.randint(0, 2, size=n_bits).astype(np.uint8),
        block_size=1023, n_blocks=48,
        key_hash=hashlib.sha256(os.urandom(6)).hexdigest(),
        key_length_bits=48,
    )
    return MultimodalVaultIBFV(
        polynomial_degree=10,
        public_key=(int.from_bytes(os.urandom(32), 'big'),
                    int.from_bytes(os.urandom(32), 'big')),
        vault_points=pts, n_genuine_fp=20, n_bonus=4, n_chaff=46,
        field_prime=fp, iris_commitment=ic, block_size=1023, factor=5,
        degree=10, bch_t=0,
    )


def _write_csv(filename, rows, headers):
    path = RESULTS_DIR / filename
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  → {path}")


# ─── Table 5: Vault Integrity ───

def bench_vault_integrity(n_trials=10000):
    """
    For each (t,n): serialize → split → reconstruct → verify SHA-256 match.
    """
    print("\n" + "=" * 60)
    print("TABLE 5: VAULT INTEGRITY")
    print("=" * 60)

    configs = [(3, 5), (4, 7), (5, 9)]
    rows = []

    for t, n in configs:
        vault = _make_vault(70, seed=t * 100 + n)
        blob = vault_serial.serialize(vault)
        original_hash = hashlib.sha256(blob).hexdigest()

        bit_errors = 0
        sha_matches = 0

        for trial in range(n_trials):
            shares = shamir.split(blob, t, n)
            # Use random t-subset for diversity
            subset = list(np.random.choice(len(shares), t, replace=False))
            used = [shares[i] for i in subset]
            restored = shamir.reconstruct(used, t)
            restored_hash = hashlib.sha256(restored).hexdigest()

            if restored_hash == original_hash:
                sha_matches += 1
            # Count bit errors
            for a, b in zip(blob, restored):
                diff = a ^ b
                bit_errors += bin(diff).count('1')

        match_rate = sha_matches / n_trials * 100
        rows.append([f"({t},{n})", n_trials, bit_errors, f"{match_rate:.1f}%", "Exact"])
        print(f"  ({t},{n}): {n_trials} trials, {bit_errors} bit errors, "
              f"SHA-256 match={match_rate:.1f}%")

    _write_csv("vault_integrity.csv",
               rows,
               ["Config", "Trials", "Bit_Errors", "SHA256_Match", "EER_vs_Centralized"])
    print("[DONE] Vault integrity")


# ─── Table 7: Enrollment Latency (Multi-Platform) ───

def bench_enrollment_latency(platforms, ipfs, n_users=100, n_trials=3):
    """
    Enrollment timing per phase, per platform.
    n_users × n_trials enrollments per platform.
    For slow platforms (Fabric), automatically reduces count.
    """
    print("\n" + "=" * 60)
    print("TABLE 7: ENROLLMENT LATENCY (Multi-Platform)")
    print("=" * 60)

    all_rows = []

    for pname, bc in platforms:
        # Fabric is ~2s/tx, so scale down to keep runtime reasonable
        if pname == "Fabric":
            eff_n = min(n_users, 30) * n_trials
        elif pname in ("Sepolia", "Amoy"):
            eff_n = min(n_users, 50) * n_trials
        else:
            eff_n = n_users * n_trials

        print(f"\n  --- {pname} ({eff_n} enrollments) ---")
        timings = {
            'serialize': [], 'split': [], 'ipfs': [],
            'blockchain': [], 'total': []
        }

        for i in range(eff_n):
            vault = _make_vault(70, seed=i + hash(pname) % 10000)
            uid = f"enroll_{pname}_{i:06d}"
            r = enrollment.enroll(uid, vault, t=3, n=5, ipfs=ipfs, blockchain=bc)
            if not r.success:
                print(f"    WARNING: enrollment {i} failed: {r.error}")
                continue
            timings['serialize'].append(r.time_serialize_ms)
            timings['split'].append(r.time_shamir_split_ms)
            timings['ipfs'].append(r.time_ipfs_upload_ms)
            timings['blockchain'].append(r.time_blockchain_tx_ms)
            timings['total'].append(r.time_total_ms)

        for phase in ['serialize', 'split', 'ipfs', 'blockchain', 'total']:
            vals = timings[phase]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                all_rows.append([pname, phase, f"{mean:.1f}", f"{std:.1f}", len(vals)])
                print(f"    {phase:15s}: {mean:.1f} ± {std:.1f} ms (n={len(vals)})")

    _write_csv("enrollment_latency.csv",
               all_rows,
               ["Platform", "Phase", "Mean_ms", "Std_ms", "N"])
    print("[DONE] Enrollment latency")


# ─── Table 8: Verification Latency (Multi-Platform) ───

def bench_verification_latency(platforms, ipfs, n_trials=1000):
    """
    Verification timing per phase, per platform. Pre-enrolls users on each
    platform, then measures: blockchain lookup → IPFS download → Shamir
    reconstruct → integrity check.
    """
    print("\n" + "=" * 60)
    print("TABLE 8: VERIFICATION LATENCY (Multi-Platform)")
    print("=" * 60)

    all_rows = []

    for pname, bc in platforms:
        # Scale trials per platform based on tx speed
        if pname == "Fabric":
            eff_trials = min(n_trials, 100)
        elif pname in ("Sepolia", "Amoy"):
            eff_trials = min(n_trials, 200)
        else:
            eff_trials = n_trials

        print(f"\n  --- {pname} ({eff_trials} verifications) ---")

        # Pre-enroll some users
        n_enroll = min(eff_trials, 50) if pname == "Fabric" else min(eff_trials, 100)
        uids = []
        for i in range(n_enroll):
            vault = _make_vault(70, seed=i + 50000 + hash(pname) % 10000)
            uid = f"verify_{pname}_{i:06d}"
            r = enrollment.enroll(uid, vault, t=3, n=5, ipfs=ipfs, blockchain=bc)
            if r.success:
                uids.append(uid)
        print(f"    Pre-enrolled {len(uids)} users")

        if not uids:
            print(f"    SKIP — no successful enrollments")
            continue

        timings = {
            'blockchain_lookup': [], 'ipfs_download': [],
            'shamir_reconstruct': [], 'deserialize': [], 'total': []
        }

        for i in range(eff_trials):
            uid = uids[i % len(uids)]

            try:
                t0 = time.perf_counter()
                cids, t_val, n_val = bc.lookup_vault(uid)
                t_lookup = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                share_cids = list(zip(range(1, t_val + 1), cids[:t_val]))
                downloaded = ipfs.download_shares(share_cids)
                t_ipfs = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                blob = shamir.reconstruct(downloaded, t_val)
                t_recon = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                _ = vault_serial.deserialize(blob)
                t_deser = (time.perf_counter() - t0) * 1000

                t_total = t_lookup + t_ipfs + t_recon + t_deser

                timings['blockchain_lookup'].append(t_lookup)
                timings['ipfs_download'].append(t_ipfs)
                timings['shamir_reconstruct'].append(t_recon)
                timings['deserialize'].append(t_deser)
                timings['total'].append(t_total)
            except Exception as e:
                if i % 100 == 0:
                    print(f"    verify {i} failed: {e}")
                time.sleep(0.05)  # Brief pause on error to let IPFS recover

        for phase in ['blockchain_lookup', 'ipfs_download',
                      'shamir_reconstruct', 'deserialize', 'total']:
            vals = timings[phase]
            mean = np.mean(vals)
            std = np.std(vals)
            all_rows.append([pname, phase, f"{mean:.2f}", f"{std:.2f}", len(vals)])
            print(f"    {phase:25s}: {mean:.2f} ± {std:.2f} ms")

    _write_csv("verification_latency.csv",
               all_rows,
               ["Platform", "Phase", "Mean_ms", "Std_ms", "N"])
    print("[DONE] Verification latency")


# ─── Table 10: Storage Costs ───

def bench_storage_costs():
    """
    Per-user storage costs for each (t,n) config.
    """
    print("\n" + "=" * 60)
    print("TABLE 10: STORAGE COSTS")
    print("=" * 60)

    vault = _make_vault(70, seed=42)
    blob = vault_serial.serialize(vault)
    vault_kb = len(blob) / 1024

    rows = []
    for t, n in [(2, 3), (3, 5), (4, 7), (5, 9)]:
        shares = shamir.split(blob, t, n)
        share_size = len(shares[0][1])
        total_ipfs_kb = n * share_size / 1024
        blockchain_kb = 0.5  # ~0.5 KB of CIDs + metadata on chain

        rows.append([f"({t},{n})", f"{vault_kb:.1f}",
                     f"{share_size/1024:.1f}", n,
                     f"{total_ipfs_kb:.0f}", f"{blockchain_kb:.1f}",
                     f"{total_ipfs_kb + blockchain_kb:.1f}"])
        print(f"  ({t},{n}): vault={vault_kb:.1f}KB, share={share_size/1024:.1f}KB, "
              f"IPFS={total_ipfs_kb:.0f}KB, chain={blockchain_kb:.1f}KB, "
              f"total={total_ipfs_kb + blockchain_kb:.1f}KB")

    _write_csv("storage_costs.csv",
               rows,
               ["Config", "Vault_KB", "Share_KB", "N_Shares",
                "IPFS_Total_KB", "Blockchain_KB", "Total_KB"])
    print("[DONE] Storage costs")


# ─── Table 11: Node Failure Resilience ───

def bench_node_failure():
    """
    For (3,5)-Shamir: withhold k nodes, test all C(5, 5-k) recovery patterns.
    """
    print("\n" + "=" * 60)
    print("TABLE 11: NODE FAILURE RESILIENCE")
    print("=" * 60)

    vault = _make_vault(70, seed=42)
    blob = vault_serial.serialize(vault)
    original_hash = hashlib.sha256(blob).hexdigest()
    t, n = 3, 5

    shares = shamir.split(blob, t, n)

    rows = []
    for failures in range(n):
        available = n - failures
        patterns = list(itertools.combinations(range(n), available))
        successes = 0

        for pattern in patterns:
            used = [shares[i] for i in pattern]
            try:
                if available >= t:
                    restored = shamir.reconstruct(used, t)
                    if hashlib.sha256(restored).hexdigest() == original_hash:
                        successes += 1
                # If available < t, reconstruction should fail (correctly)
            except Exception:
                pass

        rate = successes / len(patterns) * 100 if patterns else 0
        rows.append([failures, available, len(patterns),
                     f"{rate:.0f}%", successes, len(patterns) - successes])
        print(f"  {failures} failures: {available} available, "
              f"{len(patterns)} patterns, {rate:.0f}% success")

    _write_csv("node_failure.csv",
               rows,
               ["Failures", "Available", "Patterns",
                "Recon_Success_Rate", "Successes", "Fails"])
    print("[DONE] Node failure resilience")


# ─── Table 12: Shamir Parameter Sensitivity ───

def bench_shamir_params(n_trials=100):
    """
    Split/reconstruct timing and storage for each (t,n) config.
    """
    print("\n" + "=" * 60)
    print("TABLE 12: SHAMIR PARAMETER SENSITIVITY")
    print("=" * 60)

    vault = _make_vault(70, seed=42)
    blob = vault_serial.serialize(vault)

    rows = []
    for t, n in [(2, 3), (3, 5), (4, 7), (5, 9)]:
        fault_tol = n - t
        shares = shamir.split(blob, t, n)
        share_kb = len(shares[0][1]) / 1024
        total_kb = n * share_kb

        # Benchmark split
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = shamir.split(blob, t, n)
        split_ms = (time.perf_counter() - t0) / n_trials * 1000

        # Benchmark reconstruct
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = shamir.reconstruct(shares[:t], t)
        recon_ms = (time.perf_counter() - t0) / n_trials * 1000

        rows.append([f"({t},{n})", fault_tol,
                     f"{split_ms:.1f}", f"{recon_ms:.1f}",
                     f"{total_kb:.0f}"])
        print(f"  ({t},{n}): fault_tol={fault_tol}, split={split_ms:.1f}ms, "
              f"recon={recon_ms:.1f}ms, storage={total_kb:.0f}KB")

    _write_csv("shamir_params.csv",
               rows,
               ["Config", "Fault_Tolerance", "Split_ms", "Recon_ms", "Storage_KB"])
    print("[DONE] Shamir parameter sensitivity")


# ─── Table 9: Throughput (TPS per Platform) ───

def bench_throughput(platforms, ipfs, n_ops=50):
    """
    Measure transactions per second for store and lookup on each platform.
    Runs n_ops sequential operations and computes TPS.
    """
    print("\n" + "=" * 60)
    print("TABLE 9: THROUGHPUT (TPS per Platform)")
    print("=" * 60)

    rows = []

    for pname, bc in platforms:
        # Scale ops per platform
        if pname == "Fabric":
            eff_ops = min(n_ops, 20)
        elif pname in ("Sepolia", "Amoy"):
            eff_ops = min(n_ops, 30)
        else:
            eff_ops = n_ops

        print(f"\n  --- {pname} ({eff_ops} ops) ---")

        # Pre-create vaults and shares
        vault = _make_vault(70, seed=99)
        blob = vault_serial.serialize(vault)
        shares = shamir.split(blob, 3, 5)

        # Measure STORE throughput
        store_times = []
        for i in range(eff_ops):
            uid = f"tps_store_{pname}_{i:06d}"
            # Upload shares to IPFS
            cid_map = ipfs.upload_shares(shares)
            cids = [cid for _, cid in cid_map]

            t0 = time.perf_counter()
            try:
                bc.store_vault(uid, cids, 3, 5)
                store_times.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"    store {i} failed: {e}")

        # Measure LOOKUP throughput
        lookup_times = []
        for i in range(eff_ops):
            uid = f"tps_store_{pname}_{i:06d}"
            t0 = time.perf_counter()
            try:
                bc.lookup_vault(uid)
                lookup_times.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"    lookup {i} failed: {e}")

        # Measure REVOKE throughput
        revoke_times = []
        for i in range(min(eff_ops, 20)):  # Fewer revocations
            uid = f"tps_store_{pname}_{i:06d}"
            t0 = time.perf_counter()
            try:
                bc.revoke_vault(uid)
                revoke_times.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"    revoke {i} failed: {e}")

        def _tps(times):
            if not times:
                return 0.0, 0.0
            total = sum(times)
            return len(times) / total if total > 0 else 0.0, np.mean(times) * 1000

        store_tps, store_avg = _tps(store_times)
        lookup_tps, lookup_avg = _tps(lookup_times)
        revoke_tps, revoke_avg = _tps(revoke_times)

        rows.append([pname, "store", f"{store_tps:.1f}", f"{store_avg:.1f}",
                     len(store_times)])
        rows.append([pname, "lookup", f"{lookup_tps:.1f}", f"{lookup_avg:.1f}",
                     len(lookup_times)])
        rows.append([pname, "revoke", f"{revoke_tps:.1f}", f"{revoke_avg:.1f}",
                     len(revoke_times)])

        print(f"    store:  {store_tps:.1f} TPS  (avg {store_avg:.1f} ms, n={len(store_times)})")
        print(f"    lookup: {lookup_tps:.1f} TPS  (avg {lookup_avg:.1f} ms, n={len(lookup_times)})")
        print(f"    revoke: {revoke_tps:.1f} TPS  (avg {revoke_avg:.1f} ms, n={len(revoke_times)})")

    _write_csv("throughput.csv",
               rows,
               ["Platform", "Operation", "TPS", "Avg_ms", "N"])
    print("[DONE] Throughput")


# ─── Table 9a: Concurrent Throughput (TPS vs Workers) ───

def bench_concurrent_throughput(platforms, ipfs, max_workers=16):
    """
    Measure true concurrent TPS by running N parallel workers.
    Tests worker counts: 1, 2, 4, 8, 16.
    Each worker submits store or lookup operations simultaneously.
    """
    print("\n" + "=" * 60)
    print("TABLE 9a: CONCURRENT THROUGHPUT (TPS vs Workers)")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    worker_counts = [w for w in [1, 2, 4, 8, 16] if w <= max_workers]
    rows = []

    for pname, bc in platforms:
        # Pre-create vault data and pre-enroll some users for lookups
        vault = _make_vault(70, seed=99)
        blob = vault_serial.serialize(vault)
        shares_template = shamir.split(blob, 3, 5)

        # Pre-enroll users for lookup tests
        n_pre = 50
        pre_uids = []
        for i in range(n_pre):
            uid = f"conc_pre_{pname}_{i:06d}"
            cid_map = ipfs.upload_shares(shares_template)
            cids = [cid for _, cid in cid_map]
            try:
                bc.store_vault(uid, cids, 3, 5)
                pre_uids.append(uid)
            except Exception:
                pass

        if not pre_uids:
            print(f"  {pname}: SKIP — no successful pre-enrollments")
            continue

        for n_workers in worker_counts:
            # Ops per worker — enough to get stable timing
            ops_per_worker = 10 if pname != "Fabric" else 5
            total_ops = n_workers * ops_per_worker

            # ── STORE ──
            def _store_worker(worker_id):
                times = []
                for j in range(ops_per_worker):
                    uid = f"conc_s_{pname}_{worker_id}_{j}"
                    cid_map = ipfs.upload_shares(shares_template)
                    cids = [cid for _, cid in cid_map]
                    t0 = time.perf_counter()
                    try:
                        bc.store_vault(uid, cids, 3, 5)
                        times.append(time.perf_counter() - t0)
                    except Exception:
                        pass
                return times

            t_wall_start = time.perf_counter()
            all_store_times = []
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_store_worker, w) for w in range(n_workers)]
                for f in as_completed(futures):
                    all_store_times.extend(f.result())
            t_wall_store = time.perf_counter() - t_wall_start

            store_tps = len(all_store_times) / t_wall_store if t_wall_store > 0 else 0
            store_avg = np.mean(all_store_times) * 1000 if all_store_times else 0

            # ── LOOKUP ──
            def _lookup_worker(worker_id):
                times = []
                for j in range(ops_per_worker):
                    uid = pre_uids[j % len(pre_uids)]
                    t0 = time.perf_counter()
                    try:
                        bc.lookup_vault(uid)
                        times.append(time.perf_counter() - t0)
                    except Exception:
                        pass
                return times

            t_wall_start = time.perf_counter()
            all_lookup_times = []
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_lookup_worker, w) for w in range(n_workers)]
                for f in as_completed(futures):
                    all_lookup_times.extend(f.result())
            t_wall_lookup = time.perf_counter() - t_wall_start

            lookup_tps = len(all_lookup_times) / t_wall_lookup if t_wall_lookup > 0 else 0
            lookup_avg = np.mean(all_lookup_times) * 1000 if all_lookup_times else 0

            rows.append([pname, n_workers,
                         f"{store_tps:.1f}", f"{store_avg:.1f}", len(all_store_times),
                         f"{lookup_tps:.1f}", f"{lookup_avg:.1f}", len(all_lookup_times)])
            print(f"  {pname} w={n_workers:2d}: "
                  f"store={store_tps:.1f} TPS ({len(all_store_times)} ops), "
                  f"lookup={lookup_tps:.1f} TPS ({len(all_lookup_times)} ops)")

    _write_csv("concurrent_throughput.csv",
               rows,
               ["Platform", "Workers", "Store_TPS", "Store_Avg_ms", "Store_N",
                "Lookup_TPS", "Lookup_Avg_ms", "Lookup_N"])
    print("[DONE] Concurrent throughput")


# ─── Scalability: Verification Latency vs Enrolled Users ───

def bench_scalability(platforms, ipfs, user_counts=None):
    """
    Measure verification latency at different enrollment scales.
    Tests: 100, 500, 1000, 5000 enrolled users.
    Measures 50 verifications at each scale.
    """
    if user_counts is None:
        user_counts = [100, 500, 1000, 5000]

    print("\n" + "=" * 60)
    print("SCALABILITY: VERIFICATION LATENCY vs ENROLLED USERS")
    print("=" * 60)

    rows = []

    for pname, bc in platforms:
        vault = _make_vault(70, seed=42)
        blob = vault_serial.serialize(vault)
        shares = shamir.split(blob, 3, 5)

        # Fabric is much slower for writes — cap enrollment
        if pname == "Fabric":
            eff_counts = [c for c in user_counts if c <= 500]
        else:
            eff_counts = user_counts

        enrolled_uids = []
        for target in eff_counts:
            # Enroll up to target
            while len(enrolled_uids) < target:
                i = len(enrolled_uids)
                uid = f"scale_{pname}_{i:06d}"
                cid_map = ipfs.upload_shares(shares)
                cids = [cid for _, cid in cid_map]
                try:
                    bc.store_vault(uid, cids, 3, 5)
                    enrolled_uids.append(uid)
                except Exception as e:
                    if i % 100 == 0:
                        print(f"    enroll {i} failed: {e}")
                    break

            actual = len(enrolled_uids)
            if actual < target * 0.9:
                print(f"  {pname} @ {target}: only {actual} enrolled, skipping")
                continue

            # Measure 50 verifications
            n_verify = 50
            latencies = []
            for j in range(n_verify):
                uid = enrolled_uids[j % actual]
                try:
                    t0 = time.perf_counter()
                    cids_ret, t_val, n_val = bc.lookup_vault(uid)
                    share_cids = list(zip(range(1, t_val + 1), cids_ret[:t_val]))
                    downloaded = ipfs.download_shares(share_cids)
                    _ = shamir.reconstruct(downloaded, t_val)
                    latencies.append((time.perf_counter() - t0) * 1000)
                except Exception:
                    pass

            if latencies:
                mean = np.mean(latencies)
                std = np.std(latencies)
                p50 = np.percentile(latencies, 50)
                p95 = np.percentile(latencies, 95)
                rows.append([pname, actual, f"{mean:.1f}", f"{std:.1f}",
                             f"{p50:.1f}", f"{p95:.1f}", len(latencies)])
                print(f"  {pname} @ {actual:5d} users: "
                      f"mean={mean:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms")

    _write_csv("scalability.csv",
               rows,
               ["Platform", "Enrolled_Users", "Mean_ms", "Std_ms",
                "P50_ms", "P95_ms", "N_Verifications"])
    print("[DONE] Scalability")


# ─── Multi-Run Median Enrollment ───

def bench_enrollment_multirun(platforms, ipfs, n_runs=5, n_per_run=30):
    """
    Run enrollment n_runs times and report median/IQR for stable numbers.
    """
    print("\n" + "=" * 60)
    print("ENROLLMENT LATENCY — MULTI-RUN MEDIAN (n_runs={})".format(n_runs))
    print("=" * 60)

    rows = []

    for pname, bc in platforms:
        run_totals = []
        run_bc_times = []

        for run in range(n_runs):
            totals = []
            bc_times = []
            for i in range(n_per_run):
                vault = _make_vault(70, seed=run * 10000 + i + hash(pname) % 10000)
                uid = f"mrun_{pname}_{run}_{i:04d}"
                r = enrollment.enroll(uid, vault, t=3, n=5, ipfs=ipfs, blockchain=bc)
                if r.success:
                    totals.append(r.time_total_ms)
                    bc_times.append(r.time_blockchain_tx_ms)
            if totals:
                run_totals.append(np.median(totals))
                run_bc_times.append(np.median(bc_times))
            print(f"    {pname} run {run+1}/{n_runs}: "
                  f"median_total={np.median(totals):.0f}ms, n={len(totals)}")

        if run_totals:
            med = np.median(run_totals)
            iqr = np.percentile(run_totals, 75) - np.percentile(run_totals, 25)
            bc_med = np.median(run_bc_times)
            bc_iqr = np.percentile(run_bc_times, 75) - np.percentile(run_bc_times, 25)
            rows.append([pname, n_runs, n_per_run,
                         f"{med:.1f}", f"{iqr:.1f}",
                         f"{bc_med:.1f}", f"{bc_iqr:.1f}"])
            print(f"  {pname}: median_total={med:.1f}ms (IQR={iqr:.1f}), "
                  f"median_bc={bc_med:.1f}ms (IQR={bc_iqr:.1f})")

    _write_csv("enrollment_multirun.csv",
               rows,
               ["Platform", "N_Runs", "N_Per_Run",
                "Median_Total_ms", "IQR_Total_ms",
                "Median_BC_ms", "IQR_BC_ms"])
    print("[DONE] Enrollment multi-run")


# ─── Table 13: Platform Comparison (Derived Summary) ───

def bench_platform_comparison():
    """
    Read all per-platform CSVs and produce a summary comparison table.
    This is a derived table — it reads from existing benchmark results.
    """
    print("\n" + "=" * 60)
    print("TABLE 13: PLATFORM COMPARISON (Derived)")
    print("=" * 60)

    rows = []

    # Read enrollment latency
    enroll_path = RESULTS_DIR / "enrollment_latency.csv"
    verify_path = RESULTS_DIR / "verification_latency.csv"
    throughput_path = RESULTS_DIR / "throughput.csv"

    enroll_data = {}
    if enroll_path.exists():
        with open(enroll_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Phase"] == "total":
                    enroll_data[row["Platform"]] = (row["Mean_ms"], row["Std_ms"])

    verify_data = {}
    if verify_path.exists():
        with open(verify_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Phase"] == "total":
                    verify_data[row["Platform"]] = (row["Mean_ms"], row["Std_ms"])

    throughput_data = {}
    if throughput_path.exists():
        with open(throughput_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Platform"] not in throughput_data:
                    throughput_data[row["Platform"]] = {}
                throughput_data[row["Platform"]][row["Operation"]] = row["TPS"]

    all_platforms = sorted(set(
        list(enroll_data.keys()) + list(verify_data.keys()) +
        list(throughput_data.keys())
    ))

    for p in all_platforms:
        e_mean, e_std = enroll_data.get(p, ("N/A", "N/A"))
        v_mean, v_std = verify_data.get(p, ("N/A", "N/A"))
        tp = throughput_data.get(p, {})
        store_tps = tp.get("store", "N/A")
        lookup_tps = tp.get("lookup", "N/A")

        rows.append([p, f"{e_mean}±{e_std}", f"{v_mean}±{v_std}",
                     store_tps, lookup_tps])
        print(f"  {p:15s}: enroll={e_mean}±{e_std}ms, "
              f"verify={v_mean}±{v_std}ms, "
              f"store={store_tps} TPS, lookup={lookup_tps} TPS")

    _write_csv("platform_comparison.csv",
               rows,
               ["Platform", "Enrollment_ms", "Verification_ms",
                "Store_TPS", "Lookup_TPS"])
    print("[DONE] Platform comparison")


# ─── Main ───

if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("D-IBFV BENCHMARKING HARNESS — MULTI-PLATFORM")
    print("=" * 70)

    # Phase 1: Offline benchmarks (no network needed)
    bench_vault_integrity(n_trials=10000)
    bench_shamir_params(n_trials=200)
    bench_storage_costs()
    bench_node_failure()

    # Phase 2: Discover platforms
    print("\n" + "=" * 60)
    print("PLATFORM DISCOVERY")
    print("=" * 60)
    ipfs = IPFSClient()
    if not ipfs.is_online():
        print("ERROR: IPFS daemon not running. Start with 'ipfs daemon &'")
        sys.exit(1)
    print("  ✓ IPFS online")

    platforms, bytecode = _get_available_platforms()

    if not platforms:
        print("ERROR: No blockchain platforms available.")
        sys.exit(1)

    print(f"\n  Active platforms: {[p[0] for p in platforms]}")

    # Phase 3: Multi-platform benchmarks
    # Fabric is slower (~2s/tx), so we use per-platform scaling
    bench_enrollment_latency(platforms, ipfs, n_users=100, n_trials=3)
    bench_verification_latency(platforms, ipfs, n_trials=1000)
    bench_throughput(platforms, ipfs, n_ops=50)

    # Phase 4: New benchmarks
    bench_concurrent_throughput(platforms, ipfs, max_workers=16)
    bench_scalability(platforms, ipfs, user_counts=[100, 500, 1000, 5000])
    bench_enrollment_multirun(platforms, ipfs, n_runs=5, n_per_run=30)

    # Phase 5: Derived summary
    bench_platform_comparison()

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("Results saved to:", RESULTS_DIR)
    print("=" * 70)
