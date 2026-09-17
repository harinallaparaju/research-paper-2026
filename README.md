# D-IBFV: Decentralized Iris-Boosted Fuzzy Vault

**M.Tech Thesis — CS & Information Security, NIT Warangal, 2026**  
**Author:** Nallaparaju V. Suryanarayana Raju
**Supervisor:** Dr. Mulagala Sandhya, Dept. of CSE, NIT Warangal

---

## What This System Does

D-IBFV is a multimodal biometric template protection system that combines
**fingerprint** and **iris** to protect templates against database breaches.

Two original contributions:

1. **IBFV** — Iris-Boosted Fuzzy Vault: iris acts as an *optional* booster
   inside an ECC fuzzy vault. Formally proven to *never* degrade fingerprint-only
   accuracy (Theorem 1 in the thesis).

2. **D-IBFV** — Distributed IBFV: vault is split using Shamir Secret Sharing,
   shares stored on IPFS, share locations recorded on blockchain (Ethereum/Fabric).
   A single server breach reveals zero information about the vault.

---

## Quick Start (5 minutes, no setup needed)

> **Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
# 1. Clone / copy this folder onto the PC
cd "Surya's Project"

# 2. Copy the environment template
cp .env.example .env

# 3. Start all services (IPFS + Ganache + Python environment)
docker-compose up -d

# 4. Open a shell inside the container
docker-compose exec dibfv bash

# 5. You are now inside /app/code — run any experiment (see Section 3 below)
python run_complete_experiments.py
```

Results appear in `code/results/` on your laptop in real time (folder is mounted).

To stop everything:
```bash
docker-compose down
```

---

## Without Docker (manual install on the PC)

If Docker is not available, install manually:

```bash
# Python 3.10 or 3.11 required
python --version

# Install all Python dependencies
pip install -r requirements.txt

# All experiment scripts must be run from the code/ directory
cd "Surya's Project/code"

python run_complete_experiments.py
```

> **IPFS and Ganache** are only needed for the D-IBFV distributed experiments
> (Section 3.4 below). The core biometric experiments (EER, ROC, ablation)
> run without them.

---

## Project Structure

```
Surya's Project/
│
├── README.md                   ← You are here
├── requirements.txt            ← Python packages
├── .env.example                ← Environment variable template
├── Dockerfile                  ← Container definition
├── docker-compose.yml          ← Starts IPFS + Ganache + Python
│
├── code/                       ← ALL source code (run scripts from here)
│   │
│   ├── unimodal/               ← Fingerprint ECC fuzzy vault (core algorithm)
│   │   ├── fingerprint_vault.py    Main vault: quantize → EC map → polynomial → vault
│   │   └── run_baseline.py         Quick baseline EER check
│   │
│   ├── architectures/          ← All 5 vault architectures
│   │   ├── architecture_a.py       AND-fusion (Nandakumar & Jain 2008 style)
│   │   ├── architecture_b.py       Polynomial blinding
│   │   ├── architecture_c.py       Dense chaff seeding
│   │   ├── architecture_d.py       AES-256-GCM vault encryption
│   │   └── architecture_ibfv.py    ★ IBFV — our novel contribution
│   │
│   ├── iris_extraction/        ← Iris biometric pipeline
│   │   ├── segmentation.py         Pupil + iris boundary detection (NIR images)
│   │   ├── normalization.py        Daugman rubber-sheet → 64×512 strip
│   │   ├── encoding.py             Log-Gabor → 131,072-bit iris code
│   │   ├── matching.py             Fractional Hamming Distance (FHD) + rotation
│   │   ├── extract_all.py          Batch extractor for CASIA-Iris-Interval
│   │   └── validate.py             Sanity checks on extracted codes
│   │
│   ├── iris_stabilizer.py      ← Iris fuzzy commitment (bit interleaving + majority vote)
│   │                               Converts noisy iris code → stable 514-bit key
│   │
│   ├── chimeric_db.py          ← Chimeric database loader (FP subject i + Iris subject i)
│   ├── chimeric_db/            ← Pre-generated chimeric JSON files (8 databases × 2 setups)
│   │   └── chimeric_fvc2002_1_min2.json  (and 7 more)
│   │
│   ├── dibfv/                  ← ★ D-IBFV distributed system
│   │   ├── shamir.py               Shamir Secret Sharing over GF(257)
│   │   ├── gf257.py                GF(257) finite field arithmetic
│   │   ├── vault_serial.py         Vault serialization / deserialization
│   │   ├── ipfs_client.py          IPFS share upload / retrieval
│   │   ├── blockchain_client.py    Ethereum (Ganache/Sepolia/Polygon) client
│   │   ├── fabric_client.py        Hyperledger Fabric client
│   │   ├── enrollment.py           Full enrollment pipeline (vault → Shamir → IPFS → chain)
│   │   ├── verification.py         Full verification pipeline (chain → IPFS → Shamir → IBFV)
│   │   ├── revocation.py           Revoke + re-enroll
│   │   ├── deploy_all.py           Deploy VaultRegistry.sol to EVM platforms
│   │   ├── benchmark.py            ★ Latency / throughput benchmarks (Tables 7–13)
│   │   ├── config.py               Platform configuration (reads from .env)
│   │   ├── blockchain/
│   │   │   ├── ethereum/
│   │   │   │   └── VaultRegistry.sol   Solidity smart contract (CID registry)
│   │   │   └── fabric/
│   │   │       └── vault_registry_cc/  Hyperledger Fabric chaincode (Go)
│   │   └── test_*.py               Unit tests (shamir, IPFS, blockchain, end-to-end)
│   │
│   ├── ── EXPERIMENT RUNNERS ──────────────────────────────────────────────
│   │
│   ├── run_complete_experiments.py  ★ MAIN — full EER sweep (Tables 2–4 in paper)
│   ├── run_bootstrap_ci.py          Bootstrap 95% CI on EER/GAR/FAR
│   ├── run_unlinkability_dsys.py    D_sys unlinkability analysis (Table 5)
│   ├── run_entropy_analysis.py      Vault entropy metrics (Table 6)
│   ├── run_bs_fhd_sweep.py          Block-size / FHD parameter sweep (iris tuning)
│   ├── run_minimal_bcd.py           Minimal B/C/D architecture experiments
│   │
│   ├── ── FIGURE GENERATORS ───────────────────────────────────────────────
│   │                   ★ = the one to run. Others are earlier drafts — do not run.
│   │
│   ├── generate_final_figures.py    ★ RUN THIS — all final paper figures
│   ├── generate_blockchain_figures.py  ★ RUN THIS — latency / TPS figures (Tables 7–13)
│   │
│   │   ── Earlier drafts (do NOT run — produces stale/wrong figures) ──
│   ├── generate_paper_figures.py    draft v1 — superseded by generate_final_figures.py
│   ├── generate_sweep_figures.py    draft v2 — superseded
│   ├── generate_updated_figures.py  draft v3 — superseded
│   ├── generate_fig2_fig5.py        one-off draft — superseded
│   │
│   ├── results/                ← All experiment output (CSV, JSON, PDF figures)
│   │
│   ├── unimodal_implementation/ ← OLD reference code (Noob's Project style)
│   │                               Not part of the clean system — kept for reference
│   ├── matlab_plotting/        ← MATLAB ROC/DET plotting scripts
│   ├── published_paper/        ← Reference PDF of base ECC vault paper (Maurya 2025)
│   └── _deprecated/            ← One-off exploration scripts (ignore)
│
├── data/                       ← Biometric data (NOT uploaded to git if large)
│   ├── Iris/
│   │   ├── CASIA-Iris-Interval/    Raw NIR iris images (249 subjects)
│   │   ├── iris_codes/             ★ Pre-extracted 131,072-bit iris codes (USE THIS)
│   │   └── iris_codes_old_v1/      Old extraction pass — DO NOT USE (different params)
│   └── minutiae/
│       ├── 2002/                   Pre-extracted FVC2002 minutiae (DB1, DB2, DB3)
│       └── 2004/                   Pre-extracted FVC2004 minutiae (DB1)
│
├── paper_thesis/               ← M.Tech thesis LaTeX source + PDF
├── paper_ieee/                 ← IEEE conference paper
├── paper_elsevier/             ← Elsevier journal paper
├── paper_extended/             ← Extended version
├── related papers/             ← Reference papers
├── _archive/                   ← Old experiments and dead code (ignore)
└── _checkpoints/               ← Milestone zip snapshots
```

---

## Running Experiments

> All commands assume you are in the `code/` directory:
> ```bash
> cd "Surya's Project/code"
> # or inside the Docker container (already in /app/code)
> ```

### 3.1  Main EER Experiment (Tables 2–4: Unimodal vs Arch A/B/C/D vs IBFV)

This is the primary experiment. Runs the full 864-configuration sweep.
**Takes 2–4 hours on 8 cores.** Results are saved incrementally to CSV.

```bash
python run_complete_experiments.py
```

Output: `results/complete_experiments/eer_results_setup1.csv`
        `results/complete_experiments/eer_results_setup2.csv`

### 3.2  Bootstrap Confidence Intervals

Computes 95% CI on EER/GAR/FAR at best configurations.

```bash
python run_bootstrap_ci.py
```

Output: `results/paper_experiments/bootstrap_ci.csv`

### 3.3  Unlinkability (D_sys Analysis)

Tests ISO/IEC 24745 Property 2. Compares same-factor vs diverse-factor scenarios.

```bash
python run_unlinkability_dsys.py
```

Output: `results/paper_experiments/dsys_results.csv`

### 3.4  D-IBFV Distributed Benchmarks (Tables 7–13: Latency, TPS, Storage)

**Requires IPFS and Ganache running** (either via Docker Compose, or manually).

**Step 1:** Deploy the smart contract to Ganache.
```bash
cd dibfv
python deploy_all.py --ganache
# This prints the deployed contract address — it is saved automatically to .env
```

**Step 2:** Run the full benchmark suite.
```bash
python benchmark.py
```

Output: `dibfv/results/` (CSV files for all tables)

### 3.5  Vault Entropy Analysis

```bash
python run_entropy_analysis.py
```

### 3.6  Iris Block-Size / FHD Sweep

Sweeps block size B_s to find optimal iris key extraction parameters.

```bash
python run_bs_fhd_sweep.py
```

### 3.7  Generate All Paper Figures

After running experiments, generate publication-quality PDF figures.

```bash
# Biometric figures (ROC, EER vs f, ablation heatmap, unlinkability)
python generate_final_figures.py

# Blockchain latency / TPS figures (Tables 7–13)
python generate_blockchain_figures.py
```

Output: figures saved to `../paper_ieee/figures/`

> The other `generate_*.py` files (`generate_paper_figures.py`,
> `generate_sweep_figures.py`, `generate_updated_figures.py`,
> `generate_fig2_fig5.py`) are earlier drafts. **Do not run them** — they
> produce stale figures with outdated parameters.

### 3.8  Quick Unit Tests (D-IBFV Components)

Test Shamir, IPFS, and the end-to-end pipeline without real biometric data.

```bash
cd dibfv
python test_shamir.py      # Shamir SSS — 30,000 round-trip trials
python test_ipfs.py        # IPFS upload/download (needs IPFS running)
python test_blockchain.py  # Smart contract (needs Ganache running)
python test_e2e.py         # Full enrollment → verification → revocation
```

---

## Setting Up IPFS and Ganache Without Docker

If Docker is not an option, install manually:

**IPFS (Kubo):**
```bash
# macOS
brew install ipfs
ipfs init
ipfs daemon &   # runs in background on port 5001

# Ubuntu/Debian
wget https://dist.ipfs.tech/kubo/v0.27.0/kubo_v0.27.0_linux-amd64.tar.gz
tar -xzf kubo_v0.27.0_linux-amd64.tar.gz
sudo mv kubo/ipfs /usr/local/bin/
ipfs init
ipfs daemon &
```

**Ganache:**
```bash
# Requires Node.js >= 16
npm install -g ganache
ganache --deterministic --accounts 10 --networkId 1337 &
```

---

## Setting Up Hyperledger Fabric (Optional — Advanced)

Fabric requires additional setup. This is only needed to reproduce the
Fabric latency numbers in the paper. The biometric results and Ganache
results are independent of Fabric.

```bash
cd code/dibfv/blockchain/fabric

# Download Fabric binaries and Docker images (~2 GB)
chmod +x install-fabric.sh
./install-fabric.sh

# Start the test network
cd fabric-samples/test-network
./network.sh up createChannel -c mychannel -ca

# Deploy the chaincode
./network.sh deployCC \
  -ccn vaultregistry \
  -ccp ../../vault_registry_cc \
  -ccl go

# Run the Fabric benchmark
cd ../../..
python benchmark.py --fabric
```

---

## Data Requirements

The `data/` folder must contain:

| Path | What | Size |
|---|---|---|
| `data/Iris/CASIA-Iris-Interval/` | Raw NIR iris images | ~800 MB |
| `data/Iris/iris_codes/` | Pre-extracted iris codes **(use this one)** | ~500 MB |
| `data/minutiae/2002/` | FVC2002 minutiae (DB1, DB2, DB3) | ~50 MB |
| `data/minutiae/2004/` | FVC2004 minutiae (DB1) | ~15 MB |

> **Important:** there is also a folder `data/Iris/iris_codes_old_v1/` — ignore it.
> It is an earlier extraction pass with different parameters and will give wrong results.
> Always use `data/Iris/iris_codes/`.

**Pre-extracted iris codes** are already in `data/Iris/iris_codes/`.
Use these — no need to re-run iris extraction unless you change the pipeline.

To re-extract iris codes from scratch:
```bash
cd code
python iris_extraction/extract_all.py
```

The **chimeric database JSON files** are already generated and stored in
`code/chimeric_db/`. These pair FVC fingerprint subject i with CASIA iris
subject i. No need to regenerate them.

---

## Key Numbers (Paper Results)

| Metric | Value |
|---|---|
| IBFV EER (DB3, Setup 1) | 1.384% → **49.9% reduction** vs unimodal 2.762% |
| IBFV EER (DB4, Setup 1) | 0.929% → **61.3% reduction** vs unimodal 2.404% |
| Arch B/C/D EER floor | **~19%** (structural, all databases) |
| Theorem 1 zero-violations | **144/144** Setup-2 configurations |
| Shamir integrity (30,000 trials) | **0 bit errors** |
| Ganache enrollment latency | **363 ms** |
| Ganache verification latency | **83 ms** |
| Fabric lookup TPS (4 workers) | **48.1 TPS** |
| Unlinkability D_sys (diverse f) | **≤ 0.03** |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tinyec'`**
→ Run `pip install -r requirements.txt` from the project root.

**`ModuleNotFoundError: No module named 'cv2'`**
→ Run `pip install opencv-python`.

**`Connection refused` on port 5001 (IPFS)**
→ Start IPFS: `ipfs daemon &`  or  `docker-compose up -d ipfs`

**`Connection refused` on port 8545 (Ganache)**
→ Start Ganache: `ganache --deterministic &`  or  `docker-compose up -d ganache`

**`ContractNotDeployed` error**
→ Run `cd code/dibfv && python deploy_all.py --ganache` first.

**Experiments very slow on Windows**
→ Use Docker. The `multiprocessing` module works best on Linux/macOS.
  Inside the container, performance is normal.

**`web3` import works but `ExtraDataToPOAMiddleware` not found**
→ Run `pip install --upgrade web3` (need web3 ≥ 6.0).

---

## Citation

```bibtex
@mastersthesis{nallaparaju2026dibfv,
  author  = {Nallaparaju, V. Suryanarayana Raju},
  title   = {Decentralized Iris-Boosted Fuzzy Vault: Zero-Penalty Multimodal
             Biometric Template Protection with Threshold-Distributed Storage},
  school  = {National Institute of Technology Warangal},
  year    = {2026},
  type    = {{M.Tech} Thesis},
  note    = {Supervisor: Dr. Mulagala Sandhya}
}
```

---

