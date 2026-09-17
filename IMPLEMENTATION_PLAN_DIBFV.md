# D-IBFV Implementation Plan: Complete Gold Standard

## Overview

This document is the **complete, brutal, thorough implementation plan** for the Decentralized Iris-Boosted Fuzzy Vault (D-IBFV) system. Every component that the extended paper claims must be implemented, benchmarked, and validated.

**What we already have (DONE):**
- IBFV core: vault creation, iris fuzzy commitment, all 5 architectures, 960+ configs evaluated
- Iris pipeline: segmentation, normalization, Log-Gabor encoding, FHD matching
- ECC vault: Brainpool P256r1, Lagrange interpolation, chaff generation
- All biometric experiments: Tables 2–11 in the paper are backed by real data

**What we must implement (THIS PLAN):**
- Shamir Secret Sharing module
- Vault serialization/deserialization
- IPFS integration
- Blockchain smart contracts (3 platforms)
- Benchmarking harness
- Node failure simulation

---

## Phase 1: Shamir Secret Sharing Module (Python)

### 1.1 GF(257) Arithmetic
- **File**: `code/dibfv/gf257.py`
- Implement modular arithmetic over GF(257) — the smallest prime field containing all byte values {0..255}
- Operations: add, subtract, multiply, inverse (Fermat's little theorem: a^{-1} = a^{255} mod 257), divide
- All operations mod 257
- **Test**: verify all 256 byte values are valid elements, verify a * a^{-1} = 1 for all nonzero a

### 1.2 Polynomial Operations over GF(257)
- **File**: `code/dibfv/polynomial_gf257.py`
- Polynomial evaluation: Horner's method over GF(257)
- Lagrange interpolation at x=0 over GF(257) for reconstruction
- Random polynomial generation: degree t-1 with constant term = secret byte
- **Test**: generate random poly, evaluate at n points, reconstruct from any t points, verify exact match

### 1.3 Shamir Split and Reconstruct
- **File**: `code/dibfv/shamir.py`
- `split(secret_bytes: bytes, t: int, n: int) -> List[bytes]`
  - For each byte position: generate random degree-(t-1) polynomial with constant = byte value
  - Evaluate at x = 1, 2, ..., n → one share per party
  - Each share is the same length as the input
  - Fresh randomness per byte position (critical for information-theoretic security)
- `reconstruct(shares: List[Tuple[int, bytes]], t: int) -> bytes`
  - Given ≥ t shares with their indices (1-indexed)
  - For each byte position: Lagrange interpolation at x=0 over GF(257)
  - Return reconstructed bytes
- **Security invariant**: any t-1 shares must be statistically independent of the secret
- **Test**: 
  - Split a known vault, reconstruct from every possible t-subset → all must match exactly
  - Split, corrupt one share, reconstruct from t shares including corrupted → must detect (via SHA-256 trailer)
  - Performance: split 24KB vault should take < 50ms

### 1.4 Unit Tests
- **File**: `code/dibfv/test_shamir.py`
- Test all (t,n) combos from paper: (2,3), (3,5), (4,7), (5,9)
- Property-based test: for random vault bytes and random (t,n), reconstruction from any t-subset is exact
- Confidentiality test: given t-1 shares, empirically verify uniform distribution of each byte
- Edge cases: empty vault, 1-byte vault, vault with byte value 256 (impossible — must reject)

---

## Phase 2: Vault Serialization

### 2.1 Deterministic Serialization
- **File**: `code/dibfv/vault_serial.py`
- `serialize(vault_points, public_key_S, helper_h, key_hash, Bs, K) -> bytes`
  - Header: 4-byte magic (b'IBFV'), 2-byte version (0x0001), 2-byte field ID (0x0100 for BrainpoolP256r1)
  - Vault points: |V| × 64 bytes (x as 32-byte big-endian, y as 32-byte big-endian)
  - Public key S: 64 bytes (S.x, S.y)
  - Helper data h: ceil(N/8) bytes = 16384 bytes
  - Key hash H(k): 32 bytes
  - Parameters: Bs (2 bytes), K (1 byte), reserved (1 byte)
  - Integrity trailer: SHA-256 of all preceding bytes (32 bytes)
- `deserialize(data: bytes) -> tuple`
  - Parse header, verify magic/version
  - Extract vault points, public key, helper data, key hash, parameters
  - Verify SHA-256 integrity trailer
  - Return structured vault object
- **Test**: serialize → deserialize → verify identical; corrupt one byte → integrity check fails

### 2.2 Integration with Existing IBFV Code
- Identify where the current IBFV code stores the vault (likely as Python objects / pickle)
- Add serialize/deserialize calls at the vault save/load points
- Verify: IBFV vault creation → serialize → deserialize → IBFV verification produces identical EER
- **Critical**: zero-bit-error guarantee — the deserialized vault must produce EXACTLY the same biometric results

---

## Phase 3: IPFS Integration

### 3.1 Local IPFS Cluster Setup
- Install IPFS (go-ipfs / kubo): `brew install ipfs`
- Initialize 3 local IPFS nodes (simulating distributed storage)
- Configure as a private cluster with swarm keys
- **Test**: upload a file to node 1, retrieve from node 2 → verify content hash match

### 3.2 IPFS Client Module
- **File**: `code/dibfv/ipfs_client.py`
- `upload_share(share_bytes: bytes) -> str` — returns CID (content identifier)
  - Uses IPFS HTTP API (`/api/v0/add`)
  - Verifies returned CID matches local hash
- `download_share(cid: str) -> bytes` — returns share bytes
  - Uses IPFS HTTP API (`/api/v0/cat`)
  - Verifies downloaded content CID matches requested CID
- `pin(cid: str)` / `unpin(cid: str)` — for persistence management
- **Test**: upload 5 shares, download 3, verify Shamir reconstruction succeeds

### 3.3 Timing Measurements
- Measure upload latency per share (expect ~15-20ms locally)
- Measure download latency per share (expect ~10-15ms locally)
- Measure batch upload (5 shares sequentially vs. parallel)
- Record for Table 7 (Enrollment Latency) and Table 8 (Verification Latency)

---

## Phase 4: Blockchain Smart Contracts

### 4.1 Hyperledger Fabric Chaincode (Go)
- **File**: `code/dibfv/blockchain/fabric/chaincode/vault_registry.go`
- Chaincode functions:
  - `StoreVaultCIDs(uid string, cids []string, t int, n int) -> txid`
  - `LookupVaultCIDs(uid string) -> (cids []string, t int, n int)`
  - `RevokeVault(uid string) -> txid` (marks as revoked, immutable record)
  - `GetHistory(uid string) -> []Event` (enrollment + revocation history)
- Deploy on local Fabric test network (2 orgs, 2 peers each, Raft orderer)
- Benchmark via Hyperledger Caliper 0.5:
  - Enrollment TPS (target: ~85)
  - Verification read TPS (target: ~220)
  - Transaction latency (target: < 500ms)

### 4.2 Ethereum/Polygon Solidity Contract
- **File**: `code/dibfv/blockchain/ethereum/VaultRegistry.sol`
- Solidity 0.8.20 contract:
  ```solidity
  struct VaultRecord {
      string[] cids;
      uint8 threshold;
      uint8 shareCount;
      bool revoked;
      uint256 timestamp;
  }
  mapping(bytes32 => VaultRecord) public vaults;
  
  function storeVaultCIDs(bytes32 uid, string[] calldata cids, uint8 t, uint8 n) external;
  function lookupVaultCIDs(bytes32 uid) external view returns (string[] memory, uint8, uint8);
  function revokeVault(bytes32 uid) external;
  event VaultStored(bytes32 indexed uid, uint256 timestamp);
  event VaultRevoked(bytes32 indexed uid, uint256 timestamp);
  ```
- Deploy to:
  - Sepolia testnet (Ethereum L1, PoS, ~12s block time)
  - Mumbai/Amoy testnet (Polygon PoS, ~2s block time)
- Benchmark via web3.py:
  - Enrollment TX gas cost (target: ~$0.45 ETH L1, ~$0.004 Polygon)
  - Enrollment latency (target: ~14s ETH, ~2.5s Polygon)
  - Verification read latency (free view call, target: < 500ms)

### 4.3 Blockchain Client Module
- **File**: `code/dibfv/blockchain_client.py`
- Unified interface for all 3 platforms:
  ```python
  class BlockchainClient:
      def store_vault(uid, cids, t, n) -> txid
      def lookup_vault(uid) -> (cids, t, n)
      def revoke_vault(uid) -> txid
      def get_history(uid) -> List[Event]
  ```
- Platform-specific implementations:
  - `FabricClient` — uses Fabric SDK for Python (fabric-sdk-py or REST gateway)
  - `EthereumClient` — uses web3.py
  - `PolygonClient` — uses web3.py (same contract, different RPC endpoint)

---

## Phase 5: End-to-End D-IBFV Pipeline

### 5.1 Distributed Enrollment
- **File**: `code/dibfv/enrollment.py`
- Full pipeline: IBFV vault creation → serialize → Shamir split → IPFS upload (×n) → blockchain store CIDs
- Secure erasure of local vault bytes after share distribution
- Timing instrumentation per phase (matches Table 7)

### 5.2 Distributed Verification
- **File**: `code/dibfv/verification.py`
- Full pipeline: blockchain lookup → IPFS download (×t) → Shamir reconstruct → integrity check → IBFV verify
- Secure erasure of reconstructed vault after verification
- Timing instrumentation per phase (matches Table 8)

### 5.3 Revocation
- **File**: `code/dibfv/revocation.py`
- Pipeline: blockchain revoke TX → (optional) IPFS unpin old shares → re-enroll with fresh params

---

## Phase 6: Benchmarking Harness

### 6.1 Vault Integrity Experiment (Table 5)
- For each (t,n) in {(3,5), (4,7), (5,9)}: 
  - 10,000 trials: create vault → serialize → split → reconstruct → verify SHA-256 match
  - Record: bit errors (expect 0), SHA-256 match rate (expect 100%), EER match vs centralized (expect Exact)

### 6.2 Enrollment Latency Experiment (Table 7)
- 100 users × 3 trials × 3 platforms
- Measure per-phase: vault creation, serialization, Shamir splitting, IPFS upload, blockchain TX
- Report mean ± std

### 6.3 Verification Latency Experiment (Table 8)
- 10,000 attempts × 3 platforms
- Measure per-phase: blockchain lookup, IPFS retrieval, Shamir reconstruction, integrity check, IBFV verification
- Report mean ± std

### 6.4 Throughput Experiment (Table 9)
- Concurrent enrollment/verification submissions
- Fabric: Hyperledger Caliper workload driver
- Ethereum/Polygon: parallel web3.py sessions (asyncio)
- Report peak TPS

### 6.5 Storage Cost Experiment (Table 10)
- Measure: IPFS storage per user (n × vault_size), blockchain storage per user (~0.5KB of CIDs)
- Measure: gas costs on Sepolia and Mumbai

### 6.6 Node Failure Simulation (Table 11)
- For (3,5)-Shamir: withhold k shares for k = 0, 1, 2, 3
- Test all C(5,k) failure patterns
- Record: reconstruction success rate per failure count

### 6.7 Shamir Parameter Sensitivity (Table 12)
- For (t,n) in {(2,3), (3,5), (4,7), (5,9)}:
  - Measure: split time, reconstruct time, total IPFS storage
  - Use 24KB vault, 100 trials each

### 6.8 Results Export
- Export all results to CSV files matching paper table numbering
- Generate LaTeX table snippets for direct copy into paper

---

## Phase 7: Integration Testing

### 7.1 End-to-End Test
- Full pipeline test for each platform:
  1. Create chimeric user (FVC fingerprint + CASIA iris)
  2. Run IBFV enrollment
  3. D-IBFV distributed enrollment (serialize → split → IPFS → blockchain)
  4. D-IBFV distributed verification (blockchain → IPFS → reconstruct → IBFV verify)
  5. Verify biometric Accept/Reject matches centralized IBFV exactly
  6. Revoke → re-enroll → verify with new vault → verify old vault rejected

### 7.2 Regression Test
- Run the full 960-config biometric sweep through D-IBFV pipeline
- Verify every single EER value matches the centralized IBFV EER exactly
- This is the GOLD STANDARD validation: decentralization must not change any biometric result

---

## Dependency Graph

```
Phase 1 (Shamir) ──────────┐
                            ├──→ Phase 5 (E2E Pipeline)
Phase 2 (Serialization) ───┤         │
                            │         ├──→ Phase 6 (Benchmarking)
Phase 3 (IPFS) ────────────┤         │         │
                            │         │         └──→ Phase 7 (Integration)
Phase 4 (Blockchain) ──────┘         │
                                     └──→ Paper tables updated with real numbers
```

**Phases 1–2** can be done in parallel and tested independently.  
**Phase 3** (IPFS) is independent of 1–2.  
**Phase 4** (blockchain) is independent of 1–3.  
**Phase 5** depends on all of 1–4.  
**Phase 6** depends on 5.  
**Phase 7** depends on 5–6.

---

## File Structure

```
code/dibfv/
├── gf257.py                    # GF(257) arithmetic
├── polynomial_gf257.py          # Polynomial ops over GF(257)
├── shamir.py                    # Shamir split/reconstruct
├── vault_serial.py              # Vault serialization/deserialization
├── ipfs_client.py               # IPFS upload/download
├── blockchain_client.py         # Unified blockchain interface
├── enrollment.py                # D-IBFV distributed enrollment
├── verification.py              # D-IBFV distributed verification
├── revocation.py                # Revocation + re-enrollment
├── benchmark.py                 # Full benchmarking harness
├── test_shamir.py               # Shamir unit tests
├── test_serial.py               # Serialization tests
├── test_e2e.py                  # End-to-end integration tests
├── blockchain/
│   ├── fabric/
│   │   ├── chaincode/
│   │   │   └── vault_registry.go
│   │   └── network-config.yaml
│   └── ethereum/
│       ├── VaultRegistry.sol
│       ├── deploy.py
│       └── truffle-config.js
└── results/
    ├── vault_integrity.csv
    ├── enrollment_latency.csv
    ├── verification_latency.csv
    ├── throughput.csv
    ├── storage_costs.csv
    ├── node_failure.csv
    └── shamir_params.csv
```

---

## Success Criteria (Gold Standard)

1. **Zero bit errors**: Shamir reconstruct(split(vault)) == vault, 100% of the time, for all (t,n) configurations
2. **Zero biometric accuracy change**: D-IBFV EER == IBFV EER for every single configuration
3. **All paper tables reproducible**: Every number in Tables 5–13 backed by logged experimental data
4. **Three platforms operational**: Fabric, Ethereum L1, Polygon all deployed and benchmarked
5. **Node failure resilience verified**: System tolerates n-t failures gracefully, rejects below threshold
6. **Information-theoretic security**: Empirically verify t-1 shares reveal no information about vault bytes
7. **Audit trail**: Blockchain records all enrollment/revocation events, queryable history

---

## Estimated Complexity

| Phase | New Python LOC | New Go/Solidity LOC | External Dependencies |
|-------|---------------|--------------------|-----------------------|
| 1. Shamir | ~200 | — | — |
| 2. Serialization | ~150 | — | — |
| 3. IPFS | ~100 | — | ipfshttpclient, go-ipfs |
| 4. Blockchain | ~200 | ~150 (Go) + ~80 (Sol) | web3.py, Fabric SDK, Solidity compiler |
| 5. E2E Pipeline | ~250 | — | — |
| 6. Benchmarking | ~300 | — | Hyperledger Caliper |
| 7. Testing | ~200 | — | pytest |
| **Total** | **~1400** | **~230** | |
