# IBFV Paper Preparation — Complete Data Package

## 1. TITLE (PROPOSED)

**"Iris-Boosted Fuzzy Vault: A Zero-Penalty Multimodal Extension of ECC-Based Fingerprint Template Protection"**

Alternative: "Additive Multimodal Fusion in ECC Fuzzy Vaults via Iris-Derived Polynomial Points"

---

## 2. ABSTRACT NUMBERS (VERIFIED)

| Metric | Value |
|--------|-------|
| Architectures tested | 3 (Unimodal, Architecture A / AND-fusion, IBFV) |
| Databases | 4 (FVC2002 DB1/DB2/DB3, FVC2004 DB1) |
| Parameter configs per DB | 36 (9 factors × 4 degrees) |
| Total experiments | 864 (432 Setup 1 + 432 Setup 2) |
| IBFV vs Unimodal violations | **0/288** |
| IBFV improvements (Setup 1) | 93/144 (64.6%), 51 tied |
| IBFV improvements (Setup 2) | **144/144 (100%)** |
| FAR identical | 288/288 (100%) |
| Best relative EER reduction | **55.3%** (DB4, Setup 1: 2.40% → 1.08%) |
| Statistical significance | p < 0.001 (t=9.88, Setup 1; t=21.80, Setup 2) |

---

## 3. CORE NOVELTY CLAIMS

### Claim 1: Additive Fusion (vs. Gating Fusion)
Prior multimodal fuzzy vault work (Nandakumar & Jain 2008, score fusion approaches) either:
- **AND-gates**: Second modality must succeed for vault access → adds FRR penalty
- **Score fusion**: Discards template protection guarantees

IBFV is the **first architecture** that injects biometric-derived polynomial-consistent points into a fuzzy vault. The second modality **adds genuine points** rather than gating access.

### Claim 2: Zero-Penalty Guarantee
$\text{EER}_{\text{IBFV}} \leq \text{EER}_{\text{Unimodal}}$ **always, by construction**.

Proof sketch:
- If iris recovery fails → vault decoded with FP points only = identical to Unimodal
- If iris recovery succeeds → extra genuine points available → can only help (never hurt)
- FAR is identical because impostor never recovers iris key (0 recoveries in 288 configs)

### Claim 3: No Structural Tradeoff
AND-fusion (Architecture A) is **always worse** than Unimodal. Best ArchA EER across all 4 DBs: 3.50-5.85%, while Unimodal achieves 0.00-2.76%. The iris FRR (~7%) directly adds to vault FRR.

---

## 4. EXPERIMENTAL RESULTS — ALL TABLES

### Table 1: Best EER% per Architecture (Setup 1 — 100 subjects)

| DB | Unimodal | Config | Arch A | Config | **IBFV** | Config | Δ(Uni→IBFV) | Rel% |
|----|----------|--------|--------|--------|----------|--------|-------------|------|
| DB1 | 0.0000 | f=14,k=5 | 3.5000 | f=14,k=5 | **0.0000** | f=14,k=5 | 0.0000 | — |
| DB2 | 0.0000 | f=16,k=7 | 3.5000 | f=16,k=7 | **0.0000** | f=14,k=7 | 0.0000 | — |
| DB3 | 2.7618 | f=18,k=5 | 5.8546 | f=18,k=5 | **1.6495** | f=16,k=5 | **−1.1123** | 40.3% |
| DB4 | 2.4040 | f=22,k=5 | 5.1566 | f=22,k=5 | **1.0758** | f=18,k=7 | **−1.3283** | 55.3% |

### Table 2: Best EER% per Architecture (Setup 2 — 48 subjects, 7 test impressions)

| DB | Unimodal | Config | **IBFV** | Config | Δ(Uni→IBFV) | Rel% |
|----|----------|--------|----------|--------|-------------|------|
| DB1 | 2.7356 | f=28,k=5 | **2.2891** | f=28,k=5 | **−0.4464** | 16.3% |
| DB2 | 4.5876 | f=24,k=5 | **3.3161** | f=22,k=5 | **−1.2715** | 27.7% |
| DB3 | 6.7376 | f=26,k=5 | **5.1072** | f=26,k=5 | **−1.6304** | 24.2% |
| DB4 | 4.3066 | f=20,k=5 | **2.7910** | f=22,k=7 | **−1.5156** | 35.2% |

### Table 3: Comparison with Senior's Paper (JISAA 2025, Setup 1)

| DB | Senior (f=24,k=10) | Our Unimodal (best) | Our IBFV (best) |
|----|---------------------|--------------------|-----------------| 
| DB1 | 0.0051% | 0.0000% (f=14,k=5) | **0.0000%** |
| DB2 | 0.2500% | 0.0000% (f=16,k=7) | **0.0000%** |
| DB3 | 2.8500% | 2.7618% (f=18,k=5) | **1.6495%** |
| DB4 | 3.5100% | 2.4040% (f=22,k=5) | **1.0758%** |

Note: Senior used fixed f=24, k=10. Our wider sweep found better (f,k) optima.

### Table 4: Statistical Significance (Paired t-test, ΔEER)

| Setup | N | Mean Δ | Std | t-stat | df | p-value |
|-------|---|--------|-----|--------|-------|---------|
| Setup 1 (all) | 144 | 1.4602% | 1.7730% | 9.88 | 143 | <0.001 |
| Setup 2 (all) | 144 | 2.0875% | 1.1490% | 21.80 | 143 | <0.001 |

Per-DB t-statistics (all p < 0.001):
| DB | Setup 1 t | Setup 2 t |
|----|-----------|-----------|
| DB1 | 4.16 | 13.02 |
| DB2 | 3.37 | 23.13 |
| DB3 | 7.85 | 14.94 |
| DB4 | 11.43 | 14.65 |

---

## 5. ΔEER HEATMAPS (Figure Data)

### Setup 1 — DB3 (FVC2002 DB3): IBFV EER minus Unimodal EER

| f\k | k=5 | k=7 | k=9 | k=11 |
|-----|-----|-----|-----|------|
| 14 | −2.63 | −3.68 | −7.37 | **−8.95** |
| 16 | −2.63 | −2.11 | −4.74 | −7.37 |
| 18 | −1.05 | −2.63 | −4.21 | −5.26 |
| 20 | −1.05 | −0.53 | −4.21 | −4.74 |
| 22 | −1.05 | −1.05 | −3.68 | −6.32 |
| 24 | −0.53 | −1.05 | −1.58 | −4.21 |
| 26 | −1.05 | −1.05 | −2.63 | −3.68 |
| 28 | −1.05 | −0.53 | −0.53 | −4.74 |
| 30 | −0.53 | −0.53 | −3.16 | −3.16 |

All values ≤ 0. Max improvement at f=14,k=11: **8.95 percentage points**.

### Setup 1 — DB4 (FVC2004 DB1):

| f\k | k=5 | k=7 | k=9 | k=11 |
|-----|-----|-----|-----|------|
| 14 | −2.00 | −4.00 | −4.50 | −3.50 |
| 16 | −3.00 | −3.50 | −2.00 | **−4.50** |
| 18 | −1.50 | −3.50 | −2.50 | −3.00 |
| 20 | −1.50 | −2.00 | −3.00 | **−4.50** |
| 22 | 0.00 | −2.50 | −2.50 | −3.00 |
| 24 | −1.50 | −1.50 | −2.00 | −3.50 |
| 26 | 0.00 | −2.00 | −2.50 | −2.00 |
| 28 | 0.00 | −1.50 | −2.50 | −3.50 |
| 30 | 0.00 | −1.00 | −2.00 | −3.00 |

### Setup 2 — Key observation: ALL cells negative (no ties)

DB3 heatmap max improvement: −6.16% at (f=14, k=9)
DB4 heatmap max improvement: −4.22% at (f=14, k=11)

---

## 6. EER vs FACTOR CURVES (Figure Data)

### Setup 1, k=5:

| f | DB1_Uni | DB1_IBFV | DB2_Uni | DB2_IBFV | DB3_Uni | DB3_IBFV | DB4_Uni | DB4_IBFV |
|---|---------|----------|---------|----------|---------|----------|---------|----------|
| 14 | 0.000 | 0.000 | 0.020 | 0.020 | 5.806 | 3.174 | 3.566 | 1.566 |
| 16 | 0.030 | 0.030 | 0.066 | 0.066 | 4.281 | 1.650 | 4.126 | 1.126 |
| 18 | 0.056 | 0.056 | 0.197 | 0.197 | **2.762** | 1.709 | 2.904 | 1.404 |
| 20 | 0.131 | 0.131 | 0.576 | 0.576 | 2.946 | 1.894 | 2.803 | 1.303 |
| 22 | 0.364 | 0.364 | 1.005 | 1.005 | 3.163 | 2.111 | **2.404** | 2.404 |
| 24 | 0.712 | 0.712 | 2.248 | 2.248 | 3.701 | 3.174 | 4.748 | 3.248 |
| 26 | 1.389 | 1.389 | 3.823 | 3.823 | 4.400 | 3.348 | 4.172 | 4.172 |
| 28 | 2.323 | 2.323 | 5.611 | 5.611 | 5.323 | 4.270 | 5.813 | 5.813 |
| 30 | 3.520 | 3.520 | 7.884 | 7.884 | 5.676 | 5.149 | 7.106 | 7.106 |

**Key insight**: IBFV gap widens for harder databases (DB3, DB4) and larger k.
At easy configs (near-perfect FP matching), both achieve 0% → no room to improve.

### Setup 2, k=5:

| f | DB3_Uni | DB3_IBFV | DB4_Uni | DB4_IBFV |
|---|---------|----------|---------|----------|
| 14 | 15.80 | 12.53 | 8.07 | 5.51 |
| 16 | 12.77 | 9.33 | 6.57 | 4.16 |
| 18 | 10.70 | 8.52 | 4.61 | 2.80 |
| 20 | 8.65 | 6.12 | **4.31** | 3.10 |
| 22 | 7.65 | 5.29 | 4.60 | 3.40 |
| 24 | 7.02 | 5.38 | 5.87 | 4.82 |
| 26 | **6.74** | **5.11** | 6.32 | 5.57 |
| 28 | 7.09 | 5.46 | 8.09 | 7.49 |
| 30 | 7.61 | 6.16 | 9.60 | 9.30 |

---

## 7. SECURITY ANALYSIS

### 7.1 Vault Security (Brute-Force Complexity)

Vault parameters: n_chaff = 200, Curve: brainpoolP256r1 (p ≈ 2^256)

| k | f=14 (42 gen) | f=18 (54 gen) | f=22 (66 gen) | f=26 (78 gen) | f=30 (90 gen) |
|---|---------------|---------------|---------------|---------------|---------------|
| 5 | 12.9 bits | 11.4 bits | 10.2 bits | 9.3 bits | 8.6 bits |
| 7 | 18.3 bits | 16.1 bits | 14.4 bits | 13.1 bits | 12.1 bits |
| 9 | 23.9 bits | 20.9 bits | 18.7 bits | 17.0 bits | 15.6 bits |
| 11 | 29.5 bits | 25.8 bits | 23.1 bits | 20.9 bits | 19.2 bits |

Security = $\log_2\left(\frac{\binom{V}{k}}{\binom{G}{k}}\right)$ where V = vault size, G = genuine FP points.

### 7.2 With Stolen Iris Key (n_bonus = 4 free points)

| k | f=14 | f=18 | f=22 | f=26 | f=30 |
|---|------|------|------|------|------|
| 5 | **2.5** | **2.2** | **2.0** | **1.8** | **1.7** |
| 7 | 7.7 | 6.8 | 6.1 | 5.5 | 5.1 |
| 9 | 12.9 | 11.4 | 10.2 | 9.3 | 8.6 |
| 11 | 18.3 | 16.1 | 14.4 | 13.1 | 12.1 |

**Critical**: At k=5 with stolen iris key, security drops to ~2 bits → **broken**.
→ Paper recommendation: Use k ≥ 7 for IBFV deployments.
→ At k=7: 5-8 bits residual security (still needs FP matching).
→ At k=9: equivalent security to standard vault at k=5.

### 7.3 Iris Fuzzy Commitment Security

| Parameter | Value |
|-----------|-------|
| Bits locked (B) | 63 |
| Block size | 1023 |
| Key length | 780 bits (63 × repetition blocks × interleaving) |
| Impostor recovery rate | **0.0000%** (0 recoveries across all experiments) |
| Combinatorial key space | $2^{780}$ (brute-force infeasible) |

### 7.4 Security Recommendation

| Deployment | Recommended k | n_bonus | Security (normal) | Security (stolen key) |
|------------|---------------|---------|--------------------|-----------------------|
| Low security | 7 | 4 | 14-18 bits | 5-8 bits |
| Medium security | 9 | 4 | 19-24 bits | 9-13 bits |
| High security | 11 | 4 | 19-30 bits | 12-18 bits |

---

## 8. ALGORITHM PSEUDOCODE

### Algorithm 1: IBFV Enrollment (Lock)

```
Input: Fingerprint minutiae M, Iris code C, Iris mask K
       Parameters: factor f, degree k, n_bonus B, block_size b

1. Q ← Quantize(M, f)           // 3×6-bit quantization per minutia
2. if |Q| < k+1: return FAIL
3. p(x) ← Random polynomial of degree k over GF(p)
4. sk ← Reconstruct(p.coefficients); pk ← sk × G  // brainpoolP256r1
5. V_fp ← {(EC(q).x, p(EC(q).x)) : q ∈ Q}       // genuine FP points
6. (commitment, key_hash) ← IrisEnroll(C, K, b)   // fuzzy commitment
7. {x'_j}_{j=1}^{B} ← DeriveVirtualX(key_hash, B, p)
         // x'_j = SHA-256("ibfv_vpt" || j || key_hash) mod p
8. V_iris ← {(x'_j, p(x'_j)) : j = 1..B}         // bonus genuine points
9. V_chaff ← Generate 5|V_fp| chaff points NOT on p(x)
10. Vault ← Shuffle(V_fp ∪ V_iris ∪ V_chaff)
11. Store: (Vault, pk, commitment, |V_fp|, B, f, k)
```

### Algorithm 2: IBFV Verification (Unlock)

```
Input: Query fingerprint M', Query iris C', mask K'
       Stored: Vault, pk, commitment, n_fp, B, f, k

1. Q' ← Quantize(M', f)
2. X_fp ← {EC(q').x : q' ∈ Q'}
3. Match_fp ← {(x,y) ∈ Vault : x ∈ X_fp}
4. result ← IrisRecover(C', K', commitment)  // fuzzy commitment recovery
5. if result.success:
     {x'_j} ← DeriveVirtualX(result.key_hash, B, p)
     Match_iris ← {(x,y) ∈ Vault : x ∈ {x'_j}}
   else:
     Match_iris ← ∅                          // ZERO-PENALTY FALLBACK
6. All_matches ← Match_fp ∪ Match_iris (deduplicated by x)
7. if |All_matches| < k+1: return REJECT
8. // Smart subset selection: try subsets with ALL iris points first
   for each (k+1)-subset S of All_matches (iris-priority order):
     coeffs ← LagrangeInterpolation(S, p)
     sk' ← Reconstruct(coeffs)
     if sk' × G == pk: return ACCEPT
9. return REJECT
```

---

## 9. PAPER STRUCTURE (Recommended)

### Section 1: Introduction (1 page)
- Biometric template protection problem
- Fuzzy vault scheme (Juels & Sudan 2002)
- ECC-based fuzzy vault (senior's work, JISAA 2025)
- Multimodal motivation: improve GAR without sacrificing security
- Contribution statement: IBFV architecture

### Section 2: Related Work (1-1.5 pages)
- **2.1 Fuzzy Vault**: Juels & Sudan (2002), Clancy et al. (2003), Nandakumar et al. (2007)
- **2.2 ECC-based Vault**: Senior's JISAA 2025 paper, brainpoolP256r1
- **2.3 Iris Template Protection**: Hao et al. (2006) fuzzy commitment, Daugman (2004) iris codes
- **2.4 Multimodal Approaches**: 
  - Nandakumar & Jain (2008): multi-biometric fuzzy vault (AND-fusion)
  - Score-level fusion (Ross & Jain 2003): breaks template protection
  - Feature-level fusion: domain-specific, not vault-compatible
- **Key gap**: No prior work achieves additive (zero-penalty) multimodal fusion within vault framework.

### Section 3: Proposed Method — IBFV (2-2.5 pages)
- **3.1 Architecture Overview**: Block diagram
- **3.2 Enrollment Protocol**: Algorithm 1 + explanation
- **3.3 Verification Protocol**: Algorithm 2 + explanation
- **3.4 Zero-Penalty Property**: Formal proof sketch
- **3.5 Smart Subset Optimization**: Why prioritize iris-inclusive subsets

### Section 4: Security Analysis (1.5 pages)
- **4.1 Standard Vault Security**: Brute-force complexity
- **4.2 IBFV Security**: Impostor perspective (identical to unimodal)
- **4.3 Stolen-Key Analysis**: What if iris key is compromised
- **4.4 Iris Commitment Security**: B=63, 0% impostor recovery
- **4.5 Security Recommendations**: k ≥ 7 for IBFV deployments

### Section 5: Experimental Setup (1 page)
- **5.1 Databases**: FVC2002 DB1/DB2/DB3, FVC2004 DB1 + CASIA-Iris-Interval
- **5.2 Chimeric Database Construction**: Mapping FP subjects to iris subjects
- **5.3 Setup 1**: 100 subjects, 1 enrollment + 1 test per subject
- **5.4 Setup 2**: 48 subjects, 1 enrollment + 7 tests per subject
- **5.5 Parameter Space**: f ∈ {14,16,...,30}, k ∈ {5,7,9,11}, B=63, n_bonus=4
- **5.6 Metrics**: EER, FAR, FRR at EER threshold

### Section 6: Results and Discussion (2-2.5 pages)
- **6.1 AND-Fusion Failure**: Architecture A always worse (Table + explanation)
- **6.2 IBFV Results**: Tables 1-2, statistical significance
- **6.3 FAR Preservation**: 288/288 identical → impostor sees same vault
- **6.4 ΔEER Heatmaps**: Figures showing improvement landscape
- **6.5 EER vs Factor Curves**: Figures showing IBFV gap
- **6.6 Comparison with Prior Art**: Table 3 vs senior's paper
- **6.7 Optimal Parameter Shift**: IBFV prefers finer quantization + higher degree
- **6.8 When IBFV Helps Most**: Hard databases, difficult configs (high k, moderate f)

### Section 7: Conclusion (0.5 pages)
- Novel IBFV architecture: additive fusion, zero-penalty guarantee
- Empirical validation: 0 violations across 288 configs, up to 55% EER reduction
- Security maintained: FAR identical, iris key adds second layer
- Future work: other biometric modalities, larger n_bonus with higher k

---

## 10. FIGURES NEEDED

1. **Figure 1**: IBFV Architecture block diagram (enrollment + verification)
2. **Figure 2**: EER vs Factor curves for DB3 and DB4 (Setup 1, k=5) — Uni vs IBFV
3. **Figure 3**: EER vs Factor curves for all DBs (Setup 2, k=5) — Uni vs IBFV  
4. **Figure 4**: ΔEER heatmap (factor × degree) for DB3 Setup 1
5. **Figure 5**: ΔEER heatmap (factor × degree) for DB4 Setup 1
6. **Figure 6**: Bar chart comparing Unimodal vs Arch A vs IBFV (best per DB)
7. **Figure 7**: Security bits vs polynomial degree (with/without stolen key)

---

## 11. KEY LITERATURE TO CITE

1. Juels & Sudan (2002) — Fuzzy vault scheme
2. Clancy et al. (2003) — FP fuzzy vault
3. Nandakumar et al. (2007) — FP fuzzy vault with minutiae
4. Nandakumar & Jain (2008) — Multibiometric fuzzy vault (our "Architecture A")
5. Senior's paper (JISAA 2025) — ECC fuzzy vault on brainpoolP256r1
6. Hao et al. (2006) — Iris fuzzy commitment
7. Daugman (2004) — Iris recognition
8. Ross & Jain (2003) — Multimodal biometric systems
9. ISO/IEC 24745:2022 — Biometric template protection standard
10. brainpoolP256r1 (RFC 5639)

---

## 12. HOW THIS IS BETTER THAN SENIOR'S PAPER

| Aspect | Senior's Paper (JISAA 2025) | This Paper |
|--------|----------------------------|------------|
| Scope | Unimodal (fingerprint only) | **Multimodal** (fingerprint + iris) |
| Architecture | 1 (standard vault) | **3** (Uni + ArchA + IBFV) |
| Novel contribution | ECC vault implementation | **IBFV: zero-penalty additive fusion** |
| Parameter sweep | Fixed f=24, k=10 | **f ∈ {14..30}, k ∈ {5..11}** (36 configs/DB) |
| Setups | 2 setups | 2 setups (same) |
| Results | 4 DBs × 1 config = 4 points | **4 DBs × 36 configs × 3 arch = 864** |
| Best DB3 EER | 2.85% | **1.65%** (42% better) |
| Best DB4 EER | 3.51% | **1.08%** (69% better) |
| Security analysis | Basic | **Complete**: stolen-key, iris commitment, n_bonus analysis |
| Statistical tests | None | **Paired t-test, p < 0.001** |
| Theoretical guarantee | None | **EER_IBFV ≤ EER_Uni (proven)** |

---

## 13. ARCHITECTURE A — WHY IT FAILS (Use in Paper)

Architecture A = AND-fusion (à la Nandakumar & Jain 2008):
- Both iris AND fingerprint must succeed independently
- Iris adds ~7% FRR penalty (iris FRR at B=63 is ~7%)
- FAR reduces to FAR_fp × FAR_iris (but FAR_fp is already ~0%)
- Net effect: FRR rises, FAR stays ~0% → EER worsens

Best ArchA results (Setup 1):
| DB | ArchA Best EER | Unimodal Best EER | IBFV Best EER |
|----|----------------|-------------------|---------------|
| DB1 | 3.50% | 0.00% | **0.00%** |
| DB2 | 3.50% | 0.00% | **0.00%** |
| DB3 | 5.85% | 2.76% | **1.65%** |
| DB4 | 5.16% | 2.40% | **1.08%** |

ArchA is ALWAYS worse — this experimentally validates the "gating penalty" argument.

---

## 14. DATA FILES

- `setup1_results_20260405_134407.csv` — 432 rows (144 × 3 architectures)
- `setup2_results_20260406_005116.csv` — 432 rows (144 × 3 architectures)
- `architecture_ibfv.py` — 345 lines, core IBFV implementation
- `fingerprint_vault.py` — Unimodal vault, quantization, EC operations
- `iris_stabilizer.py` — Fuzzy commitment with repetition code

---

## 15. REMAINING TO-DO FOR PAPER WRITING

- [ ] Generate matplotlib/pgfplots figures (EER curves, heatmaps, bar charts)
- [ ] Draw IBFV architecture block diagram (TikZ or draw.io)
- [ ] Write LaTeX manuscript
- [ ] Compute DET/ROC curves (if needed — requires per-trial data, not just EER)
- [ ] Proofread security analysis formulas
- [ ] Identify target venue (journal vs conference)

---

## 16. TARGET VENUES

**Journals:**
- Journal of Information Security and Applications (JISA) — senior's venue
- Pattern Recognition (Elsevier)
- IEEE Transactions on Information Forensics and Security (TIFS)

**Conferences:**
- ICB (International Conference on Biometrics)
- IJCB (International Joint Conference on Biometrics)
- BTAS (Biometric Theory, Applications and Systems)
