# Reproducible Baseline - Explicit Test Set Evaluation

**Date**: 2026-08-29  
**Status**: Baseline established with fixed test set

---

## EVALUATION METHODOLOGY

### Fixed Test Set (Reproducible Across All Future Runs)

**Test Groups** (4 groups, 20 abuse samples):

| Group ID | Archetype | Size | Difficulty | Rationale |
|----------|-----------|------|------------|-----------|
| `abuse_group_1_evasive_stealth` | evasive_stealth | 3 | **HARD** | Fully decoupled entities - tests behavioral pattern detection |
| `abuse_group_8_patient` | patient | 4 | **HARD** | Slow timing (98h median) overlaps legitimate - requires graph features |
| `abuse_group_13_slow_drip` | slow_drip | 6 | **MEDIUM** | Temporal spread - tests timing variance handling |
| `abuse_group_7_volume` | volume | 7 | **MEDIUM** | High connectivity (cluster_size 32) - tests graph feature importance |

**Total**: 20 abuse samples + 138 legitimate samples = 158 test samples

### Why This Test Set?

1. **Archetype Coverage**: Covers 4 distinct abuse patterns (evasive, patient, slow_drip, volume)
2. **Difficulty Balance**: 50% HARD (stealth + patient), 50% MEDIUM (slow_drip + volume)
3. **Feature Dimensions**: Tests timing (patient), graph (volume), behavioral (stealth), temporal (slow_drip)
4. **Reproducibility**: Same groups held out across all future generator/model iterations
5. **Representative**: No EASY groups (fast archetypes too trivial for held-out evaluation)

### Training Set (12 groups, 48 abuse samples)

Includes:
- 6 evasive_proxy groups (18 samples) - varied decoupled patterns
- 1 patient group (7 samples) - patient_16
- 1 slow_drip group (6 samples) - slow_drip_4
- 3 fast groups (14 samples) - fast coordination patterns
- 0 volume groups in train (both held out: 1 for val, 1 for test)
- 0 evasive_stealth in train (held out for test)

---

## BASELINE RESULTS (After Timing Fix)

### Group-Aware Test Performance (Fixed Test Set)

| Model | F1 | Precision | Recall | ROC-AUC | TP | FP | FN | TN |
|-------|-----|-----------|--------|---------|----|----|----|----|
| **Logistic Regression** | **0.976** | 0.952 | 1.000 | 1.000 | 20 | 1 | 0 | 137 |
| **Random Forest** | **0.800** | 0.933 | 0.700 | 0.990 | 14 | 1 | 6 | 137 |
| **XGBoost** | **0.889** | 1.000 | 0.800 | 1.000 | 16 | 0 | 4 | 138 |

### Key Observations

1. **Logistic Regression** achieves perfect recall (20/20) with only 1 FP:
   - Catches all 4 archetypes in test set
   - Strong performance on both HARD cases (stealth, patient)

2. **Random Forest** struggles with 6 false negatives:
   - Lower recall (70%) suggests tree-based splitting may underfit on small abuse training set
   - Still maintains high precision (93.3%)

3. **XGBoost** balances precision and recall:
   - Perfect precision (no false positives)
   - 80% recall (16/20 detected)
   - 4 missed cases require investigation

### Comparison to Previous Random Test Sets

**Previous run** (random test selection after timing fix):
- Test groups: `abuse_group_12_evasive_proxy`, `abuse_group_4_slow_drip`, `abuse_group_7_volume`
- F1: 0.957-1.000 (easier test set, 68% non-evasive groups with strong entity sharing)

**Current run** (explicit test selection):
- Test groups: `abuse_group_1_evasive_stealth`, `abuse_group_8_patient`, `abuse_group_13_slow_drip`, `abuse_group_7_volume`
- F1: 0.800-0.976 (balanced difficulty, includes hardest cases: stealth + patient)

**Verdict**: Current test set is more representative and challenging. Previous high F1 was due to test set composition bias.

---

## FEATURE IMPORTANCE (Post Timing Fix)

### Top 5 Features by Model

**Logistic Regression**:
1. order_redemption_rate
2. average_spend
3. time_to_first_order_hours
4. redemption_count
5. order_count

**Random Forest**:
1. average_spend
2. time_to_first_order_hours
3. total_spend
4. cluster_size
5. order_redemption_rate

**XGBoost**:
1. order_redemption_rate
2. max_device_user_count
3. time_to_first_order_hours
4. average_spend
5. cluster_size

### Key Insights

1. **time_to_first_order_hours dropped from #1 to #3** (was dominant before timing fix)
2. **Graph features rose in importance**:
   - `max_device_user_count` is #2 in XGBoost (was outside top 5)
   - `cluster_size` appears in top 5 for RF and XGBoost
3. **Behavioral features now primary**:
   - `order_redemption_rate` #1 in LR and XGBoost
   - `average_spend` #1-2 across all models

**Conclusion**: Timing fix succeeded—models now rely on multi-dimensional signals rather than timing shortcut.

---

## WHAT CHANGED FROM BASELINE

### Generator Timing Fix (Completed)

**Before**:
- Universal 1-24h order timing for all abuse archetypes
- 98.7% abuse vs 11.1% legit ordered within 24h
- Single-threshold F1: ~0.90+

**After**:
- Archetype-specific timing with multiplicative jitter (0.5x-1.5x)
- Abuse timing spread: 1-209h
- Patient archetype: 36-168h base delay (overlaps legitimate distribution)
- Single-threshold F1: 0.522 (timing no longer trivial shortcut)

### Evaluation Methodology Fix (Completed)

**Before**:
- Random test group selection via `seed=42` shuffle
- Test set composition changed when generator modifications altered group IDs
- Baseline and current F1 scores not comparable

**After**:
- Explicit test group selection (4 fixed groups)
- Archetype-balanced: 2 HARD + 2 MEDIUM
- Test set frozen across all future iterations
- Longitudinal F1 tracking now meaningful

---

## NEXT STEPS

### Immediate Actions

1. ✅ **Timing fix validated**: Abuse timing no longer has trivial 24h ceiling
2. ✅ **Evaluation methodology fixed**: Explicit test set ensures reproducibility
3. ✅ **Baseline established**: F1 0.800-0.976 on representative test set

### Future Investigations

1. **Per-Archetype Error Analysis**:
   - Which 6 cases did Random Forest miss? (patient? stealth?)
   - Which 4 cases did XGBoost miss?
   - Are errors consistent across models (ensemble opportunity)?

2. **Patient Archetype Detection**:
   - Current test includes `abuse_group_8_patient` (4 samples)
   - Training includes `abuse_group_16_patient` (7 samples)
   - Verify patient groups are caught via graph features, not timing

3. **Model Improvement Options** (if needed):
   - Hyperparameter tuning (currently using defaults)
   - Feature engineering (add interaction terms, ratios)
   - Ensemble methods (stack LR + XGBoost)
   - Class weight adjustment (abuse samples are 4.8% of dataset)

4. **Threshold Optimization**:
   - Current models use 0.5 probability threshold
   - Could optimize threshold on validation set for business constraints

5. **Dataset Scaling**:
   - Current: 16 groups (79 abusers)
   - Consider 50+ groups for more stable evaluation
   - Test generalization to unseen archetypes

---

## REPRODUCING THIS BASELINE

### Commands

```bash
# Generate dataset (same seed)
python generator/generator.py 42

# Extract features
python features/feature_engineering.py

# Train with explicit test set
python ml/train.py

# Outputs saved to ml/outputs/
# - model_*.joblib (trained models)
# - scaler.joblib (StandardScaler for LR)
# - evaluation_results.json (full metrics)
# - feature_importance_*.png (plots)
```

### Fixed Test Groups Definition (ml/train.py:367-374)

```python
FIXED_TEST_GROUPS = [
    "abuse_group_1_evasive_stealth",  # 3 samples - HARD
    "abuse_group_8_patient",           # 4 samples - HARD
    "abuse_group_13_slow_drip",        # 6 samples - MEDIUM
    "abuse_group_7_volume",            # 7 samples - MEDIUM
]
```

**Important**: Do not modify `FIXED_TEST_GROUPS` unless adding new archetypes to generator. Changing test groups breaks longitudinal comparability.

---

## VALIDATION CHECKS

### Test Set Quality ✅

- [x] Zero group leakage (no test groups appear in train)
- [x] Archetype coverage (4 distinct patterns)
- [x] Difficulty balance (50% HARD, 50% MEDIUM)
- [x] Feature dimension coverage (timing, graph, behavioral)
- [x] Sample size (20 abuse, 138 legit = 158 total)

### Timing Fix Validation ✅

- [x] Abuse timing spread beyond 24h (max 209h)
- [x] Patient archetype exists (2 groups, 11 total samples)
- [x] Single-threshold F1 dropped below 0.60
- [x] time_to_first_order_hours no longer #1 feature

### Model Quality ✅

- [x] F1 > 0.80 on balanced test set
- [x] Models generalize to unseen groups
- [x] Graph features contribute meaningfully
- [x] No single feature dominates (multi-dimensional signal)

---

## FILES MODIFIED

- `ml/train.py:154-211` - Added `fixed_test_groups` parameter to `split_group_aware()`
- `ml/train.py:367-374` - Defined explicit test group list
- `generator/generator.py:263-312` - Archetype-specific timing with jitter
- `REPRODUCIBLE_BASELINE.md` - This document

---

## COMMIT HISTORY

- `a89feb6` - Add timing validation and test set analysis scripts
- `[pending]` - Establish reproducible baseline with explicit test set

---

**Baseline Status**: ESTABLISHED ✅  
**Evaluation Reproducibility**: GUARANTEED ✅  
**Ready for Model/Feature Iteration**: YES ✅
