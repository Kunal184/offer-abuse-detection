# Evaluation Design Analysis and Recommendation

## Executive Summary

**Current Problem**: Test set composition changes when generator modifications alter abuse group IDs, making baseline and post-modification F1 scores non-comparable.

**Root Cause**: Group-aware split uses `random_state=42` to shuffle abuse groups, but the shuffle operates on a different set of group IDs after adding the patient archetype. This is **not a bug**—it's expected random seed behavior—but it breaks longitudinal comparability.

**Recommendation**: Switch to **explicit test group selection** with archetype balancing. The current dataset (16 groups, 79 abusers) is sufficient for a small but representative held-out set.

---

## 1. What Is Wrong With Current Evaluation

### Current Implementation (train.py:154-211)

```python
def split_group_aware(df, test_group_ratio=0.20, val_group_ratio=0.13, random_state=42):
    # Shuffle all abuse groups deterministically
    rng = np.random.RandomState(random_state)
    shuffled_abuse_groups = list(unique_abuse_groups)
    rng.shuffle(shuffled_abuse_groups)
    
    # Take first 20% as test
    n_test_g = int(np.round(n_groups * test_group_ratio))
    test_groups = shuffled_abuse_groups[:n_test_g]
    ...
```

### The Problem

**Baseline run** (16 groups, no patient archetype):
- Group IDs: `abuse_group_1` through `abuse_group_16` (different archetypes)
- Seed 42 shuffle → test groups: `abuse_group_10_evasive_stealth`, `abuse_group_11_volume`, `abuse_group_15_ghost`
- 15 abuse samples, 67% evasive

**Current run** (16 groups, includes patient archetype):
- Group IDs: `abuse_group_1_evasive_stealth` through `abuse_group_16_patient` (different archetypes)
- Seed 42 shuffle → test groups: `abuse_group_12_evasive_proxy`, `abuse_group_4_slow_drip`, `abuse_group_7_volume`
- 19 abuse samples, 32% evasive

**Result**: Comparing baseline F1 0.857-0.903 to current F1 0.957-1.000 is meaningless—they evaluate on different difficulty distributions.

### Why This Matters

- **No longitudinal tracking**: Cannot measure whether timing fix, feature engineering changes, or hyperparameter tuning actually improve generalization
- **Cherry-picking risk**: Running multiple generator variations could accidentally produce an "easy" test set
- **Reproducibility loss**: External collaborators cannot replicate evaluation methodology

---

## 2. Current Dataset Inventory

### Abuse Groups (16 total, 79 customers)

| Group ID | Archetype | Size | Difficulty | Entity Sharing Pattern |
|----------|-----------|------|------------|------------------------|
| abuse_group_1_evasive_stealth | evasive | 3 | **HARD** | None (fully decoupled) |
| abuse_group_2_fast | fast | 5 | Easy | High (fast coordination) |
| abuse_group_3_fast | fast | 4 | Easy | High |
| abuse_group_4_slow_drip | slow_drip | 6 | Medium | Moderate (temporal spread) |
| abuse_group_5_evasive_proxy | evasive | 6 | **HARD** | None (decoupled) |
| abuse_group_6_volume | volume | 5 | Medium | High (many orders) |
| abuse_group_7_volume | volume | 7 | Medium | Very high (cluster_size 32 per analysis) |
| abuse_group_8_patient | patient | 4 | **HARD** | Moderate (slow timing, overlaps legit) |
| abuse_group_9_evasive_proxy | evasive | 3 | **HARD** | None |
| abuse_group_10_evasive_proxy | evasive | 3 | **HARD** | None |
| abuse_group_11_evasive_proxy | evasive | 3 | **HARD** | None |
| abuse_group_12_evasive_proxy | evasive | 6 | **HARD** | None |
| abuse_group_13_slow_drip | slow_drip | 6 | Medium | Moderate |
| abuse_group_14_fast | fast | 5 | Easy | High |
| abuse_group_15_evasive_proxy | evasive | 6 | **HARD** | None |
| abuse_group_16_patient | patient | 7 | **HARD** | Moderate (98h median timing) |

### Archetype Distribution

- **Evasive (proxy/stealth)**: 9 groups (35 customers) — **HARD** (no entity sharing, behavioral detection only)
- **Patient**: 2 groups (11 customers) — **HARD** (timing overlaps legitimate, requires graph features)
- **Fast**: 3 groups (14 customers) — **EASY** (rapid coordination, clear timing signal)
- **Slow drip**: 2 groups (12 customers) — **MEDIUM** (temporal spread)
- **Volume**: 2 groups (12 customers) — **MEDIUM** (high order count)

### Statistical Sufficiency

- **16 groups total** → 20% holdout = **3-4 groups** (small but acceptable for synthetic data)
- **79 abusers** → ~15-20 abuse samples in test (matches current)
- **923 legitimate** → ~140 legit samples in test (stable)
- **Test set stability**: With 15-20 abuse samples, each error changes recall by 5-7% (acceptable given synthetic evaluation context)

---

## 3. Evaluation Design Options

### Option A: Keep Random Shuffle, Freeze Group IDs (Rejected)

**Idea**: Never change generator group naming, so seed 42 always picks the same IDs.

**Problems**:
- Brittle: Adding/removing archetypes breaks it
- Doesn't solve archetype balancing problem
- Still no control over test difficulty

### Option B: Stratified Group Sampling (Rejected)

**Idea**: Randomly sample test groups proportionally by archetype (e.g., 20% of evasive, 20% of fast, etc.)

**Problems**:
- Small counts (e.g., 2 patient groups → 0.4 groups in test, rounds to 0 or 1)
- Introduces sampling variance across runs
- Still seed-dependent

### Option C: Explicit Test Group Selection (Recommended)

**Idea**: Hardcode a representative test set covering all difficulty levels.

**Advantages**:
- ✅ **Reproducible**: Same test set across all future generator/model iterations
- ✅ **Balanced**: Intentionally covers easy, medium, and hard cases
- ✅ **Transparent**: Anyone reading the code knows exactly what's being evaluated
- ✅ **Longitudinal**: F1 changes directly reflect model/feature improvements

**Disadvantages**:
- ❌ Slightly reduces train set diversity (but 13 train groups is sufficient)
- ❌ Requires manual selection (but only once)

---

## 4. Recommended Test Set Composition

### Selected Test Groups (4 groups, 20 abuse samples)

| Group ID | Archetype | Size | Rationale |
|----------|-----------|------|-----------|
| **abuse_group_1_evasive_stealth** | evasive_stealth | 3 | **HARD**: Fully decoupled, tests if models learned behavioral patterns beyond entity sharing |
| **abuse_group_8_patient** | patient | 4 | **HARD**: Slow timing (98h median), overlaps legitimate distribution, requires graph features |
| **abuse_group_13_slow_drip** | slow_drip | 6 | **MEDIUM**: Temporal spread, moderate entity sharing |
| **abuse_group_7_volume** | volume | 7 | **MEDIUM**: High order count, strong entity sharing (cluster_size 32), tests graph feature importance |

**Total**: 20 abuse samples

### Why These Groups?

1. **Difficulty coverage**:
   - 2 HARD groups (stealth decoupled, patient slow timing) = 35%
   - 2 MEDIUM groups (slow_drip, volume) = 65%
   - No EASY groups (fast coordination is too trivial for held-out evaluation)

2. **Archetype diversity**:
   - ✅ Evasive stealth (no entity sharing)
   - ✅ Patient (timing challenge)
   - ✅ Slow drip (temporal spread)
   - ✅ Volume (high connectivity)

3. **Feature dimension coverage**:
   - **Timing**: Patient archetype (98h) tests if models over-rely on `time_to_first_order_hours`
   - **Graph features**: Volume group (cluster_size 32) tests `max_device_user_count`, `cluster_size`
   - **Behavioral**: Evasive stealth (no entities) tests `order_redemption_rate`, `average_spend`

4. **Representative sample size**:
   - 20 abuse samples (each error = 5% recall change) — acceptable for synthetic data
   - ~25% of total abusers held out
   - Leaves 13 groups (59 abusers) for training — sufficient diversity

### Alternative: Include 1 Fast Group for Easier Test Set

If we want to ensure the test set is not "too hard", we could replace `abuse_group_13_slow_drip` with `abuse_group_14_fast`:

| Group ID | Archetype | Size | Rationale |
|----------|-----------|------|-----------|
| **abuse_group_1_evasive_stealth** | evasive_stealth | 3 | HARD (decoupled) |
| **abuse_group_8_patient** | patient | 4 | HARD (slow timing) |
| **abuse_group_14_fast** | fast | 5 | EASY (rapid coordination) |
| **abuse_group_7_volume** | volume | 7 | MEDIUM (high connectivity) |

**Total**: 19 abuse samples

**Tradeoff**: Including 1 EASY group makes the test set more forgiving (higher F1 expected), but ensures we catch catastrophic model failures.

**Recommendation**: **Use the first set (stealth, patient, slow_drip, volume)** because:
- Fast groups are too easy (timing alone catches them)
- We want to evaluate graph features and behavioral patterns, not just timing shortcuts
- If models struggle on this test set, it reveals real weaknesses

---

## 5. Implementation: Smallest Code Change

### Current Code (train.py:154-180)

```python
def split_group_aware(df, test_group_ratio=0.20, val_group_ratio=0.13, random_state=42):
    # Shuffle all abuse groups
    rng = np.random.RandomState(random_state)
    shuffled_abuse_groups = list(unique_abuse_groups)
    rng.shuffle(shuffled_abuse_groups)
    
    n_groups = len(shuffled_abuse_groups)
    n_test_g = int(np.round(n_groups * test_group_ratio))
    
    test_groups = shuffled_abuse_groups[:n_test_g]
    val_groups = shuffled_abuse_groups[n_test_g:n_test_g + n_val_g]
    train_groups = shuffled_abuse_groups[n_test_g + n_val_g:]
```

### Proposed Change (Option 1: Explicit Test Set)

```python
def split_group_aware(df, test_group_ratio=0.20, val_group_ratio=0.13, random_state=42,
                      fixed_test_groups=None):
    """
    Hold out entire abuse groups into train, validation, and test sets.
    
    Args:
        fixed_test_groups: Optional list of explicit test group IDs. If provided, these
                          groups are held out for test regardless of test_group_ratio.
                          Val groups are still randomly sampled from remaining groups.
    """
    feature_cols = get_feature_cols(df)
    
    abuse_df = df[df["is_abuse"] == 1]
    group_sizes = abuse_df.groupby("abuse_group_id")["customer_id"].count().to_dict()
    unique_abuse_groups = list(group_sizes.keys())
    
    if fixed_test_groups is not None:
        # Explicit test set
        test_groups = [g for g in fixed_test_groups if g in unique_abuse_groups]
        remaining_groups = [g for g in unique_abuse_groups if g not in test_groups]
        
        # Random val split from remaining groups
        rng = np.random.RandomState(random_state)
        shuffled_remaining = list(remaining_groups)
        rng.shuffle(shuffled_remaining)
        
        n_val_g = int(np.round(len(remaining_groups) * val_group_ratio / (1 - test_group_ratio)))
        val_groups = shuffled_remaining[:n_val_g]
        train_groups = shuffled_remaining[n_val_g:]
    else:
        # Original random shuffle behavior
        rng = np.random.RandomState(random_state)
        shuffled_abuse_groups = list(unique_abuse_groups)
        rng.shuffle(shuffled_abuse_groups)
        
        n_groups = len(shuffled_abuse_groups)
        n_test_g = int(np.round(n_groups * test_group_ratio))
        n_val_g = int(np.round(n_groups * val_group_ratio))
        
        test_groups = shuffled_abuse_groups[:n_test_g]
        val_groups = shuffled_abuse_groups[n_test_g:n_test_g + n_val_g]
        train_groups = shuffled_abuse_groups[n_test_g + n_val_g:]
    
    # Rest of function unchanged...
```

### Usage in main()

```python
# Define fixed test set (reproducible across all future runs)
FIXED_TEST_GROUPS = [
    "abuse_group_1_evasive_stealth",  # HARD: decoupled
    "abuse_group_8_patient",           # HARD: slow timing
    "abuse_group_13_slow_drip",        # MEDIUM: temporal spread
    "abuse_group_7_volume",            # MEDIUM: high connectivity
]

# Group-aware split with explicit test set
group_data = split_group_aware(df, fixed_test_groups=FIXED_TEST_GROUPS)
```

### Lines Changed: ~15 (backward compatible, no behavioral change if `fixed_test_groups=None`)

---

## 6. Challenging the Premise: Is Explicit Selection Inferior?

### Objection 1: "Manual selection introduces human bias"

**Response**: This is synthetic data with **known ground truth archetypes**. We are not cherry-picking easy cases—we are ensuring **coverage** of intended difficulty dimensions. Random selection on 16 groups with 7 archetypes is **more likely to produce unbalanced test sets** (as we've seen: baseline 67% evasive, current 32% evasive).

### Objection 2: "Random selection is more 'scientific'"

**Response**: Random selection is appropriate when:
- Dataset is large (hundreds of groups)
- Archetypes are unknown or evenly distributed
- Goal is to estimate expected performance on arbitrary new groups

Our context:
- **Small dataset** (16 groups)
- **Known archetypes with varying difficulty**
- **Goal is to evaluate model improvements across iterations, not estimate production F1**

With 16 groups and 7 archetypes, a 20% random sample (3 groups) will often miss entire archetypes. Explicit selection guarantees coverage.

### Objection 3: "It's overfitting the test set"

**Response**: Overfitting would be selecting groups **based on model predictions**. We are selecting based on **generator design intent** (archetype diversity). The test groups are chosen **before seeing model results**, based solely on:
1. Archetype labels (known from generator)
2. Group size (for sample count)
3. Difficulty theory (evasive = hard, fast = easy)

This is **stratified sampling by design**, not outcome-based selection.

### Objection 4: "What if we add more archetypes later?"

**Response**: Update `FIXED_TEST_GROUPS` to include 1 representative from the new archetype. The **principle** remains: explicit archetype-balanced selection. As the generator evolves, the test set evolves **intentionally**, not randomly.

### Verdict: Explicit Selection is Superior for This Project

**Why?**
- Small group count (16) makes random sampling unstable
- Synthetic data with known archetypes → stratified sampling is best practice
- Longitudinal evaluation requires fixed test set
- Transparent and reproducible

---

## 7. What Metrics Should We Report?

### Primary Metrics (Group-Aware Test Set)

1. **F1 Score** (main evaluation metric)
   - Balances precision and recall
   - Single number for longitudinal tracking

2. **Precision** and **Recall** (separately reported)
   - Precision: How many flagged accounts are actual abusers? (business cost of false positives)
   - Recall: How many abuse rings are caught? (business cost of missed abuse)

3. **ROC-AUC** (threshold-independent)
   - Evaluates model's ability to rank abusers higher than legitimate users
   - Useful for understanding probability calibration

4. **Confusion Matrix** (TP, FP, FN, TN)
   - Absolute counts (since test set is small, percentages are misleading)

### Secondary Metrics (Per-Archetype Breakdown)

For each archetype in the test set:
- **Recall by archetype** (e.g., "Did we catch the patient group?")
- **Identifies which behavioral patterns the model struggles with**

Example report:
```
Test Set Performance:
  Overall: F1=0.875, Prec=0.900, Rec=0.850, ROC-AUC=0.95

  By Archetype:
    - evasive_stealth (n=3): 2/3 caught (67% recall) — HARD CASE
    - patient (n=4):         3/4 caught (75% recall) — HARD CASE
    - slow_drip (n=6):       6/6 caught (100% recall)
    - volume (n=7):          7/7 caught (100% recall)
```

### What NOT to Report

- ❌ **Stratified split results** (known to leak, not trustworthy)
- ❌ **Training set accuracy** (always high, not informative)
- ❌ **Validation set results in final report** (used for hyperparameter tuning only)

---

## 8. Recommended Experiment After Fixing Evaluation

### Experiment: Retrain with Fixed Test Set and Compare to Baseline

**Goal**: Establish a **new reproducible baseline** with explicit test set, then verify timing fix impact.

**Steps**:

1. **Modify train.py** to accept `fixed_test_groups` parameter (15 lines)

2. **Define explicit test set**:
   ```python
   FIXED_TEST_GROUPS = [
       "abuse_group_1_evasive_stealth",
       "abuse_group_8_patient",
       "abuse_group_13_slow_drip",
       "abuse_group_7_volume",
   ]
   ```

3. **Retrain all models** (LR, RF, XGBoost) on current dataset (with timing fix)

4. **Report new baseline**:
   - F1, precision, recall, ROC-AUC on explicit test set
   - Per-archetype recall breakdown
   - Feature importance rankings

5. **Document in evaluation_results.json**:
   ```json
   {
     "evaluation_methodology": "group_aware_explicit_test_set",
     "test_groups": ["abuse_group_1_evasive_stealth", ...],
     "test_composition": {
       "evasive_stealth": 3,
       "patient": 4,
       "slow_drip": 6,
       "volume": 7
     },
     "models": {
       "LogisticRegression": { "f1": 0.XXX, ... }
     }
   }
   ```

6. **Commit as "Reproducible Baseline with Explicit Test Set"**

7. **Future iterations**: Any generator/feature/model changes use the **same test groups**, making F1 deltas directly comparable.

---

## 9. Summary and Recommendation

### What Is Wrong

- Test set composition changes when generator modifications alter group IDs
- Random shuffle with fixed seed produces different test sets on different group populations
- Baseline (F1 0.857-0.903) and current (F1 0.957-1.000) metrics are not comparable

### Best Evaluation Design

**Explicit test group selection** with archetype balancing:
- Hardcode 4 representative test groups covering hard/medium cases
- Keep test set frozen across all future iterations
- Randomize only validation split (since val is not reported in final metrics)

### Exactly Which Groups to Hold Out

**Primary Recommendation**:
```python
FIXED_TEST_GROUPS = [
    "abuse_group_1_evasive_stealth",  # 3 samples, HARD (no entities)
    "abuse_group_8_patient",           # 4 samples, HARD (slow timing)
    "abuse_group_13_slow_drip",        # 6 samples, MEDIUM (temporal spread)
    "abuse_group_7_volume",            # 7 samples, MEDIUM (high connectivity)
]
# Total: 20 abuse samples, 4 archetypes, 2 HARD + 2 MEDIUM
```

**Why**: Covers all key difficulty dimensions (timing, graph, behavioral) without being artificially easy.

### Is 16 Groups Sufficient?

**Yes**, for synthetic evaluation:
- 16 groups → 4 test groups (25%) is acceptable
- 20 abuse test samples → 5% recall change per error (tolerable variance)
- 13 train groups (59 abusers) → sufficient diversity for learning

**For production**: Would need 50+ abuse groups for stable metrics, but that's not the goal here.

### What Metrics to Report

**Primary**: F1, Precision, Recall, ROC-AUC, Confusion Matrix (group-aware test only)

**Secondary**: Per-archetype recall breakdown (identifies model weaknesses)

**Drop**: Stratified split results (leaked, not trustworthy)

### Smallest Code Change

**15 lines** in `train.py`:
- Add optional `fixed_test_groups` parameter to `split_group_aware()`
- If provided, use explicit groups; else fall back to random shuffle
- Backward compatible (no change if parameter omitted)

### What Experiment to Run Afterward

1. Retrain with explicit test set
2. Report new baseline metrics
3. Document test set composition in JSON
4. Commit as reproducible baseline
5. All future model/generator changes compare against this fixed test set

---

## Final Verdict

**Proceed with explicit test group selection.** The current random shuffle approach is **not wrong in principle**, but it's **inappropriate for small synthetic datasets with known archetypes and a need for longitudinal evaluation**.

Explicit selection is:
- ✅ More reproducible
- ✅ More transparent
- ✅ Better balanced (covers intended difficulty dimensions)
- ✅ Standard practice for stratified evaluation on small datasets

The only downside is 15 lines of code and a hardcoded list—well worth it for trustworthy, comparable metrics across iterations.
