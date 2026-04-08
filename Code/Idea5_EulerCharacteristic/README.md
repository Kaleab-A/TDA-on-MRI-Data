# Idea 5 — Euler Characteristic and Betti Curves

## Research Question

> **Do ADHD and typically-developing brains have measurably different topological structure in their functional connectivity networks, and can this structure be summarized by a single curve?**

Classical functional connectivity analysis collapses the network into a single correlation matrix and compares scalar summaries (e.g., mean connectivity). This idea takes a different approach: it treats the brain connectivity network as a topological object and asks how its shape — the number of isolated clusters, loops, and enclosed voids — evolves as we progressively connect more and more brain regions. This evolution is captured by **Betti curves** and the **Euler characteristic curve**, which are functions of a connectivity threshold rather than a single number.

---

## Pipeline Overview

```
Raw fMRI (4D NIfTI)
        │
        ▼
[1] Atlas Masking (MSDL, 39 ROIs)
        │  NiftiMapsMasker extracts mean signal per ROI per time point
        │  Output: time series matrix T ∈ ℝ^{n_timepoints × 39}
        ▼
[2] Functional Connectivity Matrix
        │  Pearson correlation across all ROI pairs → FC ∈ ℝ^{39 × 39}
        │  Standardised per-subject (z-score_sample)
        ▼
[3] Distance Matrix
        │  D = 1 − |FC|, clipped to [0, 1], diagonal = 0
        │  D is a valid metric for Vietoris–Rips filtration
        ▼
[4] Persistent Homology (ripser, maxdim=2)
        │  One ripser call per subject; returns persistence diagrams
        │  for dimensions 0, 1, 2
        ▼
[5] Betti Curve Computation (sweep ε ∈ [0, 1], 100 steps)
        │  β_k(ε) = number of homological features of dimension k
        │           born before ε and not yet dead at ε
        │  Output: β₀(ε), β₁(ε), β₂(ε), χ(ε) per subject
        ▼
[6] Group Aggregation
        │  Mean ± std curves per group (ADHD / Control)
        ▼
[Exp 1] Group EC and Betti Curve Plots
[Exp 2] Severity Correlation  ← requires fuller clinical data
[Exp 3] Network-specific Betti Curves (DMN, Aud, IPS)
[Exp 4] Functional Data Analysis (FDA / FPCA)
```

---

## Step-by-Step Explanation

### Step 1 — Atlas Masking

The MSDL (Multi-Subject Dictionary Learning) atlas parcellates the whole brain into 39 functional regions. Because MSDL is a **probabilistic** (soft-assignment, 4D) atlas, each voxel can belong to multiple regions with different weights. `NiftiMapsMasker` computes a weighted average signal per region at each time point.

The output for each subject is a matrix **T** of shape `(n_timepoints, 39)`, where `n_timepoints` varies across sites and sessions (77–261 TRs in this dataset at TR=2s).

### Step 2 — Functional Connectivity Matrix

The FC matrix is computed using Pearson correlation across time points for every pair of the 39 ROIs:

```
FC[i, j] = corr(T[:, i], T[:, j])
```

`ConnectivityMeasure` from nilearn handles this with z-score standardisation (`zscore_sample`) applied before computing correlations to reduce site/session effects.

The resulting **FC ∈ ℝ^{39×39}** is symmetric, with 1s on the diagonal and values in `[−1, 1]` off-diagonal.

### Step 3 — Distance Matrix

Vietoris–Rips filtration requires a **distance** (not similarity) matrix. The transformation is:

```
D[i, j] = 1 − |FC[i, j]|
```

Taking the absolute value means both strongly positive *and* strongly negative correlations are treated as "close" (distance near 0), while uncorrelated ROIs are far apart (distance near 1). The matrix is clipped to `[0, 1]`, the diagonal is forced to 0, and it is symmetrised to correct floating-point asymmetry.

### Step 4 — Persistent Homology

`ripser` builds a **Vietoris–Rips filtration** on the distance matrix. At each threshold ε, two ROIs are connected by an edge if `D[i,j] ≤ ε`. As ε increases from 0 to 1, the simplicial complex grows:

- At ε = 0: all 39 ROIs are isolated points (39 connected components)
- As ε increases: ROIs with low distance (high correlation) connect first
- Eventually: at ε = 1, the complex is fully connected

`ripser` tracks when each topological feature (component, loop, void) is **born** and when it **dies** as ε grows. This gives a **persistence diagram** — a set of `(birth, death)` pairs for each homological dimension k:

| Dimension | Feature | Interpretation |
|---|---|---|
| H₀ (β₀) | Connected component | A cluster of co-activating ROIs |
| H₁ (β₁) | 1-cycle (loop) | A closed circuit of correlated ROIs with no "shortcut" |
| H₂ (β₂) | 2-sphere (void) | An enclosed hollow cavity in connectivity space |

**On H₂ features and the 2D matrix:** The 39×39 distance matrix is a 2D *array*, but it encodes 39 *points* in an abstract metric space of unspecified geometric dimension. The Vietoris–Rips complex built on 39 points can have simplices of any dimension up to 38. A void (H₂ feature) requires at least 4 points forming a "hollow tetrahedron" — a triangle of edges with nothing filling the interior — which is easily possible among 39 ROIs. The matrix being stored as a 2D table does not constrain the topology of the space it represents.

In practice, β₂ values from FC distance matrices are usually small and short-lived (the features are born and die quickly as ε increases), because FC matrices tend to have a relatively low-dimensional correlation structure. The β₂ curve is included for completeness and to preserve the full Euler characteristic formula χ = β₀ − β₁ + β₂.

### Step 5 — Betti Curves

Rather than working with the persistence diagram directly, we summarise it as a **curve** over the filtration parameter:

```
β_k(ε) = |{ (b, d) ∈ dgm_k : b ≤ ε < d }|
```

This counts how many dimension-k features are alive at threshold ε. The result is a step function from ε = 0 to ε = 1.

Expected behaviour:
- **β₀(ε)**: starts at 39 (all ROIs isolated), decreases monotonically to 1 as ROIs merge into one connected component
- **β₁(ε)**: starts at 0, rises as loops form when redundant connections are added, then falls as loops fill in
- **β₂(ε)**: typically small; rises and falls similarly to β₁

The **Euler characteristic** at each ε is the alternating sum:

```
χ(ε) = β₀(ε) − β₁(ε) + β₂(ε)
```

χ(ε) is a single signed number summarising the net topological complexity of the connectivity network at each threshold. Positive values indicate more components than holes; negative values indicate the opposite.

### Step 6 — Group Aggregation

For each group (ADHD, Control), we compute the **mean ± std** Betti curve across all subjects at each ε. This produces a ribbon plot showing group-level topology with uncertainty.

---

## Experiments and Outputs

### Experiment 1 — Group Betti Curves, Euler Characteristic, and FC Matrices

**Output files:** `betti_curves_group.png`, `euler_characteristic_curves.png`, `fc_matrices_group.png`

Three subplots show mean ± std of β₀, β₁, β₂ for ADHD (pink) vs Control (blue).
The EC curve plot shows χ(ε) for both groups.
The FC matrix heatmap shows the group-averaged Pearson correlation matrix for each group side-by-side, plus their difference (ADHD − Control). This provides a direct visual of which ROI pairs drive the topological differences seen in the Betti curves — ROI pairs with large differences in the FC matrix are the edges that enter the Vietoris–Rips filtration at different ε values for each group.

**What a difference would mean:** If the ADHD curve lies consistently above or below the control curve in a particular ε range, it means the ADHD group has systematically more or fewer topological features of that type at those connectivity thresholds. For example:
- Higher β₁ in ADHD at intermediate ε → more redundant connectivity loops; the network has more "alternative paths"
- Lower β₀ drop-off in ADHD → ROIs start connecting at higher thresholds → weaker average connectivity
- EC crossing zero at a different ε → the balance between clustering and loop formation shifts at a different connectivity strength

### Experiment 2 — Severity Correlation *(disabled — data limitation)*

**Intended output:** `severity_scatter_H{dim}.png`

The area under the β_dim curve,

```
A_k = ∫₀¹ β_k(ε) dε   (approximated by trapezoidal rule)
```

would be correlated (Spearman ρ) with each subject's ADHD severity score (`conn_adhd` / `dsm_iv_tot`). This tests whether topological complexity quantitatively tracks symptom burden.

**Why it is disabled:** The nilearn ADHD-200 subset has clinical scores for only 2 of 13 ADHD subjects (both severity columns share the same 4 valid rows out of 30). A minimum of 3 is required for a non-degenerate correlation. Re-enable with `run_betti_vs_severity = True` if using a dataset with complete clinical records.

### Experiment 3 — Network-Specific Betti Curves

**Output file:** `network_betti_curves.png`

Rather than computing TDA on the full 39-ROI matrix, this experiment restricts the analysis to ROI subsets belonging to three canonical resting-state networks.

**No new correlation matrix is computed.** The per-subject 39×39 distance matrix (D = 1 − |FC|) already exists from Step 3. For each network, we simply extract the corresponding rows and columns — a sub-block of the distance matrix — giving a smaller square matrix containing only the pairwise distances between that network's ROIs:

```
# DMN has 4 ROIs at indices [3, 4, 5, 6]
D_DMN = D[ [3,4,5,6], :][:, [3,4,5,6] ]   # shape (4, 4)

# Run ripser on this 4×4 distance matrix
ripser(D_DMN, distance_matrix=True, maxdim=2)
```

This is done per subject, using `numpy.ix_` to slice both axes simultaneously. The Betti curves are then computed from the persistence diagrams of each sub-matrix, and group means are plotted as before.

| Network | MSDL ROIs | Indices | Size |
|---|---|---|---|
| DMN (Default Mode) | L DMN, Med DMN, Front DMN, R DMN | 3, 4, 5, 6 | 4×4 |
| Aud (Auditory) | L Aud, R Aud | 0, 1 | 2×2 |
| IPS (Dorsal Attention) | L IPS, R IPS, L Ant IPS, R Ant IPS | 17, 18, 37, 38 | 4×4 |

**Why this matters:** The full-brain analysis averages topological signal across all 39 ROIs. A group difference might exist within one specific network but be diluted when all networks are combined. By running TDA on each network in isolation, we can ask: *is the ADHD topological fingerprint localised to the DMN, or is it distributed across multiple networks?*

**Interpretation:** The DMN is consistently implicated in ADHD — reduced deactivation during cognitive tasks is a hallmark finding. Differences in β₁ for the DMN sub-matrix would indicate that the internal loop structure of the DMN (i.e., whether its four nodes form a "triangle with a gap" vs. a "fully filled triangle" in connectivity space) differs between groups. The IPS network is relevant because the dorsal attention system and DMN are typically anti-correlated; ADHD may disrupt this segregation.

### Experiment 4 — Functional Data Analysis (FDA)

**Output file:** `fda_results.png`

Betti curves are functions, not scalar vectors. FDA treats each subject's β₁(ε) curve as a single functional observation using `skfda`.

**Functional group test:** The `oneway_anova` test (or integrated L² permutation fallback) asks whether the mean β₁ curve of ADHD subjects is significantly different from that of controls as a *function*, accounting for the entire shape of the curve simultaneously rather than testing at each ε independently (which would require multiple-comparison correction).

```
H₀: μ_ADHD(ε) = μ_Control(ε)  for all ε ∈ [0, 1]
H₁: the mean curves differ at some ε
```

The test statistic is compared against a null distribution built by randomly permuting group labels 500 times.

**Functional PCA (FPCA):** Each subject's curve is projected onto the leading functional principal components — the "shapes" that explain the most cross-subject variance:

- **FPC1** (explains ~57% of variance in current data): the dominant mode of between-subject variation in how β₁ evolves with ε. Subjects with high scores on FPC1 differ systematically from those with low scores in the shape of their loop curve.
- The FPCA scatter plot (PC1 vs PC2) reveals whether ADHD and control subjects separate in this functional shape space.

---

## Key Numbers from the Current Run

| Parameter | Value |
|---|---|
| Subjects | 30 (13 ADHD, 17 Control) |
| Atlas | MSDL, 39 ROIs |
| Filtration grid | ε ∈ [0.0, 1.0], 100 steps |
| Homological dimensions | 0, 1, 2 |
| FDA permutations | 500 |
| FDA group test p-value | 0.952 (no significant group difference in β₁ curves) |
| FPCA explained variance | PC1=56.9%, PC2=14.1%, PC3=7.7% |
| Severity scores available | 4/30 subjects (2 ADHD, 2 Control) |

The high p-value (0.952) suggests no statistically significant difference in β₁ curves between groups with 30 subjects at this sample size. This is expected: TDA-based group tests typically require larger samples (n>50 per group) to achieve adequate power, especially since Betti curves are noisy for individual subjects.

---

## Mathematical Reference

| Symbol | Definition |
|---|---|
| T | ROI time series matrix, shape (n_timepoints, 39) |
| FC | Functional connectivity matrix; FC[i,j] = Pearson corr(T[:,i], T[:,j]) |
| D | Distance matrix; D[i,j] = 1 − \|FC[i,j]\|, ∈ [0, 1] |
| VR(D, ε) | Vietoris–Rips complex at threshold ε |
| dgm_k | Persistence diagram for dimension k: set of (birth, death) pairs |
| β_k(ε) | Betti number k at ε: count of active k-dimensional features |
| χ(ε) | Euler characteristic at ε: β₀(ε) − β₁(ε) + β₂(ε) |
| A_k | Area under β_k curve: ∫₀¹ β_k(ε) dε (trapezoidal approximation) |
| FPC_j | j-th functional principal component of β₁ curves |

---

## Entry Point

```bash
# From project root:
python -m Code.Idea5_EulerCharacteristic.run_idea5

# Or via main:
python main.py --idea 5
```

Outputs are saved to `Output/Idea5/`.
