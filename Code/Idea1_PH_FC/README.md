# Idea 1 — Persistent Homology on Functional Connectivity Matrices

## Research Question

> **Does the topological structure of whole-brain functional connectivity differ between ADHD and typically-developing brains, and can persistent homology detect these differences more sensitively than scalar connectivity summaries?**

Standard FC analysis collapses the network into pairwise correlations and asks which edges are stronger or weaker in ADHD. This idea instead asks about the *global shape* of the connectivity network: how many clusters form, how many loops exist, and how long those features persist as we gradually increase the connectivity threshold. The central hypothesis is that ADHD disrupts the modular organisation of brain networks, and that topological descriptors — which are sensitive to the geometry of the entire network simultaneously — will capture this disruption in ways that scalar metrics miss.

---

## Pipeline Overview

```
Raw fMRI (4D NIfTI)
        │
        ▼
[1] Atlas Masking (MSDL, 39 ROIs)
        │  NiftiMapsMasker → time series T ∈ ℝ^{n_timepoints × 39}
        ▼
[2] Functional Connectivity Matrix
        │  Pearson correlation → FC ∈ ℝ^{39 × 39}
        ▼
[3] Distance Matrix
        │  D = 1 − |FC|, clipped to [0, 1], diagonal = 0
        ▼
[4] Persistent Homology (ripser, H0/H1/H2)
        │  One ripser call per subject → persistence diagrams
        ▼
[5] Diagram Cleaning
        │  Strip infinite-death bars; extract finite (birth, death) pairs
        ▼
[Exp 1] Group comparison via total persistence + Mann-Whitney U
[Exp 2] H0 vs H1 vs H2 descriptive summary
[Exp 3] Subtype analysis — KMeans on H1 total persistence
[Exp 4] Atlas scale dependence — repeat with Schaefer100 / Schaefer200
```

---

## Step-by-Step Explanation

### Step 1 — Atlas Masking

The MSDL (Multi-Subject Dictionary Learning) atlas parcellates the brain into 39 probabilistic regions. Because MSDL is a 4D soft-assignment atlas, `NiftiMapsMasker` is used (not `NiftiLabelsMasker`). It computes a weighted-average signal per region per time point.

Output per subject: **T** of shape `(n_timepoints, 39)`. Time-series length varies across sites (77–261 TRs at TR = 2 s).

### Step 2 — Functional Connectivity Matrix

```
FC[i, j] = Pearson correlation(T[:, i], T[:, j])
```

`ConnectivityMeasure(kind='correlation', standardize='zscore_sample')` handles this, applying z-score standardisation before computing correlations to reduce site and session effects. The resulting FC matrix is 39×39, symmetric, with values in [−1, 1].

### Step 3 — Distance Matrix

Vietoris–Rips filtration requires distances, not similarities. The transformation is:

```
D[i, j] = 1 − |FC[i, j]|
```

Taking the **absolute value** means that both strongly *positive* and strongly *negative* correlations map to low distance (near 0). Uncorrelated ROI pairs map to high distance (near 1). This treats anti-correlated networks (e.g. DMN and dorsal attention) as equally "close" as positively correlated ones, encoding the full connectivity structure symmetrically.

The matrix is:
- Clipped to [0, 1]
- Diagonal forced to 0 (a point is distance 0 from itself)
- Symmetrised: `D = (D + D.T) / 2` to correct floating-point asymmetry

### Step 4 — Persistent Homology

`ripser` builds a **Vietoris–Rips filtration** on the distance matrix. At each threshold ε, an edge is added between ROIs i and j if `D[i,j] ≤ ε`. Higher-dimensional simplices (triangles, tetrahedra) are added whenever all their edges are present.

As ε grows from 0 to 1, the simplicial complex evolves and topological features are born and die:

| Dimension | Feature type | Interpretation |
|---|---|---|
| H₀ (β₀) | Connected component | A cluster of mutually correlated ROIs |
| H₁ (β₁) | 1-cycle (loop) | A ring of ROIs with correlated neighbours but no "shortcut" through the interior |
| H₂ (β₂) | 2-sphere (void) | A hollow enclosed cavity; requires ≥4 ROIs forming a tetrahedron with missing interior |

`ripser` is called with `distance_matrix=True` and `maxdim=2` (computes H0, H1, H2). The output is a list of persistence diagrams — one per dimension — each a set of `(birth, death)` pairs.

**Note on infinite bars:** In H₀, the last connected component (the final merge of the whole network into one cluster) has death = ∞. This infinite bar is stripped before any computation — it is an artefact of the filtration reaching its maximum, not a meaningful topological feature.

### Step 5 — Diagram Cleaning

`strip_infinite_bars()` removes any bar where `death == np.inf`. This affects:
- H₀: always removes exactly one bar (the last component to merge)
- H₁, H₂: rarely produces infinite bars in FC data, but stripped for safety

The cleaned diagrams — finite `(birth, death)` arrays — are what all downstream experiments use.

---

## Scalar Summary: Total Persistence

The primary scalar biomarker extracted from each persistence diagram is **total persistence**:

```
TP_k = Σ (death_i − birth_i)   for all bars i in H_k
```

This sums the lifetimes of all topological features in dimension k. A large TP means many long-lived features — the network has topological structure that persists across a wide range of connectivity thresholds. A small TP means features appear and disappear quickly, indicating a more uniform or noisy connectivity pattern.

Total persistence is computed with `BasePersistenceComputer.total_persistence(diagram)`, a static method defined in `Core/base_tda.py`.

---

## Experiments and Outputs

### Experiment 1 — Group Comparison (ADHD vs Control)

**Output files:** `group_comparison_total_persistence.csv`, `total_persistence_comparison.png`, `wasserstein_heatmap.png`, `persistence_diagrams_H{dim}.png`, `fc_matrix_{subject_id}.png`, `distance_matrix_{subject_id}.png`

For each homological dimension (H0, H1, H2), total persistence is computed per subject and tested with **Mann-Whitney U** (non-parametric, appropriate for small samples). Effect size is reported as **Cohen's d**.

The persistence diagram overlay plots all ADHD (pink) and control (blue) diagrams for each dimension on the same axes. Points close to the diagonal (short-lived features) represent topological noise; points far from the diagonal are long-lived, persistent features.

**Wasserstein distance heatmap:** The pairwise Wasserstein distance between every pair of subjects' H₁ diagrams is computed and displayed as a sorted heatmap (controls first, then ADHD). The Wasserstein distance between two persistence diagrams is the optimal transport cost of matching their bars — it measures how different two subjects' topological profiles are. If ADHD subjects cluster together (low within-group Wasserstein distances, high between-group distances), the heatmap will show a visible block structure.

The Wasserstein distance between diagrams A and B is:

```
W_p(A, B) = min_{matching σ} ( Σ_i ||a_i − b_{σ(i)}||^p )^{1/p}
```

where unmatched points are matched to their projection on the diagonal (representing a feature with zero lifetime). `persim` is used for computation with `order=1` (p=1).

**What a group difference would mean:** Higher H₁ total persistence in ADHD would indicate more and/or longer-lived loops in the FC network — a more complex cyclic connectivity structure. This could reflect reduced modularity (network communities that are less cleanly separated), consistent with ADHD literature showing disrupted default-mode and frontoparietal network organisation.

### Experiment 2 — H0 vs H1 vs H2 Summary

**Output file:** `h0_h1_h2_summary.csv`

Descriptive statistics (mean, std, median, min, max) of total persistence across all subjects, broken down by dimension. This experiment does not test group differences — it characterises the overall magnitude of topological signal at each scale.

Expected pattern:
- H₀ total persistence is typically large (many components merging over a wide ε range)
- H₁ total persistence is moderate (loops form and fill, but over a narrower range)
- H₂ total persistence is typically small (voids are rare and short-lived in FC data)

This establishes which homological dimension carries the most topological signal, informing which dimension is most informative for downstream classification or correlation analyses (used by Ideas 4 and 5).

### Experiment 3 — Subtype Analysis

**Output files:** `subtype_clusters.csv`, `subtype_clusters.png`

ADHD is clinically heterogeneous — the DSM-IV recognises inattentive, hyperactive-impulsive, and combined presentations. This experiment tests whether subjects separate into distinct topological subtypes using **KMeans clustering (k=2)** on H₁ total persistence values.

Clustering is applied to all subjects (not just ADHD) to allow the algorithm to find natural groupings. The **silhouette score** measures cluster quality:
- Score near 1.0: subjects are well-separated into two distinct groups
- Score near 0.0: clusters overlap and the separation is arbitrary

The scatter plots show cluster membership alongside ADHD/control labels, testing whether topological clusters correspond to clinical subtypes.

### Experiment 4 — Atlas Scale Dependence

**Output file:** `atlas_scale_comparison.png`

The same pipeline is re-run with three atlas resolutions:

| Atlas | Type | ROIs | Description |
|---|---|---|---|
| MSDL | Probabilistic (4D) | 39 | Network-based functional parcellation |
| Schaefer100 | Hard-label (3D) | 100 | Cortical parcellation, 100 regions |
| Schaefer200 | Hard-label (3D) | 200 | Cortical parcellation, 200 regions |

The bar chart shows mean total persistence per dimension for each atlas. This tests whether the topological signal is stable across spatial scales or is an artefact of a particular parcellation choice. If the same dimension (e.g. H₁) consistently shows higher total persistence in ADHD across all three atlases, that strengthens the finding. If results flip across atlases, the analysis is sensitive to the ROI definition.

Coarser atlases (fewer ROIs) produce sparser distance matrices — fewer edges enter the filtration, so fewer topological features are created. Finer atlases (more ROIs) produce richer filtrations with more features but also more noise. Scale dependence is a fundamental robustness check in TDA.

---

## Output Files Summary

| File | Produced by | Description |
|---|---|---|
| `fc_matrix_{id}.png` | Step 2 (first subject) | FC matrix heatmap (Pearson correlation) |
| `distance_matrix_{id}.png` | Step 3 (first subject) | Distance matrix heatmap (1 − \|FC\|) |
| `persistence_diagrams_H{dim}.png` | Step 4 | All subjects' diagrams overlaid, by dimension |
| `group_comparison_total_persistence.csv` | Exp 1 | Mann-Whitney p-values and Cohen's d per dimension |
| `total_persistence_comparison.png` | Exp 1 | Boxplots of total persistence by group, per dimension |
| `wasserstein_heatmap_H{dim}.png` | Exp 1 | Pairwise Wasserstein distances for H0, H1, H2 sorted by group |
| `h0_h1_h2_summary.csv` | Exp 2 | Descriptive stats of total persistence per dimension |
| `subtype_clusters_H{dim}.csv` | Exp 3 | Per-subject cluster assignment for each dimension |
| `subtype_clusters_H{dim}.png` | Exp 3 | Cluster scatter plots for H0, H1, H2 coloured by cluster and ADHD label |
| `atlas_scale_comparison.png` | Exp 4 | Bar chart of mean total persistence across atlas scales |

---

## Mathematical Reference

| Symbol | Definition |
|---|---|
| T | ROI time series, shape (n_timepoints, 39) |
| FC | Functional connectivity matrix; FC[i,j] = Pearson corr(T[:,i], T[:,j]) |
| D | Distance matrix; D[i,j] = 1 − \|FC[i,j]\|, clipped to [0,1] |
| VR(D, ε) | Vietoris–Rips complex on D at threshold ε |
| dgm_k | Persistence diagram for dimension k: finite set of (birth, death) pairs |
| TP_k | Total persistence for dimension k: Σ (death_i − birth_i) |
| W_1(A,B) | Wasserstein-1 distance between persistence diagrams A and B |
| d | Cohen's d effect size for group comparison |
| U | Mann-Whitney U statistic |

---

## Key Parameters (`Parameters/params_idea1.py`)

| Parameter | Default | Effect |
|---|---|---|
| `n_subjects` | 30 | Number of ADHD-200 subjects to load |
| `atlas_name` | `"msdl"` | Atlas for ROI extraction |
| `max_dimension` | 2 | Highest homological dimension computed (H0, H1, H2) |
| `max_edge_length` | 1.0 | Filtration threshold cap; 1.0 = full distance range |
| `correlation_kind` | `"correlation"` | Pearson; alternatives: `"partial correlation"`, `"tangent"` |
| `run_atlas_scale` | `True` | Whether to re-run with Schaefer100 / Schaefer200 |

---

## Entry Point

```bash
# From project root:
python -m Code.Idea1_PH_FC.run_idea1

# Or via main:
python main.py --idea 1
```

Outputs are saved to `Output/Idea1/`.
