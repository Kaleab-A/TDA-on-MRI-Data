# TDA on fMRI Data

Applies Topological Data Analysis (TDA) to resting-state fMRI data from two nilearn datasets (ADHD-200 and ABIDE) across five independent experiments. Each idea has its own code folder, parameter file, and output folder.

---

## Project Layout

```
MRI_Project/
├── main.py                         # Top-level entry point — run any idea from here
├── Core/                           # Shared infrastructure (data loading, masking, base classes)
├── Parameters/                     # One params file per idea (tweak settings here)
├── Code/
│   ├── Idea1_PH_FC/                # Persistent homology on FC matrices
│   ├── Idea2_Mapper/               # Mapper graph on brain state space
│   ├── Idea3_SlidingWindow/        # Sliding-window TDA on BOLD time series
│   ├── Idea4_Classification/       # Topological feature vectors for classification
│   └── Idea5_EulerCharacteristic/  # Euler characteristic & Betti curves
└── Output/
    ├── ADHD/                       # Results when run on the ADHD-200 dataset
    │   ├── Idea1/
    │   ├── Idea2/
    │   ├── Idea3/
    │   ├── Idea4/
    │   └── Idea5/
    └── ABIDE/                      # Results when run on the ABIDE dataset
        ├── Idea1/
        └── Idea5/
```

---

## The Five Ideas

### Idea 1 — Persistent Homology on FC Matrices (`Code/Idea1_PH_FC/`)

Builds a functional connectivity (FC) matrix per subject (Pearson correlation between ROI time series), converts it to a distance matrix, and runs Vietoris-Rips persistent homology (H0, H1, H2). Compares ASD/ADHD vs. control groups using total persistence and Wasserstein distances, and clusters subjects by persistence features.

Key files:
- `fc_matrix_builder.py` — builds the FC matrix
- `persistence_computer.py` — runs ripser on the distance matrix
- `distance_analyzer.py` — Wasserstein distances + clustering
- `idea1_visualizer.py` — all plots and CSVs

Output (`Output/ADHD/Idea1/` and `Output/ABIDE/Idea1/`):
| File | What it shows |
|------|--------------|
| `fc_matrix_*.png` | Per-subject FC matrix heatmap |
| `distance_matrix_*.png` | Corresponding distance matrix |
| `persistence_diagrams_H{0,1,2}.png` | Persistence diagrams by homology dimension |
| `wasserstein_heatmap_H{0,1,2}.png` | Pairwise Wasserstein distance between subjects |
| `total_persistence_comparison.png` | Group-level total persistence (case vs. control) |
| `subtype_clusters_H{0,1,2}.png/.csv` | Clustered subjects by persistence features |
| `atlas_scale_comparison.png` | Effect of atlas resolution on persistence |
| `h0_h1_h2_summary.csv` | Per-subject H0/H1/H2 total persistence |
| `group_comparison_total_persistence.csv` | Group means + statistical test |
| `diagrams_cache.pkl` | Cached persistence diagrams (speeds up re-runs) |

---

### Idea 2 — Mapper on Brain State Space (`Code/Idea2_Mapper/`)

Applies the Mapper algorithm to the subject-level brain state space (ROI time series projected through a lens function). Builds a topological graph summarizing the shape of the data and analyzes group differences in graph structure.

Key files:
- `mapper_builder.py` — constructs the Mapper graph
- `lens_functions.py` — lens/filter functions (e.g., PCA, L2 norm)
- `mapper_analyzer.py` — graph-level statistics
- `idea2_visualizer.py` — graph visualizations

Output: `Output/ADHD/Idea2/` (run on ADHD only; no saved figures were produced).

---

### Idea 3 — Sliding-Window TDA (`Code/Idea3_SlidingWindow/`)

Embeds windowed segments of the BOLD time series using time-delay embedding, computes H1 persistent homology on each window, and tracks a "loop score" (total H1 persistence) over time. Compares loop score dynamics between groups and sweeps over window sizes.

Key files:
- `window_embedder.py` — time-delay embedding of BOLD windows
- `window_ph_computer.py` — per-window persistent homology
- `loop_score_analyzer.py` — extracts and aggregates loop scores
- `idea3_visualizer.py` — all plots and CSVs

Output (`Output/ADHD/Idea3/`):
| File | What it shows |
|------|--------------|
| `loop_score_comparison.png` | Group-level loop score distributions |
| `loop_score_autocorrelation.png` | Autocorrelation of loop score time series |
| `temporal_h1_evolution.png` | H1 persistence evolving across time windows |
| `window_size_sweep.png/.csv` | Loop score sensitivity to window size |
| `loop_score_per_subject.csv` | Per-subject mean loop scores |

---

### Idea 4 — Classification with Topological Features (`Code/Idea4_Classification/`)

Extracts vectorized topological features from persistence diagrams (persistence entropy, Betti numbers, landscape statistics) and uses them to classify ADHD vs. control with cross-validated classifiers. Includes a permutation test for significance and feature importance analysis.

Key files:
- `feature_extractor.py` — converts diagrams to feature vectors
- `classifier_pipeline.py` — cross-validated SVM/RF classification
- `permutation_tester.py` — permutation test for classification accuracy
- `idea4_visualizer.py` — accuracy plots and feature importance

Output (`Output/ADHD/Idea4/`):
| File | What it shows |
|------|--------------|
| `classification_comparison.png/.csv` | Classifier accuracy across feature sets |
| `cv_heatmap.png` | Cross-validation scores across folds and classifiers |
| `feature_importances.png/.csv` | Which topological features matter most |
| `permutation_test_results.csv` | Permutation test p-values |

---

### Idea 5 — Euler Characteristic & Betti Curves (`Code/Idea5_EulerCharacteristic/`)

Computes Euler characteristic curves and Betti curves from FC-matrix filtrations. Applies functional data analysis (FDA) to compare group-level curve shapes. Also breaks down Betti curves by resting-state network.

Key files:
- `euler_computer.py` — Euler characteristic curves from filtration
- `betti_curve_builder.py` — Betti number curves (B0, B1, B2)
- `fda_analyzer.py` — functional data analysis on the curves
- `idea5_visualizer.py` — all plots and CSVs

Output (`Output/ADHD/Idea5/` and `Output/ABIDE/Idea5/`):
| File | What it shows |
|------|--------------|
| `euler_characteristic_curves.png` | Group-mean Euler characteristic vs. filtration threshold |
| `betti_curves_group.png` | B0/B1/B2 curves by group |
| `network_betti_curves.png` | Betti curves broken down by brain network |
| `fc_matrices_group.png` | Group-averaged FC matrices |
| `fda_results.png` | FDA group comparison on Euler/Betti curves |
| `betti_summary.csv` | Per-subject Betti curve statistics |

---

## Shared Infrastructure (`Core/`)

| File | Role |
|------|------|
| `base_loader.py` | Fetches ADHD-200 or ABIDE data via nilearn |
| `base_masker.py` | Applies a brain atlas (e.g., BASC 122/197/444) to extract ROI time series |
| `base_tda.py` | Shared TDA utilities (ripser wrappers, diagram helpers) |
| `base_experiment.py` | Abstract orchestrator all ideas inherit from |
| `base_visualizer.py` | Shared plotting utilities |
| `utils.py` | `SubjectRecord` dataclass and misc helpers |

## Parameters (`Parameters/`)

Each `params_idea{N}.py` file holds a dataclass with all tunable settings for that idea — atlas choice, number of subjects, homology dimensions, window sizes, classifier hyperparameters, etc. Change settings here rather than in the code.
