"""
Parameters/params_idea4.py
Configuration for Idea 4 — Topological feature vectors for classification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Idea4Params:
    # --- Data ---
    n_subjects: int = 30
    atlas_name: str = "msdl"
    standardize: str = "zscore_sample"
    detrend: bool = True
    low_pass: float = 0.1
    high_pass: float = 0.01
    t_r: float = 2.0

    # --- TDA computation (reuses Idea1 infrastructure) ---
    max_dimension: int = 2
    max_edge_length: float = 1.0

    # --- Homological dimensions to extract features from ---
    homology_dimensions: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- Feature types ---
    use_persistence_landscapes: bool = True
    use_persistence_images: bool = True
    use_total_persistence: bool = True
    use_persistence_entropy: bool = True

    # --- Persistence landscape ---
    landscape_n_layers: int = 5
    landscape_n_bins: int = 100

    # --- Persistence images ---
    image_n_bins: int = 20
    image_sigma: float = 0.1           # Gaussian bandwidth

    # --- Classifiers ---
    classifiers: List[str] = field(
        default_factory=lambda: ["logistic_regression", "svm", "random_forest"]
    )
    svm_kernel: str = "rbf"
    rf_n_estimators: int = 100

    # --- Cross-validation ---
    cv_folds: int = 5
    cv_stratified: bool = True

    # --- Permutation testing ---
    n_permutations: int = 1000
    alpha: float = 0.05

    # --- Optional neural network (persistence images input) ---
    run_nn_classifier: bool = False
    nn_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    nn_epochs: int = 50

    # --- Dataset ---
    dataset_name: str = "ADHD"   # "ADHD" or "ABIDE" — selects loader + output subfolder
    case_label: str = "ADHD"     # display name for group-1 subjects in plots/logs

    # --- Output ---
    output_dir: Path = Path("Output/Idea4")
    random_seed: int = 42

    # --- Experiments to run ---
    run_tda_only: bool = True
    run_fc_only: bool = True
    run_combined: bool = True
    run_permutation_test: bool = True
    run_feature_importance: bool = True
