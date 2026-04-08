"""
Parameters/params_idea5.py
Configuration for Idea 5 — Euler characteristic and Betti curves.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Idea5Params:
    # --- Data ---
    n_subjects: int = 40
    atlas_name: str = "msdl"
    standardize: str = "zscore_sample"
    detrend: bool = True
    low_pass: float = 0.1
    high_pass: float = 0.01
    t_r: float = 2.0

    # --- Filtration grid ---
    epsilon_min: float = 0.0
    epsilon_max: float = 1.0
    n_epsilon_steps: int = 100

    # --- Homological dimensions for χ = β0 - β1 + β2 ---
    homology_dimensions: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- Betti curve analysis ---
    betti_area_dim: int = 1             # compute area under β_{dim} curve

    # --- Network-specific analysis ---
    # MSDL atlas network name substrings (matched case-insensitively against roi_labels_).
    # Verified against actual MSDL 39-ROI labels:
    # "DMN" → 'L DMN', 'Med DMN', 'Front DMN', 'R DMN'  (4 ROIs)
    # "Aud" → 'L Aud', 'R Aud'                           (2 ROIs, auditory)
    # "IPS" → 'L IPS', 'R IPS', 'L Ant IPS', 'R Ant IPS'(4 ROIs, dorsal attention)
    network_names: List[str] = field(
        default_factory=lambda: ["DMN", "Aud", "IPS"]
    )

    # --- Severity correlation ---
    severity_correlation_method: str = "spearman"   # 'pearson' | 'spearman'

    # --- Functional Data Analysis (skfda) ---
    run_fda: bool = True
    fda_n_basis: int = 10              # number of B-spline basis functions
    fda_smoothing_parameter: float = 1e-2
    fda_n_permutations: int = 500      # for functional group test

    # --- Statistics ---
    n_permutations: int = 1000
    alpha: float = 0.05

    # --- Output ---
    output_dir: Path = Path("Output/Idea5")
    random_seed: int = 42

    # --- Experiments to run ---
    run_group_ec_curves: bool = True
    # Severity correlation requires >=3 ADHD subjects with valid scores.
    # nilearn ADHD-200 subset has only 2/13 ADHD subjects with severity data
    # (conn_adhd / dsm_iv_tot / dsm_iv_inatt all share the same 4 valid rows).
    # Set to True if using a dataset with fuller clinical records.
    run_betti_vs_severity: bool = False
    run_network_analysis: bool = True
    run_fda_analysis: bool = True
