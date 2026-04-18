"""
Parameters/params_idea3.py
Configuration for Idea 3 — Sliding-window TDA on BOLD time series.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Idea3Params:
    # --- Data ---
    n_subjects: int = 30
    atlas_name: str = "msdl"
    standardize: str = "zscore_sample"
    detrend: bool = True
    low_pass: float = 0.1
    high_pass: float = 0.01
    t_r: float = 2.0

    # --- Signal extraction ---
    use_mean_bold: bool = True          # embed mean BOLD across all ROIs
    roi_index: int = 0                  # used if use_mean_bold=False

    # --- Sliding window ---
    window_length: int = 30             # in TRs (= 60 s at TR=2)
    step_size: int = 1

    # --- Window size sweep ---
    window_lengths_sweep: List[int] = field(
        default_factory=lambda: [15, 20, 30, 40, 50]
    )

    # --- Takens embedding (alternative) ---
    use_takens: bool = False
    takens_dimension: int = 10
    takens_stride: int = 1

    # --- Persistent homology per window ---
    max_dimension: int = 1              # H0 + H1 only (H2 too slow per window)
    max_edge_length: float = 2.0

    # --- Loop score aggregation ---
    # 'sum' = total H1 persistence, 'max' = longest bar, 'mean' = mean bar length
    loop_score_aggregation: str = "sum"

    # --- Autocorrelation ---
    autocorr_max_lag: int = 20

    # --- Statistics ---
    n_permutations: int = 1000
    alpha: float = 0.05

    # --- Dataset ---
    dataset_name: str = "ADHD"   # "ADHD" or "ABIDE" — selects loader + output subfolder
    case_label: str = "ADHD"     # display name for group-1 subjects in plots/logs

    # --- Output ---
    output_dir: Path = Path("Output/Idea3")
    random_seed: int = 42

    # --- Experiments to run ---
    run_temporal_h1: bool = True
    run_loop_score_biomarker: bool = True
    run_window_size_sweep: bool = True
    run_autocorrelation: bool = True
