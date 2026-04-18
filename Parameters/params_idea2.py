"""
Parameters/params_idea2.py
Configuration for Idea 2 — Mapper on brain state space.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Idea2Params:
    # --- Data ---
    n_subjects: int = 30
    atlas_name: str = "msdl"
    standardize: str = "zscore_sample"
    detrend: bool = True
    low_pass: float = 0.1
    high_pass: float = 0.01
    t_r: float = 2.0

    # --- Dimensionality reduction before Mapper ---
    pca_n_components: int = 10         # reduce to before lens application

    # --- Lens function ---
    # Options: 'pca', 'variance', 'time'
    lens_function: str = "pca"
    pca_component: int = 0             # which PC to use as primary lens (0-indexed)
    variance_window: int = 10          # rolling window for VarianceLens

    # --- Cover ---
    n_intervals: int = 10
    overlap_fraction: float = 0.5

    # --- Clustering within cover sets ---
    # AgglomerativeClustering linkage
    clusterer_linkage: str = "single"
    n_clusters: int = 2

    # --- Parameter stability sweep ---
    n_intervals_range: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    overlap_range: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6])

    # --- Dataset ---
    dataset_name: str = "ADHD"   # "ADHD" or "ABIDE" — selects loader + output subfolder
    case_label: str = "ADHD"     # display name for group-1 subjects in plots/logs

    # --- Output ---
    output_dir: Path = Path("Output/Idea2")
    random_seed: int = 42

    # --- Experiments to run ---
    run_group_topology: bool = True
    run_population_mapper: bool = True
    run_parameter_stability: bool = True
