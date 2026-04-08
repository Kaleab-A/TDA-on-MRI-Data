"""
Parameters/params_idea1.py
Configuration for Idea 1 — Persistent Homology on FC matrices.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Idea1Params:
    # --- Data ---
    n_subjects: int = 30
    atlas_name: str = "msdl"           # 'msdl' | 'schaefer100' | 'schaefer200'
    standardize: str = "zscore_sample"
    detrend: bool = True
    low_pass: float = 0.1
    high_pass: float = 0.01
    t_r: float = 2.0

    # --- FC computation ---
    correlation_kind: str = "correlation"   # nilearn ConnectivityMeasure kind

    # --- Distance matrix ---
    # D = 1 - |FC|, clipped to [0, 1], diagonal set to 0
    distance_formula: str = "1_minus_abs"

    # --- Vietoris-Rips ---
    max_dimension: int = 2              # computes H0, H1, H2
    max_edge_length: float = 1.0
    backend: str = "ripser"             # 'ripser' (recommended) or 'giotto'

    # --- Persistence distance metrics ---
    persistence_metrics: List[str] = field(
        default_factory=lambda: ["wasserstein", "bottleneck"]
    )
    wasserstein_order: int = 1

    # --- Group comparison ---
    n_permutations: int = 1000
    alpha: float = 0.05

    # --- Atlas scale sweep ---
    # Used by run_atlas_scale experiment
    atlas_names_sweep: List[str] = field(
        default_factory=lambda: ["msdl", "schaefer100", "schaefer200"]
    )

    # --- Output ---
    output_dir: Path = Path("Output/Idea1")
    random_seed: int = 42

    # --- Experiments to run (toggle on/off) ---
    run_group_comparison: bool = True
    run_h0_vs_h1: bool = True
    run_atlas_scale: bool = True
    run_subtype_analysis: bool = True
