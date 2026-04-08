"""
Core/base_experiment.py
Orchestration base class inherited by each Idea's entry-point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from Core.base_loader import ADHDDataLoader
from Core.base_masker import ROIMasker
from Core.utils import OutputManager, SubjectRecord

# Root of the project — resolved relative to this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_ROOT = _PROJECT_ROOT / "Output"


class BaseIdeaOrchestrator(ABC):
    """
    Template-method base class for each Idea's top-level orchestrator.

    Subclasses must implement run_all_experiments(records).
    Call execute() to run the full pipeline end-to-end.
    """

    def __init__(self, params, n_subjects: int = 30,
                 idea_name: Optional[str] = None):
        self.params = params
        self.loader = ADHDDataLoader(n_subjects=n_subjects)
        idea = idea_name or self.__class__.__name__.replace("Orchestrator", "")
        self.output_manager = OutputManager(
            idea_name=idea,
            base_output_dir=_OUTPUT_ROOT,
        )

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """Full pipeline: load → mask → run experiments."""
        records = self.load_and_mask()
        self.run_all_experiments(records)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_and_mask(self, atlas_name: Optional[str] = None) -> List[SubjectRecord]:
        """Fetch subjects and extract ROI time series."""
        records = self.loader.fetch()
        atlas = atlas_name or getattr(self.params, "atlas_name", "msdl")
        masker = ROIMasker(
            atlas_name=atlas,
            standardize=getattr(self.params, "standardize", "zscore_sample"),
            detrend=getattr(self.params, "detrend", True),
            low_pass=getattr(self.params, "low_pass", 0.1),
            high_pass=getattr(self.params, "high_pass", 0.01),
            t_r=getattr(self.params, "t_r", 2.0),
        )
        self._masker = masker  # store so subclasses can access roi_labels_
        records = masker.fit_transform(records)
        return records

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run_all_experiments(self, records: List[SubjectRecord]) -> None:
        """Subclasses implement all experimental logic here."""
