"""
main.py
Top-level CLI entry point — run any or all of the 5 Ideas.

Usage
-----
# Run a specific idea:
    python main.py --idea 1
    python main.py --idea 5

# Run all ideas sequentially:
    python main.py --all

# Run Idea 5 (cheapest) first as a sanity check:
    python main.py --idea 5
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_idea(idea_number: int) -> None:
    if idea_number == 1:
        from Parameters.params_idea1 import Idea1Params
        from Code.Idea1_PH_FC.run_idea1 import Idea1Orchestrator
        Idea1Orchestrator(Idea1Params()).execute()

    elif idea_number == 2:
        from Parameters.params_idea2 import Idea2Params
        from Code.Idea2_Mapper.run_idea2 import Idea2Orchestrator
        Idea2Orchestrator(Idea2Params()).execute()

    elif idea_number == 3:
        from Parameters.params_idea3 import Idea3Params
        from Code.Idea3_SlidingWindow.run_idea3 import Idea3Orchestrator
        Idea3Orchestrator(Idea3Params()).execute()

    elif idea_number == 4:
        from Parameters.params_idea4 import Idea4Params
        from Code.Idea4_Classification.run_idea4 import Idea4Orchestrator
        Idea4Orchestrator(Idea4Params()).execute()

    elif idea_number == 5:
        from Parameters.params_idea5 import Idea5Params
        from Code.Idea5_EulerCharacteristic.run_idea5 import Idea5Orchestrator
        Idea5Orchestrator(Idea5Params()).execute()

    else:
        print(f"Unknown idea number: {idea_number}. Choose 1-5.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TDA on fMRI/ADHD Data — run experiments."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--idea", type=int, choices=[1, 2, 3, 4, 5],
        help="Run a specific idea (1-5).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all 5 ideas sequentially.",
    )
    args = parser.parse_args()

    if args.all:
        # Recommended order: cheapest first
        for idea_number in [5, 1, 4, 3, 2]:
            print(f"\n{'='*60}")
            print(f"  Running Idea {idea_number}")
            print(f"{'='*60}")
            run_idea(idea_number)
    else:
        run_idea(args.idea)


if __name__ == "__main__":
    main()
