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


def run_idea(idea_number: int, dataset_name: str = "ADHD") -> None:
    case_label = "Autism" if dataset_name == "ABIDE" else dataset_name

    if idea_number == 1:
        from Parameters.params_idea1 import Idea1Params
        from Code.Idea1_PH_FC.run_idea1 import Idea1Orchestrator
        params = Idea1Params()
        params.dataset_name = dataset_name
        params.case_label = case_label
        Idea1Orchestrator(params).execute()

    elif idea_number == 2:
        from Parameters.params_idea2 import Idea2Params
        from Code.Idea2_Mapper.run_idea2 import Idea2Orchestrator
        params = Idea2Params()
        params.dataset_name = dataset_name
        params.case_label = case_label
        Idea2Orchestrator(params).execute()

    elif idea_number == 3:
        from Parameters.params_idea3 import Idea3Params
        from Code.Idea3_SlidingWindow.run_idea3 import Idea3Orchestrator
        params = Idea3Params()
        params.dataset_name = dataset_name
        params.case_label = case_label
        Idea3Orchestrator(params).execute()

    elif idea_number == 4:
        from Parameters.params_idea4 import Idea4Params
        from Code.Idea4_Classification.run_idea4 import Idea4Orchestrator
        params = Idea4Params()
        params.dataset_name = dataset_name
        params.case_label = case_label
        Idea4Orchestrator(params).execute()

    elif idea_number == 5:
        from Parameters.params_idea5 import Idea5Params
        from Code.Idea5_EulerCharacteristic.run_idea5 import Idea5Orchestrator
        params = Idea5Params()
        params.dataset_name = dataset_name
        params.case_label = case_label
        Idea5Orchestrator(params).execute()

    else:
        print(f"Unknown idea number: {idea_number}. Choose 1-5.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TDA on fMRI data — run experiments on ADHD or ABIDE dataset."
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
    parser.add_argument(
        "--dataset", choices=["adhd", "abide"], default="adhd",
        help="Dataset to use: 'adhd' (default) or 'abide'. "
             "Outputs go to Output/ADHD/ or Output/ABIDE/.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset.upper()

    if args.all:
        # Recommended order: cheapest first
        for idea_number in [5, 1, 4, 3, 2]:
            print(f"\n{'='*60}")
            print(f"  Running Idea {idea_number}  [{dataset_name}]")
            print(f"{'='*60}")
            run_idea(idea_number, dataset_name=dataset_name)
    else:
        run_idea(args.idea, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
