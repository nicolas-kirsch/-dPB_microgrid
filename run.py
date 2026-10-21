#!/usr/bin/env python3
"""Run training or validation of either controller from the repository root.

    python run.py train --method ren --fresh --init-seed 1 --output-root /data/runs
    python run.py validate --method ssm --model /data/runs/ssm_init1_<UTC>/selected_controller.pt --output fig/

All arguments after the method are forwarded unchanged to the method's own
main_PnP_training.py / main_PnP_validation.py (see their --help).
"""
import argparse
import sys
sys.dont_write_bytecode = True
from pathlib import Path

METHODS = {"ren": "ren_mad", "ssm": "ssm_mlp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=("train", "validate"))
    parser.add_argument("--method", required=True, choices=METHODS)
    args, forwarded = parser.parse_known_args()
    # Both folders name their package `src`, so only one method can be imported per process.
    sys.path.insert(0, str(Path(__file__).resolve().parent / METHODS[args.method]))
    sys.argv = [f"{METHODS[args.method]}/main_PnP_{'training' if args.task == 'train' else 'validation'}.py",
                *forwarded]
    if args.task == "train":
        from src.training import main as task
    else:
        import importlib
        task = importlib.import_module("main_PnP_validation").main
    task(forwarded)


if __name__ == "__main__":
    main()
