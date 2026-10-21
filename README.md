# Distributed performance boosting for PnP DC microgrids

Two distributed boosting controllers on the same six-DGU DC microgrid, with the same IMC residual reconstruction and the same training scenarios:

- [`ren_mad/`](ren_mad): the published controller. It uses gain-certified local RENs with a bounded MAD readout, and the network is re-certified at every PnP event. See [ren_mad/README.md](ren_mad/README.md).
- [`ssm_mlp/`](ssm_mlp): the alternate controller. Each DGU runs its own contextual SSM ⊠ bounded MLP, with no gain computation. See [ssm_mlp/README.md](ssm_mlp/README.md).

Each folder is self-contained; run its scripts from inside that folder.

## Seed comparison runs

`run.py` runs either method from the repository root; everything after `--method` is forwarded to that method's own script:

```bash
python run.py train --method ren --init-seed 1 --epochs 152 --budget-hours 100 --output-root /data/runs
python run.py train --method ssm --init-seed 1 --epochs 152 --budget-hours 100 --output-root /data/runs
python run.py validate --method ssm --model /data/runs/ssm_init1_<UTC>/selected_controller.pt --output /data/runs/ssm_init1_<UTC>/figure
```

`--init-seed` only seeds the controller initialization (and, for REN, implies `--fresh`). The training scenarios, their order and the guard keep the seeds in `settings.py`, so runs that differ in `--init-seed` differ only in their initial weights.

Each run gets its own new directory `<method>_init<seed>_<UTC>` under `--output-root` (default: the method's `models/`). An existing directory is never reused. The directory contains:

| File | Content |
|---|---|
| `selected_controller.pt` | controller parameters of the best guard epoch |
| `last.pt` | last complete epoch, with Adam and RNG state for `--resume` |
| `train.log` | full console output, including tracebacks |
| `run.json` | command line, configuration, init/data seeds, parameter count, git commit, host, versions |
| `history.json` | per epoch: mean training loss, guard score, learning rates, wall time (updated every epoch) |
| `summary.json` | final report, including total `wall_seconds` and per-epoch `epoch_seconds` |

Training uses one CPU thread (`torch.set_num_threads(1)`), so one CPU per job is enough and wall times are comparable between jobs on the same hardware. A SIGTERM, such as a preempted job, rolls back the unfinished epoch and saves the checkpoints. Resume with `--resume <dir>/last.pt`; this writes a new directory.
