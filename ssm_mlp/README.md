# Contextual SSM-MLP performance booster

Alternate distributed controller for the six-DGU PnP DC microgrid of [`../ren_mad`](../ren_mad).
The plant, the primary controller, the IMC residual reconstruction, the scenarios and the training loss are unchanged.
The controller replaces the certified RENs and the MAD readout.

## Controller

Each DGU *i* runs its own operator

u<sup>i</sup><sub>t</sub> = M<sub>b</sub><sup>i</sup>(w<sup>i</sup><sub>t</sub>, z<sup>i</sup><sub>t</sub>) ⊠ M<sub>p</sub><sup>i</sup>(w<sup>i</sup><sub>t</sub>, z<sup>i</sup><sub>t</sub>)

- **w<sup>i</sup>**: the IMC residual coordinates owned by DGU *i* (own voltage, current and integrator, plus the lines it owns).
- **z<sup>i</sup>** (context): the four measurements the MAD used (own voltage error, sum of active neighbors' voltage errors, converter current, sum of outward line currents), plus u<sup>j</sup><sub>t-1</sub> for every permanent incident line, zero when that line is inactive.
- **M<sub>p</sub>**: `neural_ssm.ContextualDeepSSM` driven by w. The context enters only through the bounded `mixer` and `gate` ports, never as an input to the recurrent core.
- **M<sub>b</sub>**: a bounded MLP (spectral normalization, softsign) producing a 1 × s matrix. With `feat_dim = 1` (the default) it plays the role of the MAD readout.

No gains are computed and no network certificate is checked. Agents do not exchange ℓ2 signals; they share only context.
At a plug-in or unplug event, only the DGUs that join or leave have their recurrent state reset.

Architecture and learning rates are set in [`src/settings.py`](src/settings.py).

## Requirements

On top of `numpy`, `torch` and `matplotlib`:

```bash
python -m pip install "neural_ssm @ git+https://github.com/DecodEPFL/SSM.git@32fa0e468b818593b7cb69d5483587759de3ae16"
```

## Usage

Train a new controller from scratch:

```bash
python main_PnP_training.py --epochs 150 --budget-hours 2.5
```

Resume a run, or start a new optimizer from a checkpoint:

```bash
python main_PnP_training.py --resume models/training_<UTC>/last.pt --epochs 200
python main_PnP_training.py --model models/training_<UTC>/selected_controller.pt --epochs 50
```

Generate the figures from a trained controller:

```bash
python main_PnP_validation.py --model models/training_<UTC>/selected_controller.pt --controls
```

A fresh run is gated against the primary-controller baseline. Its random initialization is never used as the parent.
