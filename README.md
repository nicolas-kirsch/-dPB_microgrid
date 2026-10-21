# Distributed performance boosting for PnP DC microgrids

Two distributed boosting controllers on the same six-DGU DC microgrid, with the same IMC residual reconstruction and the same training scenarios:

- [`ren_mad/`](ren_mad): the published controller. It uses gain-certified local RENs with a bounded MAD readout, and the network is re-certified at every PnP event. See [ren_mad/README.md](ren_mad/README.md).
- [`ssm_mlp/`](ssm_mlp): the alternate controller. Each DGU runs its own contextual SSM ⊠ bounded MLP, with no gain computation. See [ssm_mlp/README.md](ssm_mlp/README.md).

Each folder is self-contained; run its scripts from inside that folder.
