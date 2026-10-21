"""Build, load and save controller checkpoints with complete Adam/RNG state."""

import random
import torch
from copy import deepcopy
from pathlib import Path
from .controller import DistributedController
from .plant import build_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/controller.pt"
FORMAT = "dppb-ssm-mlp-v1"


def build_model(config):
    return DistributedController(build_registry(), **deepcopy(config["controller"]))


def load_model(path=DEFAULT_MODEL):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Expected a {FORMAT} checkpoint")
    model = build_model(payload["config"])
    if payload["permanent_edge_order"] != [list(e) for e in model.registry.edges]:
        raise ValueError("Checkpoint line order differs from the physical registry")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def save_checkpoint(path, model, template, *, config, epoch, optimizer=None,
                    history=(), best_guard=None, best_model=None, best_epoch=0):
    payload = dict(template)
    payload.update(format=FORMAT, kind="training_state" if optimizer else "selected",
        config=deepcopy(config), epoch=epoch,
        permanent_edge_order=[list(e) for e in model.registry.edges],
        selection=("last complete epoch with Adam/RNG" if optimizer else
                   "best feasible fixed training guard"),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer else None,
        torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate(),
        history=list(history), best_guard=best_guard,
        best_model_state_dict=best_model, best_epoch=best_epoch)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
