"""Distributed contextual-SSM / bounded-MLP performance booster.

Every DGU i runs its own operator

    u^i_t = M_b^i(w^i_t, z^i_t) ⊠ M_p^i(w^i_t, z^i_t),

with w^i_t the IMC residual coordinates owned by DGU i (own triple and owned
lines) and z^i_t its context: the four MAD measurements plus the inputs that
the active neighbors applied at t-1. M_p is a neural_ssm ContextualDeepSSM fed
by w through its recurrent core; the context only enters through the bounded
mixer and gate ports, so it never drives the core as an input. M_b is a
memoryless bounded MLP producing a (1 x s) matrix; s = 1 is the MAD readout.

The parameters are fixed within a rollout, so every weight derived from them
(spectral normalizations, LRU eigenvalues and input matrix) is computed once
in ``reset`` and reused at every step; gradients flow through that single
computation.
"""
from __future__ import annotations

import types
import torch
from torch import nn
from torch.nn import functional
from neural_ssm import ContextualDeepSSM
from neural_ssm.ssm.lti_cells import LRU
from neural_ssm.static_layers.lipschitz_mlps import L2BoundedLinearExact


# Context


FEATURE_SCHEMA = {
    "version": 2,
    "order": ["own_voltage_error", "sum_active_neighbor_voltage_errors",
              "measured_converter_current", "sum_outward_active_incident_line_currents",
              "neighbor_previous_inputs (one slot per permanent incident line)"],
    "units": ["V", "V", "A", "A", "V"],
    "line_sign": "permanent tail:+I; permanent head:-I",
    "inactive": "exact zeros for an inactive DGU or line",
}


def mad_features(physical_state, voltage_errors, registry, topology):
    """The four raw local measurements seen by the MAD direction network."""
    active, edges = set(topology.active_nodes), set(topology.edges)
    positions = {node: k for k, node in enumerate(registry.node_ids)}
    zero = physical_state.new_zeros(())
    rows = []
    for node in registry.node_ids:
        if node not in active:
            rows.append(physical_state.new_zeros(4))
            continue
        own = voltage_errors[positions[node]]
        neighbor_sum = sum((voltage_errors[positions[j]]
                            for j in registry.neighbors(topology, node)), zero)
        converter = physical_state[registry.node_indices(node)[1]]
        outward = sum(((1. if edge[0] == node else -1.)*physical_state[registry.edge_index(edge)]
                       for edge in registry.incident_edges(node) if edge in edges), zero)
        rows.append(torch.stack((own, neighbor_sum, converter, outward)))
    return rows


def neighbor_inputs(u_prev, registry, topology):
    """u^j_{t-1} for every permanent incident line of i, zero where it is absent."""
    positions = {node: k for k, node in enumerate(registry.node_ids)}
    edges = set(topology.edges)
    rows = []
    for node in registry.node_ids:
        slots = [u_prev[positions[edge[1] if edge[0] == node else edge[0]]]
                 if edge in edges else u_prev.new_zeros(())
                 for edge in registry.incident_edges(node)]
        rows.append(torch.stack(slots))
    return rows


def context(physical_state, voltage_errors, u_prev, registry, topology):
    return [torch.cat((own, neighbors)) for own, neighbors in
            zip(mad_features(physical_state, voltage_errors, registry, topology),
                neighbor_inputs(u_prev, registry, topology))]


# Operators


class BoundedMLP(nn.Module):
    """M_b(w, z) in R^{r x s}: spectrally normalized ReLU MLP with a softsign bound.

    Each weight is divided by its exact largest singular value, once per rollout.
    """

    def __init__(self, w_dim, z_dim, r, s, hidden_dim=16, num_layers=3,
                 bound=1.0, z_residual_gain=10.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.r, self.s, self.bound = r, s, float(bound)
        self.z_residual_gain = float(z_residual_gain)
        dims = [w_dim+z_dim] + [hidden_dim]*(num_layers-2) + [r*s]
        self.net = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.z_net = nn.ModuleList((nn.Linear(z_dim, hidden_dim), nn.Linear(hidden_dim, r*s)))
        self.weights = None

    def fix_weights(self):
        self.weights = [[(layer.weight/torch.linalg.svdvals(layer.weight)[0], layer.bias)
                         for layer in layers] for layers in (self.net, self.z_net)]

    @staticmethod
    def _mlp(x, weights):
        for j, (weight, bias) in enumerate(weights):
            x = functional.linear(x, weight, bias)
            if j < len(weights)-1:
                x = torch.relu(x)
        return x

    def forward(self, w, z):
        net, z_net = self.weights
        out = self._mlp(torch.cat((w, z), dim=-1), net) + self.z_residual_gain*self._mlp(z, z_net)
        out = self.bound*out/(1+out.abs())
        return out.reshape(*out.shape[:-1], self.r, self.s)


def _fixed_linear_forward(self, x):
    """L2BoundedLinearExact.forward with the weight rescaled at the start of the rollout."""
    return functional.linear(x, self.rollout_weight, bias=None)


def _fixed_lru_loop(self, input, state, detach_state=True):
    """One LRU step with lambda and B computed at the start of the rollout."""
    if input.shape[1] != 1:
        raise ValueError("The rollout-fixed LRU runs one causal step at a time")
    lambdas, BT, CT, DT = self.rollout
    u = input[:, 0]
    output = ((state @ CT).real + u @ DT).unsqueeze(1)
    following = state*lambdas + u.to(BT.dtype) @ BT
    self.state = following.detach() if detach_state else following
    return output, torch.stack((state, following), dim=1)


def fix_rollout_weights(module):
    """Compute every parameter-derived weight of ``module`` once for a rollout."""
    for m in module.modules():
        if isinstance(m, L2BoundedLinearExact):
            m.rollout_weight = m._effective_weight()
        elif isinstance(m, LRU):
            lambdas, B, C, D = m.set_param()
            m.rollout = (lambdas, B.mT, C.mT, D.mT)
        elif isinstance(m, BoundedMLP):
            m.fix_weights()


class AgentOperator(nn.Module):
    """u^i = M_b^i(w^i, z^i) ⊠ M_p^i(w^i, z^i), one causal step at a time."""

    def __init__(self, w_dim, z_dim, *, feat_dim, mb, ssm):
        super().__init__()
        self.w_dim, self.z_dim = w_dim, z_dim
        self.mp = ContextualDeepSSM(
            w_dim, z_dim, feat_dim, context_modes=("mixer", "gate"),
            d_features=ssm["d_features"], mixer_bound=ssm["mixer_bound"],
            mixer_hidden_dim=ssm["head_hidden"], gate_hidden_dim=ssm["head_hidden"],
            gate_per_channel=False, gamma=None, param=ssm["param"],
            n_layers=ssm["layers"], d_model=ssm["d_model"], d_state=ssm["d_state"],
            ff=ssm["ff"])
        self.mb = BoundedMLP(w_dim, z_dim, 1, feat_dim, hidden_dim=mb["hidden"],
                             num_layers=mb["layers"], bound=mb["bound"],
                             z_residual_gain=mb["z_residual_gain"])
        for m in self.mp.modules():
            if isinstance(m, L2BoundedLinearExact):
                m.forward = types.MethodType(_fixed_linear_forward, m)
            elif isinstance(m, LRU):
                m.forward_loop = types.MethodType(_fixed_lru_loop, m)

    def reset(self):
        self.mp.reset()

    def forward(self, w, z):
        w, z = w.reshape(1, 1, -1), z.reshape(1, 1, -1)
        v, _ = self.mp(w, z, mode="loop", reset_state=False, detach_state=False)
        return torch.einsum("btrs,bts->btr", self.mb(w, z), v).reshape(())


class DistributedController(nn.Module):
    """One independent AgentOperator per permanent DGU.

    Inactive DGUs are neither stepped nor actuated. ``reset`` starts a rollout:
    it fixes the derived weights from the current parameters and clears every
    recurrent state. ``transition`` clears the recurrent state of DGUs that plug
    in or out, while every surviving DGU keeps its own state across the event.
    """

    mad_feature_schema = FEATURE_SCHEMA

    def __init__(self, registry, *, feat_dim=1, output_scale=1.0, mb=None, ssm=None):
        super().__init__()
        self.registry = registry
        self.output_scale = float(output_scale)
        self.w_indices = {i: tuple(registry.owned_indices(i)) for i in registry.node_ids}
        self.agents = nn.ModuleDict({
            str(i): AgentOperator(len(self.w_indices[i]), 4+len(registry.incident_edges(i)),
                                  feat_dim=feat_dim, mb=mb, ssm=ssm)
            for i in registry.node_ids})
        # double() leaves the complex LRU parameters at complex64.
        self.double()
        for parameter in self.parameters():
            if parameter.is_complex():
                parameter.data = parameter.data.to(torch.complex128)

    def reset(self):
        fix_rollout_weights(self)
        for agent in self.agents.values():
            agent.reset()

    def transition(self, old_topology, new_topology):
        changed = set(old_topology.active_nodes) ^ set(new_topology.active_nodes)
        for i in changed:
            self.agents[str(i)].reset()
        return sorted(changed)

    def step(self, ehat, voltage_errors, physical_state, u_prev, topology):
        """Return (u, context rows) for the current sample."""
        contexts = context(physical_state, voltage_errors, u_prev, self.registry, topology)
        active = set(topology.active_nodes)
        u = torch.stack([
            self.agents[str(i)](ehat[list(self.w_indices[i])], contexts[k])
            if i in active else ehat.new_zeros(())
            for k, i in enumerate(self.registry.node_ids)])
        return self.output_scale*u, contexts

    def core_parameters(self):
        """Recurrent DeepSSM core of every M_p (trained at the full learning rate)."""
        return [p for agent in self.agents.values() for p in agent.mp.core.parameters()]
