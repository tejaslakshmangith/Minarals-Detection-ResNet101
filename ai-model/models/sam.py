"""SAM (Sharpness Aware Minimisation) optimiser wrapper.

Reference:
    Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
    Generalization", ICLR 2021.  https://arxiv.org/abs/2010.01412

Usage::

    base_opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer = SAM(model.parameters(), base_opt, rho=0.05)

    # First forward-backward pass
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Second forward-backward pass (same mini-batch)
    criterion(model(x), y).backward()
    optimizer.second_step(zero_grad=True)
"""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimisation wrapper around any base optimiser.

    Args:
        params: Model parameters (same as for any ``torch.optim.Optimizer``).
        base_optimizer: **Class** (not instance) of the underlying optimiser,
            e.g. ``torch.optim.Adam``.
        rho: Neighbourhood size for the perturbation step (default: 0.05).
        adaptive: Use adaptive SAM (ASAM) which scales the perturbation by the
            parameter magnitude (default: ``False``).
        **kwargs: Keyword arguments forwarded to *base_optimizer*.
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        defaults = {"rho": rho, "adaptive": adaptive}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # Keep param_groups in sync with the base optimiser
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascend to the local maximum (perturb weights toward sharp region)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Store original weights
                self.state[p]["old_p"] = p.data.clone()
                # Compute perturbation
                e_w = (p * p if group["adaptive"] else torch.ones_like(p)) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original weights and apply the base optimiser update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore original weights
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Single-step interface required by the ``Optimizer`` ABC.

        Prefer the explicit ``first_step`` / ``second_step`` API.  This
        method executes both steps with a closure that re-evaluates the loss.
        """
        if closure is None:
            raise ValueError(
                "SAM requires a closure when using the .step() interface. "
                "Use first_step / second_step instead."
            )
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step()
        return loss

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else torch.ones_like(p)) * p.grad).norm(p=2).to(shared_device)                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
