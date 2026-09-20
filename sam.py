"""
sam.py
Sharpness-Aware Minimization (SAM) optimizer.

SAM seeks parameters that lie in flat regions of the loss landscape rather
than sharp minima. Flat minima are more robust to perturbation — and
quantization is exactly a perturbation of the weights. This is why SAM is
worth comparing against Adam specifically for a tinyML/quantization project:
the hypothesis to test is "does a SAM-trained model lose less accuracy after
quantization than an Adam-trained model at the same bit-width?"

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization" (2020). This is a minimal, dependency-free
implementation compatible with any base optimizer.
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **base_kwargs):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Climb to the local worst-case point within radius rho."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Step down using the gradient computed at the perturbed point."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore original weights
        self.base_optimizer.step()  # actual update using perturbed gradient
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm

    def step(self, closure=None):
        """SAM requires a closure that re-evaluates loss+backward,
        since it needs two forward/backward passes per step."""
        assert closure is not None, "SAM requires a closure that recomputes loss"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups