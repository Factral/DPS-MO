import torch
import torch.nn.functional as F
import math
import deepinv as dinv

# Kept for compatibility with the rest of your repo.
from .gaussian_diffusion import extract_and_expand  # noqa: F401


class MeasurementOptimizer:
    """Measurement Optimizer (Algorithm 1 in the MO paper).

    The plain SGLD step of the paper (line 5) is **∇‖y − A(x)‖²**.  Here, we
    *replace* that gradient by the proximal operator of the same quadratic
    fidelity (see Remark 3.1 of the DeepInv docs).  Numerically this is often
    more stable than back‑prop **and** lets us avoid the autograd graph.

    All public arguments / return types are *unchanged* with respect to your
    original implementation.
    """

    def __init__(self, operator_A, N_sgld_steps: int, sgld_lr: float, device):
        
        filter = dinv.physics.blur.gaussian_blur(sigma=(2, 1), angle=0.3)

        self.operator_A =  dinv.physics.BlurFFT(
                        img_size=(3, 256, 256),
                        filter=filter,
                        device=device
                    ) #operator_A  # a DeepInv physics operator
        self.N_sgld_steps = N_sgld_steps  # N in the paper
        self.sgld_lr = sgld_lr          # η in the paper
        self.device = device

        # ℓ2 data‑fidelity prox ϕ(z)=½‖y−Az‖² provided by DeepInv.
        self.fidelity = dinv.optim.L2()

    @torch.no_grad()
    def run(
        self,
        denoiser_model,
        y_measurement: torch.Tensor,
        x_init: torch.Tensor,         # x_init in Alg. 1 line 1
        current_t_tensor: torch.Tensor,  # diffusion timestep t
        diffusion_sampler,
    ):
        # ------------------------------------------------------------------
        # Lines 2‑7: Stochastic Gradient Langevin Dynamics (with prox‑step)
        # ------------------------------------------------------------------
        x_t = x_init.clone().to(self.device)  # line 2
        sqrt_2lr = math.sqrt(2 * self.sgld_lr)
        last_loss = torch.tensor(0.0, device=self.device)
        y_measurement = self.operator_A(x_init)


        for _ in range(self.N_sgld_steps):  # line 4
            #   x ← prox_{η‖y−A·‖²}(x)            (gradient surrogate)
            x_t = self.fidelity.prox(
                x_t,
                y_measurement,
                self.operator_A,
                gamma=self.sgld_lr,
            )   

            # monitor loss (optional)
            y_pred = self.operator_A(x_t)
            last_loss = F.mse_loss(y_pred, y_measurement, reduction="mean")

            #   + √(2η) ε                          (Langevin noise)
            x_t.add_(sqrt_2lr * torch.randn_like(x_t, device=self.device))
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Lines 9‑10: Project back to the diffusion prior manifold
        # ------------------------------------------------------------------
        noise_for_xt = torch.randn_like(x_t, device=self.device)
        x_t_prime = diffusion_sampler.q_sample(
            x_start=x_t,
            t=current_t_tensor,
            noise=noise_for_xt,
        )

        out = diffusion_sampler.p_mean_variance(
            denoiser_model, x_t_prime, current_t_tensor
        )
        x_hat_0 = out["pred_xstart"].detach()  # line 10
        # ------------------------------------------------------------------

        return x_hat_0, float(last_loss)  # line 11
