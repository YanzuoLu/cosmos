from dataclasses import dataclass

import numpy as np
import torch

from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.Tensor
    commit_loss: torch.FloatTensor | None = None


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            zeros = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)
            self.var = zeros
            self.std = zeros

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample: torch.Tensor, dims: tuple[int, ...] = (1, 2, 3)) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        return 0.5 * torch.sum(np.log(2.0 * np.pi) + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self) -> torch.Tensor:
        return self.mean


class AutoencoderMixin:
    def enable_tiling(self):
        if not hasattr(self, "use_tiling"):
            raise NotImplementedError(f"Tiling is not implemented for {self.__class__.__name__}.")
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False

    def enable_slicing(self):
        if not hasattr(self, "use_slicing"):
            raise NotImplementedError(f"Slicing is not implemented for {self.__class__.__name__}.")
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False
