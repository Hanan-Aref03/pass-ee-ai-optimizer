"""Neural network model for PASS and conventional power regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A small residual block for tabular regression."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters."""

    hidden_dim: int = 256
    num_blocks: int = 4
    dropout: float = 0.12
    feasibility_head: bool = False
    feasibility_conditioning: str = "hidden"


class PassDnnRegressor(nn.Module):
    """Shared backbone with separate heads for PA positions and waveguide powers."""

    def __init__(
        self,
        input_dim: int,
        mode: str,
        position_dim: int,
        power_dim: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.position_dim = position_dim
        self.power_dim = power_dim
        self.config = config or ModelConfig()

        width = self.config.hidden_dim
        feasibility_conditioning = self.config.feasibility_conditioning.strip().lower()
        self.feasibility_conditioning = feasibility_conditioning
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, self.config.dropout) for _ in range(self.config.num_blocks)]
        )
        self.post_norm = nn.LayerNorm(width)

        if self.mode == "pass":
            if feasibility_conditioning not in {"hidden", "input_output"}:
                raise ValueError(
                    "ModelConfig.feasibility_conditioning must be 'hidden' or 'input_output'."
                )

            self.position_head = nn.Sequential(
                nn.Linear(width, width // 2),
                nn.SiLU(),
                nn.Linear(width // 2, position_dim),
                nn.Tanh(),
            )
            # Input = backbone features + predicted positions so power learns to be
            # jointly consistent with positions (breaks the chicken-and-egg collapse).
            self.power_head = nn.Sequential(
                nn.Linear(width + position_dim, width // 2),
                nn.SiLU(),
                nn.Linear(width // 2, power_dim),
                # No Sigmoid — clamped in forward() to prevent saturation collapse
            )
            self.feasibility_head = (
                nn.Sequential(
                    nn.Linear(width, width // 2),
                    nn.SiLU(),
                    nn.Linear(width // 2, 1),
                )
                if self.config.feasibility_head
                else None
            )

            if self.feasibility_head is not None and feasibility_conditioning == "input_output":
                self.feasibility_head = nn.Sequential(
                    nn.Linear(input_dim + position_dim + power_dim, width),
                    nn.SiLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(width, width // 2),
                    nn.SiLU(),
                    nn.Linear(width // 2, 1),
                )
        else:
            self.position_head = None
            self.feasibility_head = None
            self.power_head = nn.Sequential(
                nn.Linear(width, width // 2),
                nn.SiLU(),
                nn.Linear(width // 2, power_dim),
            )

    def _feasibility_input(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        powers: torch.Tensor,
    ) -> torch.Tensor:
        if self.feasibility_head is None:
            raise ValueError("Feasibility head is disabled for this model.")
        if self.mode != "pass":
            raise ValueError("Feasibility is only implemented for PASS mode.")

        if self.feasibility_conditioning == "hidden":
            raise ValueError("Hidden-state feasibility input requires features from forward().")

        if self.feasibility_conditioning != "input_output":
            raise ValueError(
                f"Unsupported feasibility conditioning mode: {self.feasibility_conditioning}"
            )
        return torch.cat([x, positions, powers], dim=1)

    def feasibility_from_candidate(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        powers: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the feasibility head on an arbitrary candidate output."""

        if self.feasibility_head is None:
            raise ValueError("Feasibility head is disabled for this model.")

        if self.feasibility_conditioning == "hidden":
            raise ValueError(
                "Candidate feasibility evaluation requires feasibility_conditioning='input_output'."
            )

        feasibility_input = self._feasibility_input(x, positions, powers)
        return self.feasibility_head(feasibility_input)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_layer(x)
        h = self.blocks(h)
        h = self.post_norm(h)

        if self.mode == "pass":
            positions = self.position_head(h)
            power_input = torch.cat([h, positions], dim=1)
            output = {
                "positions": positions,
                "powers": torch.clamp(self.power_head(power_input), 0.0, 1.0),
            }
            if self.feasibility_head is not None:
                if self.feasibility_conditioning == "hidden":
                    feasibility_input = h
                else:
                    feasibility_input = self._feasibility_input(
                        x,
                        output["positions"],
                        output["powers"],
                    )
                output["feasibility_logit"] = self.feasibility_head(feasibility_input)
                output["feasibility_prob"] = torch.sigmoid(output["feasibility_logit"])
            return output

        return {"powers": torch.clamp(self.power_head(h), 0.0, 1.0)}
