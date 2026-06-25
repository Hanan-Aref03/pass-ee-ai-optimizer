"""Column schemas and physical constants for the PASS DNN pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence, Tuple

INPUT_COLUMNS: Tuple[str, ...] = (
    "user1_x",
    "user1_y",
    "user2_x",
    "user2_y",
    "user3_x",
    "user3_y",
    "QoS_R",
)

PASS_POSITION_COLUMNS: Tuple[str, ...] = (
    "PA1x1",
    "PA2x1",
    "PA3x1",
    "PA1x2",
    "PA2x2",
    "PA3x2",
    "PA1x3",
    "PA2x3",
    "PA3x3",
)

PASS_POWER_COLUMNS: Tuple[str, ...] = ("power_wg1", "power_wg2", "power_wg3")

PASS_OUTPUT_COLUMNS: Tuple[str, ...] = PASS_POSITION_COLUMNS + PASS_POWER_COLUMNS

CONVENTIONAL_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "power_ant1",
    "power_ant2",
    "power_ant3",
)


@dataclass(frozen=True)
class SystemConfig:
    """Physical constants shared by the MATLAB simulator and the DNN starter."""

    area_side_m: float = 10.0
    transmitter_height_m: float = 5.0
    carrier_frequency_thz: float = 0.3
    power_budget_w: float = 0.1
    circuit_power_w: float = 0.01
    speed_of_light_m_s: float = 3e8
    num_users: int = 3
    num_waveguides: int = 3
    num_pinchers: int = 3

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light_m_s / (self.carrier_frequency_thz * 1e12)

    @property
    def min_spacing_m(self) -> float:
        return self.wavelength_m / 2.0

    @property
    def position_bound_m(self) -> float:
        return self.area_side_m / 2.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["wavelength_m"] = self.wavelength_m
        payload["min_spacing_m"] = self.min_spacing_m
        payload["position_bound_m"] = self.position_bound_m
        return payload


@dataclass(frozen=True)
class DatasetSchema:
    """Describes one training target schema supported by the project."""

    mode: str
    input_columns: Tuple[str, ...]
    output_columns: Tuple[str, ...]
    position_columns: Tuple[str, ...]
    power_columns: Tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return len(self.input_columns)

    @property
    def output_dim(self) -> int:
        return len(self.output_columns)

    @property
    def position_dim(self) -> int:
        return len(self.position_columns)

    @property
    def power_dim(self) -> int:
        return len(self.power_columns)


PASS_SCHEMA = DatasetSchema(
    mode="pass",
    input_columns=INPUT_COLUMNS,
    output_columns=PASS_OUTPUT_COLUMNS,
    position_columns=PASS_POSITION_COLUMNS,
    power_columns=PASS_POWER_COLUMNS,
)

CONVENTIONAL_SCHEMA = DatasetSchema(
    mode="conventional",
    input_columns=INPUT_COLUMNS,
    output_columns=CONVENTIONAL_OUTPUT_COLUMNS,
    position_columns=tuple(),
    power_columns=CONVENTIONAL_OUTPUT_COLUMNS,
)


def get_schema(mode: str) -> DatasetSchema:
    mode_key = mode.strip().lower()
    if mode_key == "pass":
        return PASS_SCHEMA
    if mode_key in {"conventional", "fixed", "fpa"}:
        return CONVENTIONAL_SCHEMA
    raise ValueError(f"Unsupported mode: {mode!r}")


def detect_schema(output_columns: Sequence[str]) -> DatasetSchema:
    columns = tuple(output_columns)
    if columns == PASS_OUTPUT_COLUMNS:
        return PASS_SCHEMA
    if columns == CONVENTIONAL_OUTPUT_COLUMNS:
        return CONVENTIONAL_SCHEMA
    raise ValueError(
        "Unknown output schema. Expected the PASS or conventional CSV column layout."
    )

