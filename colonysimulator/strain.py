"""Strain parameters.

Units: millimetres, hours, grams. Nutrient concentrations are in g/ml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for the type hint below; a runtime import is not needed.
    from .plate import PlateGeometry


@dataclass(frozen=True)
class CellStrain:
    name: str
    division_time: float        # h, at full nutrient supply
    m_max: float                # g nutrient / cell / h, maximal Michaelis-Menten uptake
    km: float                   # g/ml, half-saturation concentration
    nutrient_consumption: float # g nutrient / cell / h, maintenance requirement
    volume: float               # mm^3 per cell (42 um^3 = 42e-9 mm^3 for yeast)

    def __post_init__(self) -> None:
        if self.division_time <= 0:
            raise ValueError("division_time must be positive")
        if self.m_max <= 0 or self.km <= 0:
            raise ValueError("m_max and km must be positive")
        if not 0 <= self.nutrient_consumption < self.m_max:
            raise ValueError("nutrient_consumption must be non-negative and below m_max")
        if self.volume <= 0:
            raise ValueError("volume must be positive")

    @property
    def maintenance_ratio(self) -> float:
        """Food ratio C/(Km + C) below which cells cannot cover maintenance."""
        return self.nutrient_consumption / self.m_max

    def capacity_per_voxel(self, geometry: PlateGeometry) -> float:
        """Number of this strain's cells that fit in one colony-layer voxel."""
        return geometry.layer_voxel_volume / self.volume
