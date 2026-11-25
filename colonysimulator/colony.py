"""Colony state: where the cells are.

Data layout
-----------
`living` (host, pinned):   (n_layers, n_strains, bracket_count, length, width) float32
    Growing cells resolved by maturity bracket. Too large for VRAM at realistic plate
    sizes, so it stays on the host and is streamed through the GPU in layer chunks.
`living_total` (device):   (n_layers, n_strains, length, width) float32
    Sum of `living` over brackets. Kept on the GPU because diffusion sinks, spreading
    and the height map only need totals.
`dead` (device):           (n_layers, n_strains, length, width) float32
    Dead cells. They take up space and are pushed around by spreading like living cells.

Layer 0 sits directly on the agar surface. The layer axis comes first so that a chunk of
consecutive layers across all strains is one contiguous block of host memory.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cupy as cp
import numpy as np

from .plate import PlateGeometry
from .strain import CellStrain
from .streaming import ChunkFunction, LayerStreamer, chunk_layers_for_budget, pinned_zeros

# A strain can be referred to by its position in `ColonyState.strains`, by its name, or by
# the CellStrain object itself.
StrainRef = int | str | CellStrain


class ColonyState:
    geometry: PlateGeometry
    strains: list[CellStrain]
    bracket_count: int
    n_strains: int
    n_layers: int
    length: int
    width: int
    capacity: cp.ndarray        # (n_strains,) float32, cells per voxel
    living: np.ndarray          # host, see the module docstring for the shape
    living_total: cp.ndarray    # device
    dead: cp.ndarray            # device
    streamer: LayerStreamer
    active_layers: int

    def __init__(self, geometry: PlateGeometry, strains: CellStrain | Iterable[CellStrain],
                 bracket_count: int, pinned: bool = True,
                 device_memory_budget: int | None = None, chunk_layers: int | None = None,
                 n_buffers: int = 2) -> None:
        if isinstance(strains, CellStrain):
            strains = [strains]
        strains = list(strains)
        if not strains:
            raise ValueError("at least one strain is required")
        if len({strain.name for strain in strains}) != len(strains):
            raise ValueError("strain names must be unique")
        if bracket_count < 1:
            raise ValueError("bracket_count must be at least 1")

        self.geometry = geometry
        self.strains = strains
        self.bracket_count = int(bracket_count)
        self.n_strains = len(strains)
        self.n_layers = geometry.n_layers
        self.length = geometry.length
        self.width = geometry.width

        self.capacity = cp.asarray([strain.capacity_per_voxel(geometry) for strain in strains],
                                   dtype=cp.float32)

        layer_shape = (self.n_strains, self.bracket_count, self.length, self.width)
        self.living = pinned_zeros((self.n_layers,) + layer_shape, np.float32, pinned=pinned)
        self.living_total = cp.zeros((self.n_layers, self.n_strains, self.length, self.width), dtype=cp.float32)
        self.dead = cp.zeros((self.n_layers, self.n_strains, self.length, self.width), dtype=cp.float32)

        if chunk_layers is None:
            if device_memory_budget is None:
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
                device_memory_budget = free_bytes // 4
            chunk_layers = chunk_layers_for_budget(layer_shape, np.float32, device_memory_budget,
                                                   n_buffers=n_buffers, n_layers=self.n_layers)
        self.streamer = LayerStreamer(self.living, chunk_layers, n_buffers=n_buffers)

        # Number of layers, counted from the agar surface, that hold any cells. Streaming
        # passes only touch these; everything above is empty by construction.
        self.active_layers = 0

    # ---- bookkeeping -------------------------------------------------------------------

    def strain_index(self, strain: StrainRef) -> int:
        if isinstance(strain, (int, np.integer)):
            return int(strain)
        if isinstance(strain, CellStrain):
            strain = strain.name
        for index, candidate in enumerate(self.strains):
            if candidate.name == strain:
                return index
        raise KeyError(f"unknown strain {strain!r}")

    @property
    def host_bytes(self) -> int:
        return self.living.nbytes

    @property
    def device_bytes(self) -> int:
        return self.living_total.nbytes + self.dead.nbytes + self.streamer.device_bytes

    def describe(self) -> str:
        return (f"{self.n_strains} strain(s), {self.bracket_count} brackets, "
                f"{self.n_layers} layers of {self.length} x {self.width}: "
                f"{self.host_bytes / 1e9:.2f} GB on the host"
                f"{' (pinned)' if self.streamer.pinned else ' (pageable)'}, "
                f"{self.device_bytes / 1e9:.2f} GB on the device, "
                f"streamed {self.streamer.chunk_layers} layer(s) per chunk "
                f"over {self.streamer.n_buffers} buffer(s)")

    # ---- initial conditions ------------------------------------------------------------

    def seed_cells(self, strain: StrainRef, x_mm: float, y_mm: float, cells: float,
                   layer: int = 0, bracket: int = 0) -> None:
        """Put `cells` cells of `strain` into the voxel at (x_mm, y_mm) in the given layer."""
        i, j = self.geometry.to_voxel(x_mm, y_mm)
        self.seed_voxel(strain, i, j, cells, layer=layer, bracket=bracket)

    def seed_voxel(self, strain: StrainRef, i: int, j: int, cells: float,
                   layer: int = 0, bracket: int = 0) -> None:
        s = self.strain_index(strain)
        if not (0 <= layer < self.n_layers):
            raise ValueError(f"layer {layer} outside 0..{self.n_layers - 1}")
        if not bool(self.geometry.mask[i, j]):
            raise ValueError(f"voxel ({i}, {j}) lies outside the plate mask")
        self.living[layer, s, bracket, i, j] += np.float32(cells)
        self.living_total[layer, s, i, j] += np.float32(cells)
        self.active_layers = max(self.active_layers, layer + 1)

    def seed_single_cell_at_centre(self, strain: StrainRef = 0) -> None:
        i, j = self.geometry.centre_voxel()
        self.seed_voxel(strain, i, j, 1.0)

    # ---- streaming over the bracket-resolved distribution -----------------------------

    def map_layers(self, fn: ChunkFunction, write_back: bool = True,
                   all_layers: bool = False) -> list[Any]:
        """Apply `fn(device_chunk, layer_slice)` to the active layers of `living`.

        `device_chunk` has shape (chunk, n_strains, bracket_count, length, width).
        """
        n_layers = None if all_layers else self.active_layers
        return self.streamer.process(fn, n_layers=n_layers, write_back=write_back)

    def refresh_totals(self, all_layers: bool = False) -> None:
        """Recompute `living_total` from the host-side distribution."""
        def reduce_chunk(chunk: cp.ndarray, layers: slice) -> None:
            self.living_total[layers] = chunk.sum(axis=2)
        self.map_layers(reduce_chunk, write_back=False, all_layers=all_layers)
        if all_layers:
            self.update_active_layers()

    def update_active_layers(self) -> int:
        """Recount the occupied layers from the GPU-resident totals."""
        occupied = ((self.living_total + self.dead).sum(axis=(1, 2, 3)) > 0)
        indices = cp.flatnonzero(occupied)
        self.active_layers = int(indices.max()) + 1 if indices.size else 0
        return self.active_layers

    # ---- observables --------------------------------------------------------------------

    def occupancy(self) -> cp.ndarray:
        """Fraction of each voxel's capacity in use, shape (n_layers, length, width)."""
        per_strain = (self.living_total + self.dead) / self.capacity[None, :, None, None]
        return per_strain.sum(axis=1)

    def height_map(self) -> cp.ndarray:
        """Top view of the colony height in mm, shape (length, width)."""
        filled = cp.clip(self.occupancy(), 0.0, 1.0)
        return filled.sum(axis=0) * self.geometry.layer_height

    def cell_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Living and dead cell totals per strain as two host arrays of length n_strains."""
        living = self.living_total.sum(axis=(0, 2, 3)).get()
        dead = self.dead.sum(axis=(0, 2, 3)).get()
        return living, dead
