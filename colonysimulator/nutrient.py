"""Nutrient field: diffusion through agar and colony, and uptake by the cells.

Units: millimetres, hours, grams. Concentrations are in g/ml (= g/cm^3), so the nutrient mass
in a voxel is `concentration * volume_mm3 * 1e-3` grams.

Data layout
-----------
`concentration` (device): (agar_depth + n_layers, length, width) float32
    One vertical stack per (x, y) column. Rows 0 .. agar_depth-1 are agar (row 0 is the
    bottom of the dish), the rows above are the colony layers, in the same order as the
    layer axis of `ColonyState` (colony layer 0 sits on the agar). `agar` and `colony` are
    views of the two parts.

Every time step does two things, one after the other (operator splitting):

1. Vertical: in every column, diffusion between rows plus Michaelis-Menten uptake by the
   cells, solved implicitly with a tridiagonal (Thomas) solver on the GPU, one thread per
   column (`kernels/column_diffusion.cu`). The agar and the colony have their own diffusion
   coefficients; a colony voxel only conducts nutrient when it holds ECM, i.e. at least
   `ecm_min_cells` cells (living or dead).
2. Horizontal: in every row, a Gaussian spread with variance 2 D dt (the exact solution of
   the diffusion equation for one step in free space). Nutrient never leaves the medium: in
   the agar it stays inside the plate mask, in a colony layer it stays inside the ECM, and
   the mass of every connected ECM region of a layer is conserved separately. Near a wall
   this is approximate: what would cross the wall stays where it started instead of being
   reflected (see `masked_gaussian_spread`).

The uptake of each strain in each voxel during the step is returned, so that the growth
step can turn it into new cells.
"""
from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage

from .colony import ColonyState
from .plate import PlateGeometry

ML_PER_MM3 = 1e-3

# Gaussian filters are cut off at this many standard deviations.
GAUSSIAN_TRUNCATE = 4.0

# Connectivity used to find connected regions: neighbours within the same row only (8 in the
# plane, including diagonals), never across rows, so every row is labelled on its own.
_IN_PLANE_STRUCTURE = np.zeros((3, 3, 3), dtype=bool)
_IN_PLANE_STRUCTURE[1] = True

_KERNEL_SOURCE = (Path(__file__).resolve().parent / "kernels" / "column_diffusion.cu").read_text()
_THREADS_PER_BLOCK = 256


def label_regions(medium: cp.ndarray) -> tuple[cp.ndarray, int]:
    """Label the connected regions of every row of a (rows, length, width) bool array.

    Returns integer labels of the same shape (0 outside the medium, regions numbered from 1,
    unique across rows) and the number of regions.
    """
    labels, count = ndimage.label(medium, structure=_IN_PLANE_STRUCTURE)
    return labels, int(count)


def masked_gaussian_spread(block: cp.ndarray, medium: cp.ndarray, sigma: float,
                           labels: cp.ndarray | None = None,
                           region_count: int | None = None) -> cp.ndarray:
    """Spread every row of `block` horizontally with a Gaussian, keeping it inside `medium`.

    `block` is (rows, length, width) concentration, `medium` the bool mask of where nutrient
    may be, `sigma` the standard deviation in voxels (the same for every row). Values
    outside the medium are left as they are.

    Every voxel sends its content to its neighbours with Gaussian weights. The share of its
    Gaussian that would land outside the medium (beyond the plate wall, or off the edge of
    the ECM) stays in the voxel instead. Written as a matrix this is symmetric with rows
    summing to one, so it conserves mass exactly, never makes a concentration negative, and
    leaves a uniform field uniform (no spurious build-up at walls).

    That alone conserves the total of each row. When two separate regions lie closer than
    the Gaussian's reach, some nutrient would hop across the gap, so afterwards every
    connected region is rescaled to the mass it had before.
    """
    if sigma <= 0:
        return block.copy()
    if labels is None or region_count is None:
        labels, region_count = label_regions(medium)

    weight_field = medium.astype(cp.float32)
    sigmas = (0.0, float(sigma), float(sigma))
    share_inside = ndimage.gaussian_filter(weight_field, sigmas, mode="constant", cval=0.0,
                                           truncate=GAUSSIAN_TRUNCATE)
    inside_block = cp.where(medium, block, 0.0).astype(cp.float32)
    spread = ndimage.gaussian_filter(inside_block, sigmas, mode="constant", cval=0.0,
                                     truncate=GAUSSIAN_TRUNCATE)
    spread += (1.0 - share_inside) * inside_block
    spread *= weight_field

    flat_labels = labels.ravel()
    before = cp.bincount(flat_labels, weights=inside_block.ravel(),
                         minlength=region_count + 1)
    after = cp.bincount(flat_labels, weights=spread.ravel(), minlength=region_count + 1)
    scale = cp.where(after > 0, before / cp.where(after > 0, after, 1.0), 1.0)
    scale[0] = 1.0
    spread *= scale[labels].astype(cp.float32)
    return cp.where(medium, spread, block).astype(cp.float32)


class NutrientField:
    geometry: PlateGeometry
    agar_diffusion: float     # mm^2/h
    ecm_diffusion: float      # mm^2/h
    theta: float              # time weighting of vertical diffusion (0.5 Crank-Nicolson, 1 backward Euler)
    ecm_min_cells: float      # cells (living + dead, all strains) a colony voxel needs to count as ECM
    concentration: cp.ndarray

    def __init__(self, geometry: PlateGeometry, agar_diffusion: float, ecm_diffusion: float,
                 initial_concentration: float = 0.0, theta: float = 1.0,
                 ecm_min_cells: float = 0.5) -> None:
        if agar_diffusion <= 0 or ecm_diffusion <= 0:
            raise ValueError("diffusion coefficients must be positive")
        if not 0.5 <= theta <= 1.0:
            raise ValueError("theta must lie between 0.5 (Crank-Nicolson) and 1 (backward Euler)")
        if ecm_min_cells < 0:
            raise ValueError("ecm_min_cells must be non-negative")
        self.geometry = geometry
        self.agar_diffusion = float(agar_diffusion)
        self.ecm_diffusion = float(ecm_diffusion)
        self.theta = float(theta)
        self.ecm_min_cells = float(ecm_min_cells)

        self.agar_depth = geometry.agar_depth
        self.n_layers = geometry.n_layers
        self.length = geometry.length
        self.width = geometry.width
        self.area = self.length * self.width

        rows = self.agar_depth + self.n_layers
        self.concentration = cp.zeros((rows, self.length, self.width), dtype=cp.float32)
        self.set_agar_concentration(initial_concentration)

        # Row volumes in ml, used for mass bookkeeping.
        self.row_volume_ml = cp.concatenate([
            cp.full(self.agar_depth, geometry.agar_voxel_volume * ML_PER_MM3, dtype=cp.float64),
            cp.full(self.n_layers, geometry.layer_voxel_volume * ML_PER_MM3, dtype=cp.float64),
        ])

        # The agar is the same in every step, so its regions are labelled once.
        self._agar_medium = cp.broadcast_to(geometry.mask, (self.agar_depth, self.length, self.width))
        self._agar_labels, self._agar_regions = label_regions(cp.ascontiguousarray(self._agar_medium))

        self._kernel = cp.RawKernel(_KERNEL_SOURCE, "columnDiffusionKernel")
        self._sweep_c: cp.ndarray | None = None
        self._sweep_d: cp.ndarray | None = None

    # ---- views and set-up ----------------------------------------------------------------

    @property
    def agar(self) -> cp.ndarray:
        return self.concentration[:self.agar_depth]

    @property
    def colony(self) -> cp.ndarray:
        return self.concentration[self.agar_depth:]

    def set_agar_concentration(self, concentration: float | cp.ndarray) -> None:
        """Fill the agar (inside the plate mask) with `concentration` g/ml."""
        self.agar[...] = cp.asarray(concentration, dtype=cp.float32) * self.geometry.mask

    # ---- bookkeeping ---------------------------------------------------------------------

    def mass_per_row(self) -> cp.ndarray:
        """Nutrient mass in grams in every row, (agar_depth + n_layers,) float64."""
        return self.concentration.sum(axis=(1, 2), dtype=cp.float64) * self.row_volume_ml

    def total_mass(self) -> float:
        """Total nutrient mass on the plate in grams."""
        return float(self.mass_per_row().sum())

    def medium_mask(self, colony: ColonyState, n_layers: int | None = None) -> cp.ndarray:
        """Bool (n_layers, length, width): colony voxels that hold ECM."""
        n_layers = colony.active_layers if n_layers is None else int(n_layers)
        cells = (colony.living_total[:n_layers] + colony.dead[:n_layers]).sum(axis=1)
        return (cells >= self.ecm_min_cells) & (cells > 0) & self.geometry.mask[None]

    def saturation(self, colony: ColonyState, n_layers: int | None = None) -> cp.ndarray:
        """Michaelis-Menten saturation C / (Km + C) of every strain, (n_layers, n_strains, length, width)."""
        n_layers = colony.active_layers if n_layers is None else int(n_layers)
        c = self.colony[:n_layers, None]
        km = cp.asarray([strain.km for strain in colony.strains], dtype=cp.float32)[None, :, None, None]
        return c / (km + c)

    # ---- the time step -------------------------------------------------------------------

    def _scratch(self, rows: int) -> tuple[cp.ndarray, cp.ndarray]:
        if self._sweep_c is None or self._sweep_c.shape[0] < rows:
            # Grow in steps of 16 rows so a slowly rising colony does not reallocate every step.
            capacity = min(self.agar_depth + self.n_layers, rows + 16)
            self._sweep_c = cp.empty((capacity, self.area), dtype=cp.float64)
            self._sweep_d = cp.empty((capacity, self.area), dtype=cp.float64)
        assert self._sweep_d is not None
        return self._sweep_c, self._sweep_d

    def vertical_step(self, sink_rate: cp.ndarray, medium: cp.ndarray, dt: float) -> None:
        """Implicit vertical diffusion with a linear sink in every column.

        `sink_rate` (1/h) and `medium` have shape (colony_rows, length, width); the solve covers
        the agar plus these colony rows.
        """
        colony_rows = int(sink_rate.shape[0])
        rows = self.agar_depth + colony_rows
        sweep_c, sweep_d = self._scratch(rows)
        sink_rate = cp.ascontiguousarray(sink_rate, dtype=cp.float32)
        medium = cp.ascontiguousarray(medium, dtype=cp.bool_)
        if colony_rows == 0:
            # The kernel still reads the pointers; give it one valid (unused) element.
            sink_rate = cp.zeros(1, dtype=cp.float32)
            medium = cp.zeros(1, dtype=cp.bool_)
        blocks = (self.area + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        self._kernel((blocks,), (_THREADS_PER_BLOCK,), (
            self.concentration, sink_rate, medium, self.geometry.mask, sweep_c, sweep_d,
            np.int32(self.agar_depth), np.int32(colony_rows), np.int32(self.area),
            np.float64(self.geometry.spatial_resolution), np.float64(self.geometry.layer_height),
            np.float64(self.agar_diffusion), np.float64(self.ecm_diffusion),
            np.float64(dt), np.float64(self.theta),
        ))

    def horizontal_step(self, medium: cp.ndarray, dt: float) -> None:
        """Gaussian spread of every agar row and every colony row in `medium`."""
        h = self.geometry.spatial_resolution
        agar_sigma = np.sqrt(2.0 * self.agar_diffusion * dt) / h
        self.concentration[:self.agar_depth] = masked_gaussian_spread(
            self.agar, self._agar_medium, agar_sigma, self._agar_labels, self._agar_regions)

        colony_rows = int(medium.shape[0])
        if colony_rows:
            ecm_sigma = np.sqrt(2.0 * self.ecm_diffusion * dt) / h
            rows = slice(self.agar_depth, self.agar_depth + colony_rows)
            self.concentration[rows] = masked_gaussian_spread(self.concentration[rows], medium, ecm_sigma)

    def step(self, colony: ColonyState, dt: float) -> cp.ndarray:
        """Advance the nutrient field by `dt` hours with uptake by the cells in `colony`.

        Returns the nutrient taken up during the step by every strain in every voxel of the
        occupied layers, in grams, shape (colony.active_layers, n_strains, length, width).
        Uptake follows Michaelis-Menten kinetics per strain,
            uptake_s = dt * N_s * m_max_s * C_new / (Km_s + C_old),
        where C_old is the concentration at the start of the step. Using C_new in the
        numerator makes uptake implicit: it can never exceed what the voxel holds.
        """
        if colony.geometry is not self.geometry:
            raise ValueError("colony and nutrient field must share the same PlateGeometry")
        n_active = colony.active_layers
        medium = self.medium_mask(colony, n_active)

        voxel_ml = self.geometry.layer_voxel_volume * ML_PER_MM3
        old = self.colony[:n_active].copy()
        m_max = cp.asarray([strain.m_max for strain in colony.strains], dtype=cp.float32)[None, :, None, None]
        km = cp.asarray([strain.km for strain in colony.strains], dtype=cp.float32)[None, :, None, None]
        # Uptake capacity of every strain per unit concentration, g/h per (g/ml).
        uptake_coefficient = colony.living_total[:n_active] * m_max / (km + old[:, None])
        uptake_coefficient *= medium[:, None]
        sink_rate = uptake_coefficient.sum(axis=1) / voxel_ml

        self.vertical_step(sink_rate, medium, dt)

        uptake = (dt * uptake_coefficient * self.colony[:n_active, None]).astype(cp.float32)
        self.horizontal_step(medium, dt)
        return uptake
