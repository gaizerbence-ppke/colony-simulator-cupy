"""Plate geometry: voxel grid dimensions and the 2D mask that defines the plate outline.

Units throughout the package: millimetres, hours, grams.

The horizontal grid is `length x width` voxels of `spatial_resolution` mm. Below the surface
there are `agar_depth` agar voxels, each `spatial_resolution` mm tall. Above the surface
there are up to `n_layers` colony layers, each `layer_height` mm tall (typically thinner
than an agar voxel). The mask marks which (x, y) columns belong to the plate; everything
outside the mask is wall: no agar, no cells, zero flux.
"""
from __future__ import annotations

import math

import cupy as cp
import numpy as np


class PlateGeometry:
    def __init__(self, mm_length, mm_width, mm_depth, spatial_resolution, layer_height,
                 max_colony_height, mask=None):
        self.mm_length = float(mm_length)
        self.mm_width = float(mm_width)
        self.mm_depth = float(mm_depth)
        self.spatial_resolution = float(spatial_resolution)
        self.layer_height = float(layer_height)
        self.max_colony_height = float(max_colony_height)

        self.length = int(self.mm_length // self.spatial_resolution)
        self.width = int(self.mm_width // self.spatial_resolution)
        self.agar_depth = int(self.mm_depth // self.spatial_resolution)
        self.n_layers = int(math.ceil(self.max_colony_height / self.layer_height))

        if self.length < 1 or self.width < 1 or self.agar_depth < 1 or self.n_layers < 1:
            raise ValueError("Plate dimensions must span at least one voxel in every direction")

        if mask is None:
            mask = np.ones((self.length, self.width), dtype=bool)
        mask = cp.asarray(mask, dtype=bool)
        if mask.shape != (self.length, self.width):
            raise ValueError(f"mask shape {mask.shape} does not match grid ({self.length}, {self.width})")
        self.mask = mask

    # ---- factories -------------------------------------------------------------------

    @classmethod
    def rectangular(cls, mm_length, mm_width, mm_depth, spatial_resolution, layer_height,
                    max_colony_height):
        """Rectangular plate, e.g. a single-well omnitray (about 128 x 86 mm)."""
        return cls(mm_length, mm_width, mm_depth, spatial_resolution, layer_height,
                   max_colony_height, mask=None)

    @classmethod
    def circular(cls, diameter, mm_depth, spatial_resolution, layer_height, max_colony_height):
        """Circular Petri dish of the given diameter, embedded in a square grid."""
        length = int(diameter // spatial_resolution)
        centre = (length - 1) / 2.0
        radius = diameter / spatial_resolution / 2.0
        ii, jj = np.meshgrid(np.arange(length), np.arange(length), indexing="ij")
        mask = (ii - centre) ** 2 + (jj - centre) ** 2 <= radius ** 2
        return cls(diameter, diameter, mm_depth, spatial_resolution, layer_height,
                   max_colony_height, mask=mask)

    # ---- derived quantities ----------------------------------------------------------

    @property
    def voxel_area(self):
        """Horizontal voxel footprint in mm^2."""
        return self.spatial_resolution ** 2

    @property
    def agar_voxel_volume(self):
        """Volume of one agar voxel in mm^3."""
        return self.spatial_resolution ** 3

    @property
    def layer_voxel_volume(self):
        """Volume of one colony-layer voxel in mm^3."""
        return self.voxel_area * self.layer_height

    @property
    def plate_columns(self):
        """Number of (x, y) columns inside the plate mask."""
        return int(self.mask.sum())

    def to_voxel(self, x_mm, y_mm):
        """Map a position in mm (measured from the plate corner) to grid indices."""
        i = int(x_mm // self.spatial_resolution)
        j = int(y_mm // self.spatial_resolution)
        if not (0 <= i < self.length and 0 <= j < self.width):
            raise ValueError(f"({x_mm}, {y_mm}) mm lies outside the {self.mm_length} x {self.mm_width} mm grid")
        if not bool(self.mask[i, j]):
            raise ValueError(f"({x_mm}, {y_mm}) mm lies outside the plate mask")
        return i, j

    def centre_voxel(self):
        return self.length // 2, self.width // 2

    def describe(self):
        return (f"{self.mm_length} x {self.mm_width} mm plate, {self.length} x {self.width} columns "
                f"({self.plate_columns} inside the mask), {self.agar_depth} agar voxels of "
                f"{self.spatial_resolution} mm, up to {self.n_layers} colony layers of {self.layer_height} mm")
