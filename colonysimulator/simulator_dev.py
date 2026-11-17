from __future__ import annotations
import os

import numpy as np
import cupy as cp

class AgarModel:
    def __init__(self, mm_length, mm_width, mm_depth, spatial_resolution, time_resolution, diffusion_coefficient, ecm_diffusion_coefficient):
        self.mm_length = mm_length
        self.mm_width = mm_width
        self.mm_depth = mm_depth
        self.spatial_resolution = spatial_resolution
        self.time_resolution = time_resolution

        self.diffusion_coefficient = diffusion_coefficient
        self.ecm_diffusion_coefficient = ecm_diffusion_coefficient

        self.length = int(self.mm_length // self.spatial_resolution)
        self.width = int(self.mm_width // self.spatial_resolution)
        self.depth = int(self.mm_depth // self.spatial_resolution)

        self.plate_concentration = np.zeros((self.length, self.width, self.depth), dtype=np.float32)
        self.ecm_concentration = np.zeros((self.length, self.width, 1), dtype=np.float32)

        kernel_code = open(os.path.join(os.path.dirname(__file__), "diffusion_kernel.cu"), "r").read()
        self.diffusion_kernel = cp.RawKernel(kernel_code, "1D_diffusion")

    def setConcentration(self, concentration):
        self.plate_concentration[:, :, :] = concentration * self.spatial_resolution**3 * 1e-3

    def refresh_ecm_size(self, colony_model: ColonyModel):
        while len(colony_model.growing_matrix) > self.ecm_concentration.shape[2]:
            self.ecm_concentration = np.concatenate((self.ecm_concentration, np.zeros((self.length, self.width, 1), dtype=np.float32)), axis=2)

    def diffusion_step(self):
            grid_size = (self.length, self.width)
            block_size = (1, 1)
            self.diffusion_kernel(
                self.plate_concentration, 
                self.ecm_concentration, 
                self.diffusion_coefficient, 
                self.ecm_diffusion_coefficient, 
                self.spatial_resolution, 
                self.time_resolution, 
                grid=grid_size, 
                block=block_size
            )

class CellStrain:
    def __init__(self, name: str, divisionTime: float, nutrientUptake: float, nutrientConsumption: float, volume: float):
        self.name = name
        self.division_time = divisionTime
        self.nutrient_uptake = nutrientUptake
        self.nutrient_consumption = nutrientConsumption
        self.volume = volume

class ColonyModel:
    def __init__(self, agar_model : AgarModel, bracket_count: int, cell_strain: CellStrain):
        self.agar_model = agar_model
        self.bracket_count = bracket_count
        self.division_time = cell_strain.division_time
        self.nutrient_uptake = cell_strain.nutrient_uptake
        self.nutrient_consumption = cell_strain.nutrient_consumption
        self_maximum_cells_per_voxel = self.agar_model.spatial_resolution**2 * 5e-3 / cell_strain.volume

        self.growing_matrix = [np.zeros((self.bracket_count, self.agar_model.length, self.agar_model.width), dtype=cp.float32)]
