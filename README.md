# colony-simulator-cupy

Python module implementing a state-aggregated model for simulating growth of yeast colonies
on agar plates. Runs on one NVIDIA GPU through CuPy; the bracket-resolved cell distribution
lives in host memory and is streamed through the GPU in layer chunks, so plate size is bounded
by host RAM rather than VRAM.

Units throughout: millimetres, hours, grams (concentrations in g/ml).

## Installation

CuPy wheels are tied to the CUDA toolkit on the machine, so pick the matching extra.

```bash
# local checkout, editable
pip install -e ".[cuda12x]"

# Google Colab (cupy-cuda12x is preinstalled there, so no extra is needed)
pip install git+https://github.com/<owner>/colony-simulator-cupy.git

# cluster with CUDA 13
pip install "colonysimulator[cuda13x] @ git+https://github.com/<owner>/colony-simulator-cupy.git"
```

## Layout

| Module | Role |
|---|---|
| `colonysimulator/plate.py` | `PlateGeometry`: grid dimensions, agar depth, colony layers and the 2D plate mask (rectangular omnitray or circular dish) |
| `colonysimulator/strain.py` | `CellStrain`: division time, Michaelis-Menten uptake (`m_max`, `km`), maintenance consumption, cell volume |
| `colonysimulator/colony.py` | `ColonyState`: host-resident bracket-resolved distribution, GPU-resident per-layer totals and dead cells, height map |
| `colonysimulator/streaming.py` | `LayerStreamer`: double-buffered host-to-device streaming of layer chunks on CUDA streams |
| `colonysimulator/kernels/` | CUDA kernels loaded with `cupy.RawKernel` |
| `validation/` | Scripts that exercise one part of the model each and write figures to `validation_outputs/` |

## Status

The simulator is being rebuilt around a 3D colony (stacked layers of extracellular matrix on
top of the agar), a per-column implicit nutrient solver with Michaelis-Menten sinks, and
tension-driven cell spreading. Step 1 (geometry, data layout, streaming) is in place; the
nutrient solver, growth kernel wiring, spreading and the experiment script follow.
