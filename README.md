# XRecon

XRecon is a PyTorch extension for CT projection and reconstruction. It provides
explicit Python volume/geometry classes and native CPU/CUDA distance-driven
kernels.

## Features

- 2D projection, backprojection, weighted backprojection, transpose operators,
  FBP, Parker weighting, and sparse system-matrix generation for parallel,
  fan-flat, and fan-arc geometries.
- 3D projection, backprojection, weighted backprojection, transpose operators,
  FDK, and sparse system-matrix generation for parallel, cone-flat, and
  cone-arc geometries.
- CPU and CUDA kernels for the main 2D/3D operators.
- Utility Shepp-Logan phantoms for examples and quick experiments.

## Repository Layout

```text
xrecon/
  projector/          Python geometry classes, projector modules, autograd wrappers
  utils/              Phantom and data utility helpers
include/              Shared C++/CUDA structs and math helpers
src/api/              PyTorch binding layer
src/cpu/              Native CPU kernels and system matrix builders
src/kernels/          CUDA kernels
examples/             Small user-facing usage templates
scripts/              Data-specific reconstruction entry points
```

## Installation

XRecon builds a C++/CUDA PyTorch extension. The CUDA toolkit used by `nvcc`
should match the CUDA version used by PyTorch.

Create and activate a conda environment:

```bash
conda create -n xrecon python=3.10 -y
conda activate xrecon
```

Install PyTorch. Choose the CUDA package that matches your machine. For
example, for a CUDA 12.8 PyTorch build:

```bash
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia
```

Install a matching CUDA toolkit with `nvcc` inside the conda environment:

```bash
conda install -c nvidia cuda-toolkit=12.8
```

If the toolkit package is split on your platform, install the compiler and
runtime development packages instead:

```bash
conda install -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
```

Point the build to the conda CUDA toolkit:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

On Windows PowerShell, use:

```powershell
$env:CUDA_HOME = $env:CONDA_PREFIX
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
```

Check that PyTorch and `nvcc` agree:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

Then install XRecon from the repository root:

```bash
pip install -e ".[examples]" --no-build-isolation
```

After changing C++/CUDA code, rebuild in place:

```bash
python setup.py build_ext --inplace
```

Notes:

- Linux: if you build without a visible GPU, set `TORCH_CUDA_ARCH_LIST`, for
  example `export TORCH_CUDA_ARCH_LIST="8.6"`.
- Windows: install Visual Studio Build Tools or Visual Studio with the C++
  workload, then build from an x64 Developer Command Prompt or a shell where the
  MSVC environment is initialized.
- If CUDA needs a specific host compiler, set `CUDAHOSTCXX` before building.

## Geometry Conventions

Python projectors can be created with explicit classes:

```python
import xrecon

volume = xrecon.Volume3d((256, 256, 10), (1.0, 1.0, 1.0))
detector = xrecon.Detector3d((256, 10), (2.0, 2.0))
geometry = xrecon.ConeBeam3d(
    detector,
    iso_source=500.0,
    source_detector=1000.0,
    detector_type="flat",
)
projector = xrecon.Projector3d(volume, geometry, filter_type="ramp")
```

Array shapes:

```text
2D image:      [B, C, H, W]
2D sinogram:  [B, C, V, Nu]
3D image:     [B, C, D, H, W]
3D sinogram:  [B, C, V, Nv, Nu]
```

`Nu` is the detector column/channel dimension and is the last sinogram
dimension. `Nv` is the detector row dimension and is the second-to-last
sinogram dimension.

## Examples

Run examples from the repository root:

```bash
python examples/projector2d_example.py
python examples/projector3d_example.py
python examples/system_matrix_example.py
```

The examples use built-in phantoms and write preview images to
`outputs/examples`.

## Basic Usage

```python
import torch
import xrecon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

projector = xrecon.Projector2d(
    xrecon.Volume2d(256, 1.0),
    xrecon.FanBeam2d(
        xrecon.Detector2d(256, 2.0),
        iso_source=500.0,
        source_detector=1000.0,
        detector_type="flat",
    ),
).to(device)

image = torch.from_numpy(xrecon.phantom(256, dim=2)).float()[None, None].to(device)
angles = torch.arange(720, device=device, dtype=torch.float32) * (2.0 * torch.pi / 720)

sino = projector.projection(image, angles)
recon = projector.filtered_backprojection(sino, angles)
```

## Developer Notes

- Public Python interfaces should live under `xrecon/projector` or
  `xrecon/utils`. `src/` should stay as plain C++/CUDA implementation code.
- Shared math helpers belong in `include/xrmath.h`; CUDA texture helpers and
  checks belong in `include/utils.h`; CPU coefficient visitors belong in
  `include/cpu3d_common.h`.
- Transpose kernels are adjoint operators. They may use different traversal and
  thread ownership than the true projection/backprojection kernels.
