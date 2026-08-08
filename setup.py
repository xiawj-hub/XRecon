from __future__ import annotations

import os
import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


def compile_args() -> dict[str, list[str]]:
    cxx_args = ["/O2"] if platform.system() == "Windows" else ["-O3"]
    nvcc_args = ["-O3", "--use_fast_math", "-lineinfo"]

    cuda_host = os.environ.get("CUDAHOSTCXX")
    if cuda_host:
        nvcc_args += ["-ccbin", cuda_host]

    return {"cxx": cxx_args, "nvcc": nvcc_args}


SOURCES = [
    "src/api/torch_binding.cpp",
    "src/cpu/forward2d.cpp",
    "src/cpu/backward2d.cpp",
    "src/cpu/system_matrix2d.cpp",
    "src/cpu/forward3d.cpp",
    "src/cpu/backward3d.cpp",
    "src/cpu/system_matrix3d.cpp",
    "src/kernels/forward2d.cu",
    "src/kernels/forward2d_t.cu",
    "src/kernels/backward2d.cu",
    "src/kernels/backward2d_t.cu",
    "src/kernels/forward3d.cu",
    "src/kernels/forward3d_flat.cu",
    "src/kernels/forward3d_arc.cu",
    "src/kernels/backward3d.cu",
    "src/kernels/backward3d_flat.cu",
    "src/kernels/backward3d_arc.cu",
    "src/kernels/forward3d_parallel.cu",
    "src/kernels/helical_arc3d.cu",
]


setup(
    name="xrecon",
    version="0.1.0",
    author="Wenjun Xia",
    description="CT projection and reconstruction operators for PyTorch",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    extras_require={
        "examples": ["matplotlib"],
        "mayo": ["pydicom", "matplotlib"],
    },
    ext_modules=[
        CUDAExtension(
            name="xrecon_C",
            sources=SOURCES,
            include_dirs=[str(ROOT / "include")],
            extra_compile_args=compile_args(),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
