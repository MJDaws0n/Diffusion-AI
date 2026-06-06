import ctypes
import glob
import os
import site
import sys

import numpy as np


def get_backend(device="auto"):
    device = (device or "auto").lower()
    if device in {"cpu", "numpy"}:
        return np, "cpu"
    if device in {"cuda", "gpu", "auto"}:
        try:
            import cupy as cp
        except ImportError:
            if device == "auto":
                return np, "cpu"
            raise RuntimeError("CUDA requested but CuPy is not installed. Install cupy-cuda12x or cupy-cuda11x.")
        _preload_nvidia_cuda_wheels()
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            if device == "auto":
                return np, "cpu"
            raise RuntimeError(f"CUDA requested but no usable NVIDIA GPU was found: {exc}") from exc
        if count <= 0:
            if device == "auto":
                return np, "cpu"
            raise RuntimeError("CUDA requested but no NVIDIA GPU was found.")
        try:
            probe = cp.arange(2)
            _ = (probe != 1).sum()
            cp.cuda.Stream.null.synchronize()
        except Exception as exc:
            if device == "auto":
                return np, "cpu"
            raise RuntimeError(
                "CUDA requested and GPU was found, but CuPy could not compile/run a CUDA kernel. "
                "Install CUDA component wheels with `pip install -r requirements-cuda.txt` "
                "or install a matching CUDA Toolkit. Original error: "
                f"{exc}"
            ) from exc
        return cp, "cuda"
    raise ValueError("device must be cpu, cuda, or auto")


def _preload_nvidia_cuda_wheels():
    # NVIDIA's PyPI CUDA component wheels put shared libraries under
    # site-packages/nvidia/*/lib. Some Linux setups do not expose that path to
    # dlopen, so CuPy sees the GPU but fails when loading CUDA component libs.
    lib_patterns = [
        "nvidia/cuda_runtime/lib/libcudart.so*",
        "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so*",
        "nvidia/cuda_nvrtc/lib/libnvrtc.so*",
        "nvidia/cublas/lib/libcublasLt.so*",
        "nvidia/cublas/lib/libcublas.so*",
    ]
    for pattern in lib_patterns:
        for path in _find_site_library_paths(pattern):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _find_site_library_paths(pattern):
    roots = []
    try:
        roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        roots.append(site.getusersitepackages())
    except AttributeError:
        pass
    roots.extend(sys.path)

    found = []
    for root in roots:
        if not root:
            continue
        found.extend(glob.glob(os.path.join(root, pattern)))
    return sorted(set(found))


def array_module(value):
    if value.__class__.__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return cp
    return np


def asnumpy(value):
    if value.__class__.__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return np.asarray(value)


def scalar(value):
    return asnumpy(value).item()
