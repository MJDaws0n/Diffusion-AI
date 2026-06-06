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
                "Install CUDA component wheels with `pip install \"cupy-cuda12x[ctk]\"` "
                "or install a matching CUDA Toolkit. Original error: "
                f"{exc}"
            ) from exc
        return cp, "cuda"
    raise ValueError("device must be cpu, cuda, or auto")


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
