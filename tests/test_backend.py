import numpy as np

from diffusion_chatbot.backend import asnumpy, get_backend


def test_cpu_backend_returns_numpy():
    xp, device = get_backend("cpu")
    assert xp is np
    assert device == "cpu"


def test_asnumpy_returns_numpy_array():
    arr = asnumpy(np.asarray([1, 2, 3]))
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2, 3]
