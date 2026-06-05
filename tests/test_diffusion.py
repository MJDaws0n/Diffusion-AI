import numpy as np

from diffusion_chatbot.diffusion import MaskDiffusion


def test_noise_masks_non_pad_tokens_only():
    diffusion = MaskDiffusion(steps=4, min_mask_prob=1.0, max_mask_prob=1.0)
    rng = np.random.default_rng(1)
    response = np.asarray([[10, 11, 0, 0]], dtype=np.int64)
    noisy, flags = diffusion.noise_batch(response, np.asarray([4]), mask_id=1, pad_id=0, rng=rng)
    assert noisy.tolist() == [[1, 1, 0, 0]]
    assert flags.tolist() == [[True, True, False, False]]
