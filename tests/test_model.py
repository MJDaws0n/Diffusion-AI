import numpy as np

from diffusion_chatbot.config import ModelConfig
from diffusion_chatbot.model import SimpleDenoiser


def test_model_loss_and_grads_are_finite():
    cfg = ModelConfig(vocab_size=20, max_prompt_tokens=5, max_response_tokens=4, embed_dim=8, hidden_dim=16, diffusion_steps=4)
    model = SimpleDenoiser(cfg, pad_id=0, mask_id=1, invalid_sample_ids=[0, 1])
    prompts = np.asarray([[2, 7, 3, 0, 0], [2, 8, 3, 0, 0]], dtype=np.int64)
    noisy = np.asarray([[1, 1, 4, 0], [1, 9, 1, 0]], dtype=np.int64)
    targets = np.asarray([[5, 6, 4, 0], [8, 9, 4, 0]], dtype=np.int64)
    timesteps = np.asarray([4, 2], dtype=np.int64)
    mask = noisy == 1
    loss, grads = model.loss_and_grads(prompts, noisy, timesteps, targets, mask)
    assert np.isfinite(loss)
    assert set(grads) == set(model.params())
    assert all(np.all(np.isfinite(value)) for value in grads.values())
