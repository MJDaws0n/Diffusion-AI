import numpy as np

from diffusion_chatbot.config import ModelConfig
from diffusion_chatbot.model import AdamW, SimpleDenoiser, clip_grad_norm, load_optimizer_state
from diffusion_chatbot.tokenizer import WordTokenizer


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


def test_gradient_clip_limits_norm():
    grads = {"a": np.asarray([3.0, 4.0], dtype=np.float32)}
    clipped = clip_grad_norm(grads, max_norm=1.0)
    assert np.linalg.norm(clipped["a"]) <= 1.0001


def test_checkpoint_can_restore_optimizer_state(tmp_path):
    cfg = ModelConfig(vocab_size=20, max_prompt_tokens=5, max_response_tokens=4, embed_dim=8, hidden_dim=16, diffusion_steps=4)
    tokenizer = WordTokenizer.train(["hello there", "hello back"], vocab_size=20)
    model = SimpleDenoiser(cfg, pad_id=tokenizer.pad_id, mask_id=tokenizer.mask_id, invalid_sample_ids=[0, 1])
    opt = AdamW(model.params(), lr=0.001)
    opt.t = 3
    opt.m["b1"][0] = 1.25
    path = tmp_path / "model.npz"
    model.save(path, tokenizer, extra={"step": 3}, optimizer=opt)

    restored_model, _restored_tokenizer, meta = SimpleDenoiser.load(path)
    restored_opt = AdamW(restored_model.params(), lr=0.001)
    assert meta["extra"]["step"] == 3
    assert load_optimizer_state(path, restored_opt)
    assert restored_opt.t == 3
    assert restored_opt.m["b1"][0] == 1.25
