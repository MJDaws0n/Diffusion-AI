import json

import numpy as np

from .backend import array_module, asnumpy, get_backend, scalar
from .config import ModelConfig
from .tokenizer import WordTokenizer


def softmax(logits):
    xp = array_module(logits)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = xp.exp(shifted)
    return exp / xp.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


class AdamW:
    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4, grad_clip=1.0):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip) if grad_clip else 0.0
        self.t = 0
        first_param = next(iter(params.values()))
        self.xp = array_module(first_param)
        self.m = {name: self.xp.zeros_like(value) for name, value in params.items()}
        self.v = {name: self.xp.zeros_like(value) for name, value in params.items()}

    def step(self, params, grads):
        self.t += 1
        if self.grad_clip > 0:
            grads = clip_grad_norm(grads, self.grad_clip)
        for name, param in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (self.xp.sqrt(v_hat) + self.eps)

    def state_arrays(self):
        arrays = {"opt_t": np.asarray(self.t, dtype=np.int64)}
        for name in self.m:
            arrays[f"opt_m_{name}"] = asnumpy(self.m[name])
            arrays[f"opt_v_{name}"] = asnumpy(self.v[name])
        return arrays

    def load_state_arrays(self, arrays):
        if "opt_t" not in arrays:
            return False
        for name in self.m:
            m_name = f"opt_m_{name}"
            v_name = f"opt_v_{name}"
            if m_name not in arrays or v_name not in arrays:
                return False
            if arrays[m_name].shape != self.m[name].shape or arrays[v_name].shape != self.v[name].shape:
                return False
        self.t = int(arrays["opt_t"])
        for name in self.m:
            self.m[name][...] = self.xp.asarray(arrays[f"opt_m_{name}"])
            self.v[name][...] = self.xp.asarray(arrays[f"opt_v_{name}"])
        return True


def clip_grad_norm(grads, max_norm):
    first_grad = next(iter(grads.values()))
    xp = array_module(first_grad)
    total_sq = xp.asarray(0.0, dtype=xp.float32)
    for grad in grads.values():
        total_sq = total_sq + xp.sum(grad * grad)
    total_norm = scalar(xp.sqrt(total_sq))
    if total_norm <= max_norm or total_norm == 0.0:
        return grads
    scale = max_norm / (total_norm + 1e-12)
    return {name: grad * scale for name, grad in grads.items()}


class SimpleDenoiser:
    def __init__(self, config, pad_id, mask_id, invalid_sample_ids=None, rng=None, device="cpu"):
        self.config = config
        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.invalid_sample_ids = list(invalid_sample_ids or [])
        self.xp, self.device = get_backend(device)
        rng = rng or np.random.default_rng(config.seed)
        e = config.embed_dim
        h = config.hidden_dim
        v = config.vocab_size
        x = e * 7
        scale = 0.02
        self.token_embedding = self.xp.asarray(rng.normal(0.0, scale, size=(v, e)).astype(np.float32))
        self.position_embedding = self.xp.asarray(rng.normal(0.0, scale, size=(config.max_response_tokens, e)).astype(np.float32))
        self.time_embedding = self.xp.asarray(rng.normal(0.0, scale, size=(config.diffusion_steps + 1, e)).astype(np.float32))
        self.w1 = self.xp.asarray(rng.normal(0.0, np.sqrt(2.0 / x), size=(x, h)).astype(np.float32))
        self.b1 = self.xp.zeros(h, dtype=self.xp.float32)
        self.w2 = self.xp.asarray(rng.normal(0.0, np.sqrt(2.0 / h), size=(h, v)).astype(np.float32))
        self.b2 = self.xp.zeros(v, dtype=self.xp.float32)

    def params(self):
        return {
            "token_embedding": self.token_embedding,
            "position_embedding": self.position_embedding,
            "time_embedding": self.time_embedding,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    def forward(self, prompt_ids, noisy_response_ids, timesteps):
        xp = self.xp
        prompt_ids = xp.asarray(prompt_ids, dtype=xp.int64)
        noisy_response_ids = xp.asarray(noisy_response_ids, dtype=xp.int64)
        timesteps = xp.asarray(timesteps, dtype=xp.int64)

        batch, response_len = noisy_response_ids.shape
        prompt_valid = prompt_ids != self.pad_id
        prompt_counts = xp.maximum(prompt_valid.sum(axis=1, keepdims=True), 1)
        prompt_emb_tokens = self.token_embedding[prompt_ids]
        prompt_mean = (prompt_emb_tokens * prompt_valid[..., None]).sum(axis=1) / prompt_counts

        response_emb = self.token_embedding[noisy_response_ids]
        visible = (noisy_response_ids != self.pad_id) & (noisy_response_ids != self.mask_id)
        visible_counts = xp.maximum(visible.sum(axis=1, keepdims=True), 1)
        visible_mean = (response_emb * visible[..., None]).sum(axis=1) / visible_counts
        visible_context = xp.broadcast_to(visible_mean[:, None, :], response_emb.shape)

        left_context = xp.zeros_like(response_emb)
        left_visible = xp.zeros_like(visible)
        left_context[:, 1:, :] = response_emb[:, :-1, :] * visible[:, :-1, None]
        left_visible[:, 1:] = visible[:, :-1]

        right_context = xp.zeros_like(response_emb)
        right_visible = xp.zeros_like(visible)
        right_context[:, :-1, :] = response_emb[:, 1:, :] * visible[:, 1:, None]
        right_visible[:, :-1] = visible[:, 1:]

        position_emb = self.position_embedding[xp.arange(response_len)][None, :, :]
        position_emb = xp.broadcast_to(position_emb, response_emb.shape)
        time_emb = self.time_embedding[timesteps][:, None, :]
        time_emb = xp.broadcast_to(time_emb, response_emb.shape)
        prompt_context = xp.broadcast_to(prompt_mean[:, None, :], response_emb.shape)

        x = xp.concatenate(
            [response_emb, position_emb, time_emb, prompt_context, visible_context, left_context, right_context],
            axis=-1,
        )
        z1 = x @ self.w1 + self.b1
        hidden = xp.tanh(z1)
        logits = hidden @ self.w2 + self.b2
        cache = {
            "prompt_ids": prompt_ids,
            "noisy_response_ids": noisy_response_ids,
            "timesteps": timesteps,
            "prompt_valid": prompt_valid,
            "prompt_counts": prompt_counts,
            "visible": visible,
            "visible_counts": visible_counts,
            "left_visible": left_visible,
            "right_visible": right_visible,
            "x": x,
            "hidden": hidden,
        }
        return logits, cache

    def loss_and_grads(self, prompt_ids, noisy_response_ids, timesteps, target_response_ids, loss_mask):
        logits, cache = self.forward(prompt_ids, noisy_response_ids, timesteps)
        xp = self.xp
        probs = softmax(logits)
        targets = xp.asarray(target_response_ids, dtype=xp.int64)
        mask = xp.asarray(loss_mask, dtype=xp.float32)
        denom = float(max(scalar(mask.sum()), 1.0))

        clipped = xp.maximum(probs, 1e-12)
        batch_idx = xp.arange(targets.shape[0])[:, None]
        pos_idx = xp.arange(targets.shape[1])[None, :]
        nll = -xp.log(clipped[batch_idx, pos_idx, targets])
        loss = float(scalar((nll * mask).sum() / denom))

        dlogits = probs
        dlogits[batch_idx, pos_idx, targets] -= 1.0
        dlogits *= (mask / denom)[..., None]

        hidden = cache["hidden"]
        x = cache["x"]
        grads = {name: xp.zeros_like(value) for name, value in self.params().items()}
        grads["w2"] = xp.einsum("brh,brv->hv", hidden, dlogits)
        grads["b2"] = dlogits.sum(axis=(0, 1))

        dhidden = dlogits @ self.w2.T
        dz1 = dhidden * (1.0 - hidden * hidden)
        grads["w1"] = xp.einsum("brx,brh->xh", x, dz1)
        grads["b1"] = dz1.sum(axis=(0, 1))

        dx = dz1 @ self.w1.T
        e = self.config.embed_dim
        d_response = dx[..., 0:e]
        d_position = dx[..., e : 2 * e]
        d_time = dx[..., 2 * e : 3 * e]
        d_prompt = dx[..., 3 * e : 4 * e]
        d_visible_context = dx[..., 4 * e : 5 * e]
        d_left_context = dx[..., 5 * e : 6 * e]
        d_right_context = dx[..., 6 * e : 7 * e]

        xp.add.at(grads["token_embedding"], cache["noisy_response_ids"], d_response)
        grads["position_embedding"][: d_position.shape[1]] += d_position.sum(axis=0)
        xp.add.at(grads["time_embedding"], cache["timesteps"], d_time.sum(axis=1))

        d_visible_mean = d_visible_context.sum(axis=1) / cache["visible_counts"]
        for row in range(cache["noisy_response_ids"].shape[0]):
            visible_ids = cache["noisy_response_ids"][row, cache["visible"][row]]
            if len(visible_ids):
                xp.add.at(grads["token_embedding"], visible_ids, d_visible_mean[row])

            left_target_ids = cache["noisy_response_ids"][row, :-1][cache["left_visible"][row, 1:]]
            left_grads = d_left_context[row, 1:][cache["left_visible"][row, 1:]]
            if len(left_target_ids):
                xp.add.at(grads["token_embedding"], left_target_ids, left_grads)

            right_target_ids = cache["noisy_response_ids"][row, 1:][cache["right_visible"][row, :-1]]
            right_grads = d_right_context[row, :-1][cache["right_visible"][row, :-1]]
            if len(right_target_ids):
                xp.add.at(grads["token_embedding"], right_target_ids, right_grads)

        prompt_grad = d_prompt.sum(axis=1) / cache["prompt_counts"]
        for row in range(cache["prompt_ids"].shape[0]):
            ids = cache["prompt_ids"][row, cache["prompt_valid"][row]]
            if len(ids):
                xp.add.at(grads["token_embedding"], ids, prompt_grad[row])

        return loss, grads

    def predict_logits(self, prompt_ids, response_ids, timestep):
        xp = self.xp
        if np.isscalar(timestep):
            timesteps = xp.full((prompt_ids.shape[0],), int(timestep), dtype=xp.int64)
        else:
            timesteps = xp.asarray(timestep, dtype=xp.int64)
        logits, _cache = self.forward(prompt_ids, response_ids, timesteps)
        if self.invalid_sample_ids:
            logits = xp.array(logits, copy=True)
            logits[..., self.invalid_sample_ids] = -1e9
        return logits

    def save(self, path, tokenizer, extra=None, optimizer=None):
        meta = {
            "config": self.config.to_dict(),
            "pad_id": self.pad_id,
            "mask_id": self.mask_id,
            "invalid_sample_ids": self.invalid_sample_ids,
            "tokenizer": tokenizer.to_json(),
            "extra": extra or {},
            "optimizer_saved": optimizer is not None,
        }
        arrays = {name: asnumpy(value) for name, value in self.params().items()}
        if optimizer is not None:
            arrays.update(optimizer.state_arrays())
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path, device="cpu"):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            config = ModelConfig.from_dict(meta["config"])
            tokenizer = WordTokenizer.from_json(meta["tokenizer"])
            model = cls(
                config=config,
                pad_id=meta["pad_id"],
                mask_id=meta["mask_id"],
                invalid_sample_ids=meta.get("invalid_sample_ids", []),
                device=device,
            )
            for name, param in model.params().items():
                param[...] = model.xp.asarray(data[name])
        return model, tokenizer, meta


def load_optimizer_state(path, optimizer):
    with np.load(path, allow_pickle=False) as data:
        return optimizer.load_state_arrays(data)
