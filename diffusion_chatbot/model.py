import json

import numpy as np

from .config import ModelConfig
from .tokenizer import WordTokenizer


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


class AdamW:
    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params, grads):
        self.t += 1
        for name, param in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SimpleDenoiser:
    def __init__(self, config, pad_id, mask_id, invalid_sample_ids=None, rng=None):
        self.config = config
        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.invalid_sample_ids = list(invalid_sample_ids or [])
        rng = rng or np.random.default_rng(config.seed)
        e = config.embed_dim
        h = config.hidden_dim
        v = config.vocab_size
        x = e * 4
        scale = 0.02
        self.token_embedding = rng.normal(0.0, scale, size=(v, e)).astype(np.float32)
        self.position_embedding = rng.normal(0.0, scale, size=(config.max_response_tokens, e)).astype(np.float32)
        self.time_embedding = rng.normal(0.0, scale, size=(config.diffusion_steps + 1, e)).astype(np.float32)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / x), size=(x, h)).astype(np.float32)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, v)).astype(np.float32)
        self.b2 = np.zeros(v, dtype=np.float32)

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
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        noisy_response_ids = np.asarray(noisy_response_ids, dtype=np.int64)
        timesteps = np.asarray(timesteps, dtype=np.int64)

        batch, response_len = noisy_response_ids.shape
        prompt_valid = prompt_ids != self.pad_id
        prompt_counts = np.maximum(prompt_valid.sum(axis=1, keepdims=True), 1)
        prompt_emb_tokens = self.token_embedding[prompt_ids]
        prompt_mean = (prompt_emb_tokens * prompt_valid[..., None]).sum(axis=1) / prompt_counts

        response_emb = self.token_embedding[noisy_response_ids]
        position_emb = self.position_embedding[np.arange(response_len)][None, :, :]
        position_emb = np.broadcast_to(position_emb, response_emb.shape)
        time_emb = self.time_embedding[timesteps][:, None, :]
        time_emb = np.broadcast_to(time_emb, response_emb.shape)
        prompt_context = np.broadcast_to(prompt_mean[:, None, :], response_emb.shape)

        x = np.concatenate([response_emb, position_emb, time_emb, prompt_context], axis=-1)
        z1 = x @ self.w1 + self.b1
        hidden = np.tanh(z1)
        logits = hidden @ self.w2 + self.b2
        cache = {
            "prompt_ids": prompt_ids,
            "noisy_response_ids": noisy_response_ids,
            "timesteps": timesteps,
            "prompt_valid": prompt_valid,
            "prompt_counts": prompt_counts,
            "x": x,
            "hidden": hidden,
        }
        return logits, cache

    def loss_and_grads(self, prompt_ids, noisy_response_ids, timesteps, target_response_ids, loss_mask):
        logits, cache = self.forward(prompt_ids, noisy_response_ids, timesteps)
        probs = softmax(logits)
        targets = np.asarray(target_response_ids, dtype=np.int64)
        mask = np.asarray(loss_mask, dtype=np.float32)
        denom = float(max(mask.sum(), 1.0))

        clipped = np.maximum(probs, 1e-12)
        batch_idx = np.arange(targets.shape[0])[:, None]
        pos_idx = np.arange(targets.shape[1])[None, :]
        nll = -np.log(clipped[batch_idx, pos_idx, targets])
        loss = float((nll * mask).sum() / denom)

        dlogits = probs
        dlogits[batch_idx, pos_idx, targets] -= 1.0
        dlogits *= (mask / denom)[..., None]

        hidden = cache["hidden"]
        x = cache["x"]
        grads = {name: np.zeros_like(value) for name, value in self.params().items()}
        grads["w2"] = np.einsum("brh,brv->hv", hidden, dlogits)
        grads["b2"] = dlogits.sum(axis=(0, 1))

        dhidden = dlogits @ self.w2.T
        dz1 = dhidden * (1.0 - hidden * hidden)
        grads["w1"] = np.einsum("brx,brh->xh", x, dz1)
        grads["b1"] = dz1.sum(axis=(0, 1))

        dx = dz1 @ self.w1.T
        e = self.config.embed_dim
        d_response = dx[..., 0:e]
        d_position = dx[..., e : 2 * e]
        d_time = dx[..., 2 * e : 3 * e]
        d_prompt = dx[..., 3 * e : 4 * e]

        np.add.at(grads["token_embedding"], cache["noisy_response_ids"], d_response)
        grads["position_embedding"][: d_position.shape[1]] += d_position.sum(axis=0)
        np.add.at(grads["time_embedding"], cache["timesteps"], d_time.sum(axis=1))

        prompt_grad = d_prompt.sum(axis=1) / cache["prompt_counts"]
        for row in range(cache["prompt_ids"].shape[0]):
            ids = cache["prompt_ids"][row, cache["prompt_valid"][row]]
            if len(ids):
                np.add.at(grads["token_embedding"], ids, prompt_grad[row])

        return loss, grads

    def predict_logits(self, prompt_ids, response_ids, timestep):
        if np.isscalar(timestep):
            timesteps = np.full((np.asarray(prompt_ids).shape[0],), int(timestep), dtype=np.int64)
        else:
            timesteps = np.asarray(timestep, dtype=np.int64)
        logits, _cache = self.forward(prompt_ids, response_ids, timesteps)
        if self.invalid_sample_ids:
            logits = np.array(logits, copy=True)
            logits[..., self.invalid_sample_ids] = -1e9
        return logits

    def save(self, path, tokenizer, extra=None):
        meta = {
            "config": self.config.to_dict(),
            "pad_id": self.pad_id,
            "mask_id": self.mask_id,
            "invalid_sample_ids": self.invalid_sample_ids,
            "tokenizer": tokenizer.to_json(),
            "extra": extra or {},
        }
        np.savez_compressed(path, meta=json.dumps(meta), **self.params())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            config = ModelConfig.from_dict(meta["config"])
            tokenizer = WordTokenizer.from_json(meta["tokenizer"])
            model = cls(
                config=config,
                pad_id=meta["pad_id"],
                mask_id=meta["mask_id"],
                invalid_sample_ids=meta.get("invalid_sample_ids", []),
            )
            for name, param in model.params().items():
                param[...] = data[name]
        return model, tokenizer, meta
