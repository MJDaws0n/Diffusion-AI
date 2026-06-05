import numpy as np


class MaskDiffusion:
    def __init__(self, steps=16, min_mask_prob=0.05, max_mask_prob=0.95):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.min_mask_prob = float(min_mask_prob)
        self.max_mask_prob = float(max_mask_prob)

    def mask_probability(self, timestep):
        t = np.asarray(timestep, dtype=np.float32)
        frac = np.clip(t / float(self.steps), 0.0, 1.0)
        return self.min_mask_prob + (self.max_mask_prob - self.min_mask_prob) * frac

    def noise_batch(self, response_ids, timesteps, mask_id, pad_id, rng):
        noisy = np.array(response_ids, copy=True)
        mask_flags = np.zeros_like(noisy, dtype=bool)
        probs = self.mask_probability(timesteps)

        for row in range(noisy.shape[0]):
            candidates = np.flatnonzero(noisy[row] != pad_id)
            if len(candidates) == 0:
                continue
            chosen = rng.random(len(candidates)) < probs[row]
            if not chosen.any():
                chosen[rng.integers(0, len(candidates))] = True
            positions = candidates[chosen]
            noisy[row, positions] = mask_id
            mask_flags[row, positions] = True
        return noisy, mask_flags
