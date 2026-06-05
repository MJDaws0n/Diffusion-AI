import numpy as np

from .tokenizer import SPECIAL_TOKENS


def sample_response(model, tokenizer, prompt, rng=None, temperature=1.0, top_k=1, show_steps=False):
    rng = rng or np.random.default_rng()
    prompt_ids = np.asarray([tokenizer.encode_prompt(prompt, model.config.max_prompt_tokens)], dtype=np.int64)
    response = np.full((model.config.max_response_tokens,), tokenizer.mask_id, dtype=np.int64)
    stages = []
    stages.append((model.config.diffusion_steps, response.copy()))

    for timestep in range(model.config.diffusion_steps, 0, -1):
        logits = model.predict_logits(prompt_ids, response[None, :], timestep)[0]
        sampled, confidence = choose_tokens(logits, rng, temperature=temperature, top_k=top_k)
        masked_positions = np.flatnonzero(response == tokenizer.mask_id)
        if len(masked_positions) == 0:
            stages.append((timestep - 1, response.copy()))
            break

        remaining_steps = max(timestep, 1)
        commit_count = max(1, int(np.ceil(len(masked_positions) / remaining_steps)))
        ranked = masked_positions[np.argsort(-confidence[masked_positions])]
        commit_positions = ranked[:commit_count]
        response[commit_positions] = sampled[commit_positions]
        stages.append((timestep - 1, response.copy()))

    if np.any(response == tokenizer.mask_id):
        logits = model.predict_logits(prompt_ids, response[None, :], 0)[0]
        sampled, _confidence = choose_tokens(logits, rng, temperature=temperature, top_k=top_k)
        response[response == tokenizer.mask_id] = sampled[response == tokenizer.mask_id]

    text = tokenizer.decode_ids(response, stop_at_eos=True, skip_special=True)
    if show_steps:
        return text, stages
    return text


def choose_tokens(logits, rng, temperature=1.0, top_k=1):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if top_k <= 1:
        ids = logits.argmax(axis=-1)
        confidence = _softmax_confidence(logits, ids)
        return ids.astype(np.int64), confidence

    ids = np.zeros(logits.shape[0], dtype=np.int64)
    confidence = np.zeros(logits.shape[0], dtype=np.float64)
    for pos in range(logits.shape[0]):
        row = logits[pos] / temperature
        top = np.argpartition(row, -top_k)[-top_k:]
        probs = np.exp(row[top] - row[top].max())
        probs /= probs.sum()
        choice = int(rng.choice(top, p=probs))
        ids[pos] = choice
        confidence[pos] = probs[np.where(top == choice)[0][0]]
    return ids, confidence


def _softmax_confidence(logits, ids):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)
    return probs[np.arange(len(ids)), ids]


def format_stage(tokenizer, ids):
    visible = []
    for token in tokenizer.tokens_from_ids(ids):
        if token in SPECIAL_TOKENS:
            visible.append(token)
        else:
            visible.append(token)
    return " ".join(visible)


def final_stage_lines(tokenizer, stages):
    lines = []
    for timestep, ids in stages:
        rendered = format_stage(tokenizer, ids)
        lines.append(f"t={timestep:<2}  {rendered}")
    return lines
