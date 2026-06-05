import argparse
import os
import time

import numpy as np

from .config import ModelConfig
from .data import build_arrays, read_pairs
from .diffusion import MaskDiffusion
from .model import AdamW, SimpleDenoiser


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train pure NumPy masked-diffusion chatbot.")
    parser.add_argument("--data", default="data/pairs.tsv")
    parser.add_argument("--out", default="runs/basic")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=24)
    parser.add_argument("--max-response-tokens", type=int, default=18)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Validate one forward pass only; no training/update/save.")
    args = parser.parse_args(argv)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)

    pairs = read_pairs(args.data)
    tokenizer, prompt_ids, response_ids = build_arrays(pairs, config)
    config.vocab_size = tokenizer.vocab_size
    invalid_sample_ids = [
        tokenizer.pad_id,
        tokenizer.mask_id,
        tokenizer.bos_id,
        tokenizer.sep_id,
        tokenizer.unk_id,
    ]
    model = SimpleDenoiser(
        config,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
        invalid_sample_ids=invalid_sample_ids,
        rng=rng,
    )
    diffusion = MaskDiffusion(steps=config.diffusion_steps)

    batch = make_batch(prompt_ids, response_ids, args.batch_size, diffusion, tokenizer, rng)
    loss, _grads = model.loss_and_grads(*batch)
    if args.dry_run:
        print("dry_run: ok")
        print(f"pairs={len(pairs)} vocab={tokenizer.vocab_size} prompt_shape={prompt_ids.shape} response_shape={response_ids.shape}")
        print(f"initial_loss={loss:.4f}")
        print("no optimizer update, no checkpoint written")
        return

    os.makedirs(args.out, exist_ok=True)
    opt = AdamW(model.params(), lr=args.lr, grad_clip=args.grad_clip)
    started = time.time()
    ema_loss = None

    for step in range(1, args.steps + 1):
        batch = make_batch(prompt_ids, response_ids, args.batch_size, diffusion, tokenizer, rng)
        loss, grads = model.loss_and_grads(*batch)
        opt.step(model.params(), grads)
        ema_loss = loss if ema_loss is None else 0.98 * ema_loss + 0.02 * loss

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - started
            steps_per_sec = step / max(elapsed, 1e-9)
            print(f"step={step} loss={loss:.4f} ema_loss={ema_loss:.4f} steps_per_sec={steps_per_sec:.2f}")

        if step % args.save_every == 0:
            path = os.path.join(args.out, "model.npz")
            model.save(path, tokenizer, extra={"step": step, "loss": loss})
            print(f"saved {path}")

    path = os.path.join(args.out, "model.npz")
    model.save(path, tokenizer, extra={"step": args.steps, "loss": loss})
    print(f"saved {path}")


def make_batch(prompt_ids, response_ids, batch_size, diffusion, tokenizer, rng):
    idx = rng.integers(0, len(prompt_ids), size=batch_size)
    batch_prompts = prompt_ids[idx]
    batch_targets = response_ids[idx]
    timesteps = rng.integers(1, diffusion.steps + 1, size=batch_size)
    noisy, mask_flags = diffusion.noise_batch(
        batch_targets,
        timesteps=timesteps,
        mask_id=tokenizer.mask_id,
        pad_id=tokenizer.pad_id,
        rng=rng,
    )
    loss_mask = mask_flags & (batch_targets != tokenizer.pad_id)
    return batch_prompts, noisy, timesteps, batch_targets, loss_mask


if __name__ == "__main__":
    main()
