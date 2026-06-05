import argparse

import numpy as np

from .model import SimpleDenoiser
from .sampler import final_stage_lines, sample_response


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview reverse diffusion steps for one prompt.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args(argv)

    model, tokenizer, _meta = SimpleDenoiser.load(args.ckpt)
    rng = np.random.default_rng(args.seed)
    text, stages = sample_response(
        model,
        tokenizer,
        args.prompt,
        rng=rng,
        temperature=args.temperature,
        top_k=args.top_k,
        show_steps=True,
    )
    print(f"prompt: {args.prompt}")
    print()
    for line in final_stage_lines(tokenizer, stages):
        print(line)
    print()
    print(f"bot: {text}")


if __name__ == "__main__":
    main()
