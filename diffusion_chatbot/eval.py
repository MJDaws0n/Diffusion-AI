import argparse

import numpy as np

from .model import SimpleDenoiser
from .sampler import sample_response


DEFAULT_PROMPTS = [
    "hello",
    "how are you",
    "i feel sad",
    "can you help me with python",
    "what are you",
    "thanks",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run fixed prompt eval samples.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)

    model, tokenizer, _meta = SimpleDenoiser.load(args.ckpt)
    rng = np.random.default_rng(args.seed)
    for prompt in DEFAULT_PROMPTS:
        reply = sample_response(
            model,
            tokenizer,
            prompt,
            rng=rng,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"you: {prompt}")
        print(f"bot: {reply}")
        print()


if __name__ == "__main__":
    main()
