import argparse

import numpy as np

from .model import SimpleDenoiser
from .sampler import final_stage_lines, sample_response


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive masked-diffusion chatbot.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args(argv)

    model, tokenizer, _meta = SimpleDenoiser.load(args.ckpt)
    rng = np.random.default_rng(args.seed)
    print("diffusion chatbot ready. type /quit to exit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            break
        result = sample_response(
            model,
            tokenizer,
            prompt,
            rng=rng,
            temperature=args.temperature,
            top_k=args.top_k,
            show_steps=args.show_steps,
        )
        if args.show_steps:
            text, stages = result
            for line in final_stage_lines(tokenizer, stages):
                print(line)
            print(f"bot> {text}")
        else:
            print(f"bot> {result}")


if __name__ == "__main__":
    main()
