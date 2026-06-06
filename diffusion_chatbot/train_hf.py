import argparse

from .download_data import download_and_convert
from .train import main as train_main


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download Hugging Face data and train the chatbot.")

    parser.add_argument("--source", default="ConvLab/dailydialog", help="Dataset alias or Hugging Face repo id.")
    parser.add_argument("--pairs-out", default="data/pairs.tsv")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--all-adjacent", action="store_true")

    parser.add_argument("--hf-dataset", help="Deprecated alias for --source with a Hugging Face repo id.")
    parser.add_argument("--hf-file", help="File path inside the custom HF dataset repo.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--format", choices=["dailydialog", "dolly", "jsonl"])
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--context-field", default="")

    parser.add_argument("--out", default="runs/basic")
    parser.add_argument("--resume", help="Checkpoint path to continue training from.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-prompt-tokens", type=int, default=40)
    parser.add_argument("--max-response-tokens", type=int, default=40)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pairs = download_and_convert(
        source=args.source,
        out=args.pairs_out,
        cache_dir=args.cache_dir,
        max_pairs=args.max_pairs,
        split=args.split,
        all_adjacent=args.all_adjacent,
        hf_dataset=args.hf_dataset,
        hf_file=args.hf_file,
        revision=args.revision,
        data_format=args.format,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        context_field=args.context_field,
    )
    print(f"data ready: {len(pairs)} pairs at {args.pairs_out}")
    print("starting trainer")

    train_args = [
        "--data", args.pairs_out,
        "--out", args.out,
        "--device", args.device,
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--grad-clip", str(args.grad_clip),
        "--vocab-size", str(args.vocab_size),
        "--max-prompt-tokens", str(args.max_prompt_tokens),
        "--max-response-tokens", str(args.max_response_tokens),
        "--embed-dim", str(args.embed_dim),
        "--hidden-dim", str(args.hidden_dim),
        "--diffusion-steps", str(args.diffusion_steps),
        "--seed", str(args.seed),
        "--log-every", str(args.log_every),
        "--save-every", str(args.save_every),
    ]
    if args.resume:
        train_args.extend(["--resume", args.resume])
    if args.dry_run:
        train_args.append("--dry-run")
    train_main(train_args)


if __name__ == "__main__":
    main()
