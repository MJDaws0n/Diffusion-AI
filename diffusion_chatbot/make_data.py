import argparse

from .data import generate_pairs, write_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic dialogue pairs.")
    parser.add_argument("--out", default="data/pairs.tsv", help="Output TSV path.")
    parser.add_argument("--n", type=int, default=50000, help="Number of prompt-response pairs.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    pairs = generate_pairs(n=args.n, seed=args.seed)
    write_pairs(args.out, pairs)
    print(f"wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
