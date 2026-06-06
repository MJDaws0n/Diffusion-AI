import argparse
import time

from .backend import backend_report, get_backend, scalar, synchronize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark CPU/CUDA backend matrix math.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args(argv)

    xp, device = get_backend(args.device)
    print(backend_report(xp, device))

    rows = args.batch_size * args.tokens
    in_dim = args.embed_dim * 7
    x = xp.asarray(xp.random.standard_normal((rows, in_dim)), dtype=xp.float32)
    w1 = xp.asarray(xp.random.standard_normal((in_dim, args.hidden_dim)), dtype=xp.float32)
    w2 = xp.asarray(xp.random.standard_normal((args.hidden_dim, args.vocab_size)), dtype=xp.float32)

    for _ in range(3):
        hidden = x @ w1
        logits = hidden @ w2
        checksum = logits.sum()
    synchronize(xp)

    start = time.time()
    for _ in range(args.iters):
        hidden = x @ w1
        logits = hidden @ w2
        checksum = logits.sum()
    synchronize(xp)
    elapsed = time.time() - start

    print(f"shape1=({rows},{in_dim})x({in_dim},{args.hidden_dim})")
    print(f"shape2=({rows},{args.hidden_dim})x({args.hidden_dim},{args.vocab_size})")
    print(f"iters={args.iters} seconds={elapsed:.3f} iters_per_sec={args.iters / elapsed:.2f}")
    print(f"checksum={float(scalar(checksum)):.3f}")


if __name__ == "__main__":
    main()
