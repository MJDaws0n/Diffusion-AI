import argparse
import csv
import json
import os
import re
import urllib.request
import zipfile


DAILYDIALOG_URL = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and convert online chatbot data to TSV.")
    parser.add_argument("--source", choices=["dailydialog", "dolly", "both"], default="dailydialog")
    parser.add_argument("--out", default="data/pairs.tsv")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--all-adjacent", action="store_true", help="Use every adjacent dialogue turn, not only user->system.")
    args = parser.parse_args(argv)

    os.makedirs(args.cache_dir, exist_ok=True)
    pairs = []

    if args.source in {"dailydialog", "both"}:
        path = os.path.join(args.cache_dir, "dailydialog_data.zip")
        download(DAILYDIALOG_URL, path)
        pairs.extend(load_dailydialog(path, split=args.split, user_to_system_only=not args.all_adjacent))

    if args.source in {"dolly", "both"}:
        path = os.path.join(args.cache_dir, "databricks-dolly-15k.jsonl")
        download(DOLLY_URL, path)
        pairs.extend(load_dolly(path))

    pairs = dedupe_pairs(pairs)
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    write_pairs(args.out, pairs)
    print(f"wrote {len(pairs)} pairs to {args.out}")


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"using cached {path}")
        return
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as src, open(path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    print(f"saved {path}")


def load_dailydialog(path, split="train", user_to_system_only=True):
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read("data/dialogues.json").decode("utf-8"))
    return pairs_from_dailydialog(payload, split=split, user_to_system_only=user_to_system_only)


def pairs_from_dailydialog(dialogues, split="train", user_to_system_only=True):
    pairs = []
    wanted_split = None if split == "all" else split
    for dialogue in dialogues:
        if wanted_split and dialogue.get("data_split") != wanted_split:
            continue
        turns = dialogue.get("turns", [])
        for idx in range(len(turns) - 1):
            prompt_turn = turns[idx]
            reply_turn = turns[idx + 1]
            if user_to_system_only:
                if prompt_turn.get("speaker") != "user" or reply_turn.get("speaker") != "system":
                    continue
            prompt = clean_text(prompt_turn.get("utterance", ""))
            reply = clean_text(reply_turn.get("utterance", ""))
            if is_good_pair(prompt, reply):
                pairs.append((prompt, reply))
    return pairs


def load_dolly(path):
    with open(path, "r", encoding="utf-8") as f:
        return pairs_from_dolly_lines(f)


def pairs_from_dolly_lines(lines):
    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        instruction = clean_text(record.get("instruction", ""))
        context = clean_text(record.get("context", ""))
        response = clean_text(record.get("response", ""))
        prompt = instruction if not context else f"{instruction} {context}"
        if is_good_pair(prompt, response):
            pairs.append((prompt, response))
    return pairs


def clean_text(text):
    text = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_good_pair(prompt, reply):
    if not prompt or not reply:
        return False
    prompt_words = prompt.split()
    reply_words = reply.split()
    if len(prompt_words) < 2 or len(reply_words) < 1:
        return False
    if len(prompt_words) > 80 or len(reply_words) > 80:
        return False
    return True


def dedupe_pairs(pairs):
    out = []
    seen = set()
    for prompt, reply in pairs:
        key = (prompt.lower(), reply.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((prompt, reply))
    return out


def write_pairs(path, pairs):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(pairs)


if __name__ == "__main__":
    main()
