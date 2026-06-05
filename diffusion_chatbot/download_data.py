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
    parser.add_argument("--hf-dataset", help="Hugging Face dataset repo id, such as databricks/databricks-dolly-15k.")
    parser.add_argument("--hf-file", help="File path inside the Hugging Face dataset repo.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--format", choices=["dailydialog", "dolly", "jsonl"], help="Converter for --hf-dataset/--hf-file.")
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--context-field", default="")
    args = parser.parse_args(argv)

    pairs = download_and_convert(
        source=args.source,
        out=args.out,
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
    print(f"wrote {len(pairs)} pairs to {args.out}")


def download_and_convert(
    source="dailydialog",
    out="data/pairs.tsv",
    cache_dir="data/raw",
    max_pairs=0,
    split="train",
    all_adjacent=False,
    hf_dataset=None,
    hf_file=None,
    revision="main",
    data_format=None,
    prompt_field="instruction",
    response_field="response",
    context_field="",
):
    os.makedirs(cache_dir, exist_ok=True)
    pairs = []

    if hf_dataset or hf_file:
        if not hf_dataset or not hf_file or not data_format:
            raise ValueError("--hf-dataset, --hf-file, and --format are required together")
        path = download_hf_file(hf_dataset, hf_file, cache_dir=cache_dir, revision=revision)
        pairs.extend(load_by_format(
            path,
            data_format=data_format,
            split=split,
            all_adjacent=all_adjacent,
            prompt_field=prompt_field,
            response_field=response_field,
            context_field=context_field,
        ))
    else:
        if source in {"dailydialog", "both"}:
            path = os.path.join(cache_dir, "dailydialog_data.zip")
            download(DAILYDIALOG_URL, path)
            pairs.extend(load_dailydialog(path, split=split, user_to_system_only=not all_adjacent))

        if source in {"dolly", "both"}:
            path = os.path.join(cache_dir, "databricks-dolly-15k.jsonl")
            download(DOLLY_URL, path)
            pairs.extend(load_dolly(path))

    pairs = dedupe_pairs(pairs)
    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]

    write_pairs(out, pairs)
    return pairs


def download_hf_file(repo_id, filename, cache_dir="data/raw", revision="main"):
    safe_repo = repo_id.replace("/", "__")
    safe_name = filename.replace("/", "__")
    path = os.path.join(cache_dir, f"{safe_repo}__{safe_name}")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    download(url, path)
    return path


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


def load_by_format(path, data_format, split="train", all_adjacent=False, prompt_field="instruction", response_field="response", context_field=""):
    if data_format == "dailydialog":
        return load_dailydialog(path, split=split, user_to_system_only=not all_adjacent)
    if data_format == "dolly":
        return load_dolly(path)
    if data_format == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return pairs_from_jsonl_lines(
                f,
                prompt_field=prompt_field,
                response_field=response_field,
                context_field=context_field,
            )
    raise ValueError(f"Unsupported format: {data_format}")


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


def pairs_from_jsonl_lines(lines, prompt_field="instruction", response_field="response", context_field=""):
    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        prompt = clean_text(get_nested_field(record, prompt_field))
        response = clean_text(get_nested_field(record, response_field))
        context = clean_text(get_nested_field(record, context_field)) if context_field else ""
        if context:
            prompt = f"{prompt} {context}"
        if is_good_pair(prompt, response):
            pairs.append((prompt, response))
    return pairs


def get_nested_field(record, path):
    value = record
    for part in path.split("."):
        if not part:
            continue
        if not isinstance(value, dict):
            return ""
        value = value.get(part, "")
    return value


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
