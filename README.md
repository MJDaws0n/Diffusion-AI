# Diffusion AI

Small Python chatbot that uses masked diffusion for text.

It starts with masked reply tokens:

```text
[MASK] [MASK] [MASK] [MASK]
```

Then it fills them over several steps:

```text
hello [MASK] are [MASK]
hello how are you
```

This is a toy model. It uses pure NumPy, no diffusion libraries, and local synthetic training data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Get Training Data

There is no dataset download.

The training data is made locally by `diffusion_chatbot.make_data`. It writes `data/pairs.tsv`, which is ignored by git because it can be large.

After pulling the repo on another machine:

```bash
git pull
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 100000 --seed 7
```

Use this if you want cleaner loss and less prompt confusion:

```bash
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 12000 --seed 7 --stable-prompts
```

Use this if you want more rows, with repeated patterns allowed:

```bash
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 200000 --seed 7 --allow-duplicates
```

The small `data/sample_pairs.tsv` file is only for tests and dry runs.

## Train

```bash
python -m diffusion_chatbot.train --data data/pairs.tsv --out runs/basic --steps 12000 --batch-size 128 --lr 0.0015
```

## Preview Diffusion

```bash
python -m diffusion_chatbot.preview --ckpt runs/basic/model.npz --prompt "hello how are you"
```

## Chat

```bash
python -m diffusion_chatbot.chat --ckpt runs/basic/model.npz --show-steps
```

## Test Without Training

```bash
python -m pytest -q
python -m diffusion_chatbot.train --data data/sample_pairs.tsv --out runs/dry --dry-run
```
