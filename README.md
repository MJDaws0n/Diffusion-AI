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

## Make Data

```bash
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 100000 --seed 7
```

For even more rows with some repeated patterns:

```bash
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 200000 --seed 7 --allow-duplicates
```

## Train

```bash
python -m diffusion_chatbot.train --data data/pairs.tsv --out runs/basic --steps 8000 --batch-size 64
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
