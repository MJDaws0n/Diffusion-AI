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

Best option: download real online dialogue data.

This downloads DailyDialog from Hugging Face and converts dialogue turns into `data/pairs.tsv`:

```bash
python -m diffusion_chatbot.download_data --source ConvLab/dailydialog --out data/pairs.tsv
```

For more instruction-style question/answer data too:

```bash
python -m diffusion_chatbot.download_data --source databricks/databricks-dolly-15k --out data/pairs.tsv
```

Raw downloads are cached in `data/raw/`. `data/pairs.tsv` and `data/raw/` are ignored by git because they can be large.

Sources:

- DailyDialog via Hugging Face `ConvLab/dailydialog`, CC BY-NC-SA 4.0.
- Databricks Dolly 15k via Hugging Face, CC BY-SA 3.0.

Fallback synthetic data still exists:

```bash
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 100000 --seed 7
```

Cleaner synthetic data:

```bash
python -m diffusion_chatbot.make_data --out data/pairs.tsv --n 12000 --seed 7 --stable-prompts
```

The small `data/sample_pairs.tsv` file is only for tests and dry runs.

## Train

```bash
python -m diffusion_chatbot.train --data data/pairs.tsv --out runs/basic --steps 20000 --batch-size 128 --lr 0.001 --vocab-size 4096 --max-prompt-tokens 40 --max-response-tokens 40
```

Continue from a checkpoint:

```bash
python -m diffusion_chatbot.train --data data/pairs.tsv --out runs/basic --resume runs/basic/model.npz --steps 5000 --batch-size 128 --lr 0.001
```

## Download And Train

One command can pull Hugging Face data and train:

```bash
python -m diffusion_chatbot.train_hf --source ConvLab/dailydialog --out runs/basic
```

Continue Hugging Face training from a checkpoint:

```bash
python -m diffusion_chatbot.train_hf --source ConvLab/dailydialog --out runs/basic --resume runs/basic/model.npz --steps 5000
```

Mixed DailyDialog + Dolly:

```bash
python -m diffusion_chatbot.train_hf --source both --out runs/basic
```

Custom Hugging Face JSONL dataset:

```bash
python -m diffusion_chatbot.train_hf --source owner/dataset-name --hf-file file.jsonl --format jsonl --prompt-field prompt --response-field response --out runs/basic
```

Test the full pipeline without training:

```bash
python -m diffusion_chatbot.train_hf --source ConvLab/dailydialog --max-pairs 128 --dry-run
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
