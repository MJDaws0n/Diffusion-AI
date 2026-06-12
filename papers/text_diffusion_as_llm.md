# Text Diffusion as a Language Model: Notes From a Small Practical Experiment

## Abstract

Most modern large language models generate text autoregressively: they predict the next token, append it, then repeat. This method is simple, powerful, and well supported by the Transformer architecture, but it is not the only possible way to model language. Text diffusion models instead treat generation as an iterative denoising problem. A sequence may begin as noise, masks, or continuous latent vectors, and the model gradually turns that corrupted sequence into readable language.

This paper argues that text diffusion is a serious alternative direction for language modeling, but not yet a drop-in replacement for mainstream autoregressive LLMs. Diffusion may be better for parallel token generation, editing, infilling, controllable generation, and bidirectional structure. It may be worse for long-form causal reasoning, efficient deployment, and scaling in small-resource environments. The experiments in this repository support that mixed view: a small masked diffusion chatbot learned local word structure quickly and produced recognizable English earlier than expected, but useful chatbot behavior required more data, larger models, GPU acceleration, and more training than was available in the initial experiment.

## 1. Background

The Transformer became the dominant language model architecture after *Attention Is All You Need* introduced a scalable attention-based sequence model without recurrence or convolution [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). The most successful chat LLMs usually combine Transformer blocks with an autoregressive objective: predict token `x_t` from tokens `x_1 ... x_{t-1}`.

Autoregression has a strong practical advantage. It aligns naturally with text: words are read left to right, and generation can be trained by next-token prediction over huge corpora. The downside is that sampling is sequential. Each token depends on the previous token, so generating 1,000 tokens requires roughly 1,000 decode steps. Systems can optimize this heavily, but the core dependency remains.

Diffusion models became famous in images and audio. In a denoising diffusion probabilistic model, training corrupts data through a forward noising process and learns a reverse denoising process [Ho et al., 2020](https://arxiv.org/abs/2006.11239). For images, the data are continuous pixels or latent vectors. Text is harder because tokens are discrete. Replacing "slightly noised image" with "slightly noised sentence" is not straightforward.

Several research lines adapt diffusion to language:

- **Continuous text diffusion** maps tokens into continuous embeddings, applies Gaussian diffusion there, then decodes back into tokens. Diffusion-LM showed this can help controllable text generation [Li et al., 2022](https://arxiv.org/abs/2205.14217).
- **Sequence-to-sequence diffusion** corrupts a target sequence while conditioning on a source sequence. DiffuSeq showed diffusion can work for conditional generation and can produce diverse outputs [Gong et al., 2022](https://arxiv.org/abs/2210.08933).
- **Masked-token diffusion** treats generation as gradually filling masked tokens. LLaDA scales this idea much further, training a large language diffusion model with a masking forward process and a reverse process that predicts masked tokens [Nie et al., 2025](https://arxiv.org/abs/2502.09992).

This repository follows the masked-token route in a very small way. A reply begins as:

```text
[MASK] [MASK] [MASK] [MASK]
```

and is gradually denoised:

```text
hello [MASK] are [MASK]
hello how are you
```

The implementation is deliberately simple: a tokenizer, a masked noising schedule, a small denoiser, a sampler, and a training loop written directly in Python/NumPy, later extended with optional CuPy for NVIDIA GPUs.

## 2. What "Text Diffusion as an LLM" Means

There are two separate questions that often get merged:

1. Can diffusion replace the **autoregressive objective**?
2. Can diffusion replace the **Transformer architecture**?

The first question is more realistic. A diffusion language model can still use Transformer blocks internally. LLaDA, for example, uses a Transformer-like denoiser but changes the generation objective from next-token prediction to masked-token denoising [Nie et al., 2025](https://arxiv.org/abs/2502.09992). In that sense, diffusion competes with autoregression more than with attention itself.

The second question is harder. A non-Transformer denoiser can work for toy experiments, but attention is still extremely useful for language because every token may depend on every other token. My experiment initially used a simple MLP-style denoiser with token, position, timestep, prompt, and visible-response context. This was enough to learn basic local structure, but not enough to become a strong conversational model.

So, the practical framing is:

> Text diffusion is not necessarily "no Transformers." It is more accurately "not forced to decode one token at a time."

## 3. A Minimal Masked-Diffusion Chatbot Algorithm

The prototype in this repository uses a discrete masked diffusion process. It does not add Gaussian noise to text. Instead, it randomly replaces response tokens with `[MASK]` and trains the model to reconstruct the original response while conditioning on the user prompt.

The training example starts as a prompt-response pair:

```text
prompt:   hello
response: hello how are you [EOS]
```

At a sampled timestep, the forward corruption process masks part of the response:

```text
noisy response: hello [MASK] are [MASK] [EOS]
```

The denoiser receives the prompt, the noisy response, positions, and timestep. It predicts a vocabulary distribution for every response position. The loss is cross-entropy on the original clean response tokens:

```text
loss = CE(model(prompt, noisy_response, timestep), clean_response)
```

Sampling reverses the process. A response begins fully masked. At each denoising step, the model predicts all positions, commits some high-confidence tokens, and leaves the rest masked for later refinement:

```text
tokens = [MASK] [MASK] [MASK] [MASK]

for t in reversed(range(num_steps)):
    logits = model(prompt, tokens, t)
    predictions = argmax_or_sample(logits)
    confidence = max_softmax(logits)
    commit confident positions
    keep uncertain positions as [MASK]
```

This is close in spirit to the masking process used in larger masked diffusion language models, but much smaller and less sophisticated. LLaDA, for example, uses a Transformer to predict masked tokens and samples through a reverse unmasking process [Nie et al., 2025](https://arxiv.org/abs/2502.09992). Recent masked diffusion work also shows that simple masked objectives can be much stronger when paired with modern training recipes and efficient samplers [Sahoo et al., 2024](https://arxiv.org/abs/2406.07524).

The important conceptual difference from autoregression is that the model does not have to decide token 7 only after token 6 has been finalized. It can propose a whole answer, then revise uncertain parts. That makes the generation process feel more like drafting and editing than like typing from left to right.

## 4. Comparison With Autoregressive LLMs

| Question | Autoregressive LLM | Masked diffusion LM |
| --- | --- | --- |
| Training target | Predict next token from previous tokens | Predict clean tokens from corrupted or masked sequence |
| Generation order | Strictly left to right | Many positions can update in parallel |
| First-token latency | Usually strong because decoding starts immediately | Often worse because a block must be denoised |
| Full-output latency | Good with optimized serving and KV cache | Potentially strong if few denoising steps fill many tokens |
| Editing/infilling | Possible, but not the native objective | Native to the objective |
| Global revision | Hard after a token is emitted | Natural before final commitment |
| Tooling maturity | Extremely mature | Much less mature |
| Best use today | General chat, long reasoning, production serving | Editing, infilling, bounded outputs, research systems |

The table hides an important point: diffusion is not one thing. A text diffusion model may be continuous or discrete, word-level or subword-level, MLP-based or Transformer-based, small and educational or commercial-scale. The weak result from a tiny toy model does not invalidate the larger idea. Likewise, a strong result from a commercial model does not mean a hand-written prototype should be expected to beat a mature autoregressive stack.

Recent public examples show both sides. LLaDA argues that a diffusion model trained from scratch can challenge the assumption that core LLM capabilities require autoregression [Nie et al., 2025](https://arxiv.org/abs/2502.09992). Mercury reports commercial-scale diffusion LLMs for code that predict multiple tokens in parallel and emphasize the speed-quality tradeoff [Inception Labs et al., 2025](https://arxiv.org/abs/2506.17298). Google DeepMind describes Gemini Diffusion as an experimental text diffusion model focused on control, speed, iterative refinement, and block-level generation [Google DeepMind, 2025](https://deepmind.google/models/gemini-diffusion/). These systems are far beyond this repository, but they make the direction less speculative than it would have seemed a few years ago.

At the same time, the balanced research view is still cautious. Work evaluating score-entropy discrete diffusion describes real promise, including possible latency advantages, but also notes shortcomings such as weaker conditional generation from short prompts [Deschenaux and Gulcehre, 2024](https://arxiv.org/html/2406.11473v1). That matches the prototype experience: the model could learn phrase shape quickly, but short user prompts did not always anchor the response well.

## 5. Why Diffusion Could Be Better

### 5.1 Parallel Generation

Autoregressive decoding generates token 1, then token 2, then token 3. Diffusion can predict many masked positions at once during each reverse step. Even if it needs 16 or 32 denoising steps, each step can update the full sequence in parallel.

This is attractive for hardware. GPUs are good at large matrix operations. If a diffusion model can fill 64 tokens over 16 denoising steps, it may reduce the strict sequential dependency of left-to-right decoding. The speed benefit depends on implementation quality, number of denoising steps, model size, and whether the model needs repeated refinement.

In this project, the effect is visible conceptually but not fully realized computationally. The toy model does update multiple token positions per step, but the hand-written CuPy training loop has Python overhead and unfused scatter operations. On an RTX 3060, GPU utilization reached 100%, but throughput was still limited by the simplicity and inefficiency of the experimental implementation.

### 5.2 Bidirectional Structure

A masked diffusion model can condition on tokens to the left and right of a masked position. This resembles the advantage of masked language modeling in BERT, which was designed to learn bidirectional representations from unlabeled text [Devlin et al., 2018](https://arxiv.org/abs/1810.04805).

This matters because natural language is not purely left-to-right at the planning level. When humans write, they often revise earlier words after knowing later words. A diffusion model naturally supports that kind of revision. It can start with a rough global answer and refine it.

In the experiment, this was the most noticeable early positive signal. Even small models learned short phrase structure faster than expected. Outputs like:

```text
hello how are you
```

appeared after limited training. The model also learned that emotional prompts should often produce apologetic or supportive replies. The quality was shallow, but the word-shape and phrase-shape appeared early.

### 5.3 Infilling and Editing

Diffusion is naturally suited to infilling. If some tokens are known and others are masked, generation becomes "fill the gaps." This is useful for:

- editing a sentence while preserving parts of it
- completing a response with constraints
- repairing malformed output
- generating multiple alternative completions

Autoregressive models can do infilling, but they are not inherently designed around it. They often need special prompting or fill-in-the-middle training. Diffusion has infilling at the center of the objective.

### 5.4 Controllability

Diffusion-LM was motivated by controllable generation, especially fine-grained control such as syntactic structure [Li et al., 2022](https://arxiv.org/abs/2205.14217). The iterative latent path gives more places to guide generation than a single next-token distribution.

For chatbot-style systems, this suggests useful future controls:

- response length
- tone
- required keywords
- style
- safety constraints
- answer format

The experiment in this repository did not implement advanced control. However, the visible denoising preview made the generation process inspectable. Watching `[MASK]` tokens become words is useful for debugging because it exposes when the model commits to bad structure too early.

### 5.5 Diversity

DiffuSeq reports diversity as one of the interesting properties of sequence-to-sequence diffusion [Gong et al., 2022](https://arxiv.org/abs/2210.08933). Diversity is valuable in open-ended generation because many prompts do not have one correct answer.

This matters for chatbots. A deterministic next-token system can collapse into common replies. A diffusion system can sample different denoising paths and produce different valid responses. The downside is that diversity without strong modeling becomes nonsense. The small model often produced repeated or semantically confused words when undertrained.

## 6. Why Diffusion Could Be Worse

### 6.1 More Sampling Steps

Autoregressive generation needs one forward pass per generated token, but each pass benefits from KV caching and highly optimized inference. Diffusion needs multiple denoising passes over the whole sequence. If it uses 16, 32, or 64 reverse steps, the cost can become large.

A diffusion model must win by making each step parallel and effective. If each denoising step is expensive and the implementation is not optimized, it may be slower than an autoregressive model.

This was visible in the experiment. The CuPy version did run on an RTX 3060 and reached 100% GPU utilization, but a large configuration around:

```text
batch_size=96
vocab_size=12000
embed_dim=192
hidden_dim=384
max_response_tokens=64
```

ran around 0.8-0.9 steps per second. Larger batches did not improve examples per second much. The GPU was busy, but the hand-written model was not as efficient as a fused deep learning framework.

### 6.2 The Discrete Token Problem

Diffusion was first very successful in continuous spaces. Text tokens are discrete. If the model corrupts tokens by masking them, the noising process is not the same as adding small Gaussian noise to pixels. If the model uses continuous embeddings, it must map back to valid tokens.

Both approaches have tradeoffs:

- **Masked discrete diffusion** is simple and directly token-based, but the transition process is coarse.
- **Continuous embedding diffusion** may be smoother, but decoding back to text introduces extra complexity.

This project used masked discrete diffusion because it is easy to inspect and implement. That made the experiment understandable, but it likely limited expressiveness.

### 6.3 Long Reasoning and Causal Chains

Autoregressive models fit step-by-step reasoning naturally: each generated token can condition on the reasoning so far. Diffusion models can revise globally, but they may struggle to maintain a stable chain of thought unless the denoising schedule, architecture, and training objective support it.

This does not mean diffusion cannot reason. LLaDA reports competitive behavior at much larger scale [Nie et al., 2025](https://arxiv.org/abs/2502.09992). But small diffusion models are not automatically good reasoners. My experiment produced English-like fragments before it produced consistently useful answers.

### 6.4 Training Instability and Ambiguous Targets

The first synthetic dataset in this project generated many prompt-response pairs from templates. It was useful for proving the pipeline, but it was bad data. Some prompts mapped to several possible replies. Cross-entropy loss cannot fall cleanly when the same exact input has multiple conflicting targets.

This led to early plateaus. A small synthetic run fell quickly from high loss to a plateau, and the model could answer some dataset-like prompts but remained shallow. Replacing synthetic data with DailyDialog from Hugging Face improved the realism of the training pairs, but required more training and a larger model.

### 6.5 Ecosystem Disadvantage

Autoregressive Transformers have an enormous ecosystem:

- tokenizers
- pretrained checkpoints
- serving engines
- KV-cache optimization
- quantization
- evaluation harnesses
- instruction-tuning recipes

Diffusion language models have less mature tooling. In this project, even basic GPU support required adding CuPy manually and dealing with missing CUDA component libraries such as NVRTC and cuBLAS. That engineering overhead is part of the current downside.

## 7. Personal Experiment

The goal of this repository was not to beat existing LLMs. It was to build a working chatbot that uses diffusion-style denoising instead of normal left-to-right output. The target behavior was:

```text
[MASK] [MASK] [MASK] [MASK]
-> Hello [MASK] are [MASK]
-> Hello how are you
```

The first implementation used:

- word-level tokenizer
- `[MASK]`, `[PAD]`, `[BOS]`, `[SEP]`, `[EOS]`, `[UNK]` special tokens
- synthetic prompt-response data
- a masked noising schedule
- a small denoiser written without existing diffusion libraries
- reverse sampling that gradually commits high-confidence tokens

The early results were encouraging but limited. The model learned basic word order and short phrases quickly. It could produce outputs like:

```text
bot> hello how are you
```

For emotional prompts, it learned rough support patterns:

```text
you> hello, i am sad
bot> i am hard you better soon
```

That answer is not good English, but it shows partial structure: first-person phrasing, an emotional context, and a "better soon" support phrase. The failure is also informative. The model mixed fragments from different training replies because the data and model were too small.

Several improvements followed:

- The synthetic generator was expanded, then partially replaced by real DailyDialog data.
- A Hugging Face downloader was added so data can be pulled with commands like:

```bash
python -m diffusion_chatbot.download_data --source ConvLab/dailydialog --out data/pairs.tsv
```

- Checkpoint resume was added.
- Optional NVIDIA GPU training was added through CuPy.
- A benchmark command was added to confirm whether CUDA is actually active.

On the RTX 3060 machine, `nvidia-smi` showed the Python process using around 3 GB of VRAM and 100% GPU utilization during training. That confirms the GPU path was active. However, throughput remained modest because the model is a manually written MLP-like denoiser with large output logits and scatter-heavy gradient accumulation.

The training logs also showed why raw loss can be misleading. In one small synthetic run, loss fell quickly from roughly 4.8 to around 0.8, then moved sideways for thousands of steps. The chatbot could answer a few prompt types from the dataset, but the generated language still had repeated fragments and odd substitutions. This is a useful failure mode: the model had learned the surface distribution of the toy data, but not enough general conversational structure.

The larger CUDA run had a second practical lesson. A bigger batch did not automatically mean better throughput. The batch-96 run reported around 0.8-0.9 steps per second, while larger batches could show lower steps per second even when the total examples per second stayed similar. That suggests the bottleneck was not only GPU occupancy. The output projection over a 12,000-token vocabulary and the hand-written gradient accumulation both matter. In a serious implementation, these operations should be handled by an optimized tensor framework or custom kernels.

The main personal observation is:

> The model seemed to learn local word structure faster than expected, but scaling it into a useful chatbot required more data, more training time, and a stronger denoiser than was available in the initial experiment.

That result is consistent with the broader literature. Diffusion language models are promising, but the strongest evidence appears at much larger scale, with careful objectives and Transformer-class denoisers [Nie et al., 2025](https://arxiv.org/abs/2502.09992).

## 8. What the Experiment Suggests

The experiment is useful because it separates "can this work at all?" from "is this currently competitive?" The answer to the first question was yes. The answer to the second was no.

The most positive result was speed of early structure learning. Even a small model trained on basic prompt-response pairs learned that greetings should produce greeting-shaped replies. It also learned that a response normally has a short phrase shape rather than random tokens. The denoising trace made this visible. For example, a successful run did not jump instantly from masks to a final answer; it gradually exposed a skeleton:

```text
t=16  [MASK] [MASK] [MASK] [MASK]
t=13  hello [MASK] [MASK] [MASK]
t=11  hello [MASK] are [MASK]
t=10  hello how are [MASK]
t=8   hello how are you
```

That behavior is exactly why text diffusion is interesting. The model can commit confident words while leaving uncertain positions for later. The output feels like a rough draft becoming a sentence.

The negative result was semantic brittleness. On a sad-user prompt, the prototype produced:

```text
you> hello, i am sad
bot> i am hard you better soon
```

This is not a good chatbot answer. It is grammatically broken and emotionally clumsy. Still, it reveals a partial learning signal: the model latched onto "I am ..." and "... better soon" patterns but failed to bind them into a coherent reply. In other words, it learned phrase fragments before it learned meaning.

That pattern matches what should be expected from the setup. The first model had limited capacity, word-level tokens, synthetic data, and no attention. It could memorize and recombine local patterns. It could not reliably track the user's emotional state or choose a response with stable intent.

The later DailyDialog path was the right direction because it replaced weak synthetic examples with real dialogue turns. The downloader supports:

```bash
python -m diffusion_chatbot.download_data --source ConvLab/dailydialog --out data/pairs.tsv
```

and the same mechanism supports instruction-style sources such as Dolly:

```bash
python -m diffusion_chatbot.download_data --source databricks/databricks-dolly-15k --out data/pairs.tsv
```

DailyDialog is closer to the intended chatbot behavior. Dolly is less conversational, but useful for instruction-response shape. A stronger training run should likely mix dialogue, instruction, and small high-quality hand-authored examples instead of relying on a synthetic generator.

## 9. Interpreting the Loss Plateau

A recurring result in the experiment was that the loss fell quickly, then plateaued. This can happen for several reasons:

1. **Ambiguous data**: one prompt can have many valid replies.
2. **Weak architecture**: the denoiser may not have enough context modeling.
3. **Large vocabulary cost**: predicting over thousands of tokens is hard.
4. **Short training**: diffusion models may need more steps to refine sequence-level behavior.
5. **Word-level tokenization**: rare words become sparse targets, making generalization harder.

The plateau does not mean the idea failed. It means the prototype reached the capacity of its data, architecture, and compute budget. In traditional LLM terms, the experiment was closer to a small learned phrase model than a real LLM.

The plateau also has a measurement trap. Cross-entropy over a full vocabulary punishes every valid alternative answer that does not match the dataset target. If one prompt could reasonably be answered with "I am sorry to hear that", "That sounds hard", or "I hope you feel better soon", the training loss treats two of those as wrong for a given example. A high plateau can therefore mean the model is bad, the data is ambiguous, or both.

The more useful question is whether validation samples improve. For this project, the best next measurement would be a small fixed evaluation suite:

- greeting prompts
- mood prompts
- factual toy prompts
- short instruction prompts
- held-out DailyDialog turns
- diversity samples from the same prompt

The model should be judged by readable outputs, diversity without collapse, and whether denoising traces become cleaner over time. Loss alone is not enough.

## 10. Threats to Validity

This paper is partly based on a personal experiment, so the evidence has limits.

First, the implementation is intentionally simple. It is not a fair benchmark against PyTorch, JAX, vLLM, FlashAttention, or production autoregressive inference. The GPU numbers show that CUDA was active, not that the implementation is optimized.

Second, the model is too small to make strong claims about language-model scaling. A word-level MLP-style denoiser cannot represent language the way a Transformer denoiser can. If the prototype fails, that may say more about the architecture than about diffusion.

Third, the early data was weak. Synthetic prompt-response generation was useful for smoke testing, but bad synthetic data teaches bad behavior. The model's strange replies were partly a data problem.

Fourth, the experiment was resource-limited. The observed result was that local word structure appeared quickly, but there was not enough time or compute to push the model into a stronger regime. Many generative modeling ideas look poor at tiny scale and become interesting only after the architecture, data, and optimization recipe are right.

The conclusion should therefore be modest: this repository demonstrates a working masked diffusion chatbot prototype and gives practical intuition for the idea. It does not prove that text diffusion beats autoregressive LLMs.

## 11. When Diffusion Might Win

Text diffusion is most likely to be useful when:

- output length is known or bounded
- infilling/editing is central
- diversity matters
- global consistency matters more than streaming token-by-token output
- generation can be done in parallel blocks
- the model is large enough to learn strong bidirectional structure

Examples:

- rewriting a paragraph
- filling missing parts of code or text
- generating multiple candidate answers
- controlled dialogue response generation
- structured text where the whole output shape matters

The key advantage is that the model can look at and revise the whole answer during generation.

## 12. When Autoregression Still Wins

Autoregressive LLMs are still the default choice when:

- streaming output matters
- latency for first token matters
- very long generation is needed
- tooling and serving reliability matter
- maximum reasoning quality is needed today
- pretrained models and fine-tuning recipes are required

Autoregressive Transformers have been scaled, optimized, and studied for years. Diffusion language models are newer and less mature. A toy diffusion chatbot is not a competitor to GPT-style systems. The more realistic claim is that diffusion offers a different path that might become competitive when scaled properly.

## 13. Practical Recommendation

For this repository, the practical path is not to keep making the MLP wider forever. The current model is a good educational prototype, but the next serious version should change the architecture.

The recommended path is:

1. Keep masked diffusion as the generation objective.
2. Replace the denoiser with a small Transformer encoder.
3. Replace word tokens with subword tokens.
4. Train first on real dialogue and instruction data.
5. Add validation samples that are printed at fixed intervals.
6. Compare against a tiny autoregressive baseline trained on the same data.

That comparison matters. Without a baseline, it is easy to mistake "cool visible diffusion steps" for actual modeling improvement. A tiny autoregressive model trained on the same data would answer whether diffusion is learning faster, producing more diverse outputs, or simply producing different failure modes.

The best near-term use of this project is as a research playground: a small system where the denoising process is visible and modifiable. It should not be framed as a production chatbot yet.

## 14. Future Work

The next serious version of this project would need:

1. **Transformer denoiser**: replace the MLP-style denoiser with self-attention.
2. **Subword tokenizer**: use BPE or unigram tokenization instead of word-level tokens.
3. **Better noising schedule**: tune mask rates and token commitment strategy.
4. **Length modeling**: predict or condition response length rather than using fixed maximum length.
5. **Better evaluation**: track validation loss, exact-match on simple tasks, diversity, and human-rated coherence.
6. **Fused GPU framework**: use PyTorch, JAX, Triton, or custom kernels for efficient training.
7. **Instruction tuning**: train on real instruction/chat data after base denoising pretraining.
8. **Hybrid decoding**: combine diffusion planning with autoregressive finalization.

The most important architectural improvement would be attention. Diffusion changes the generation process, but language still needs long-range token interaction. A diffusion objective plus a Transformer denoiser is likely a stronger path than trying to remove Transformers entirely.

The next experiment should answer a few concrete questions:

- How many denoising steps are actually needed before quality stops improving?
- Does remasking low-confidence tokens improve coherence or only add noise?
- Does mixed dialogue plus instruction data reduce the plateau compared with dialogue alone?
- Does a small Transformer denoiser beat a similarly sized autoregressive baseline on the same data?
- Can validation samples separate surface grammar from actual response relevance?

Those questions are more useful than asking whether diffusion is "better" in the abstract. The answer depends on the task, model size, sampling budget, and implementation.

## 15. Conclusion

Text diffusion is a credible alternative to standard autoregressive language modeling, but it is not magic. It offers appealing properties: parallel updates, bidirectional conditioning, natural infilling, visible refinement, and possible controllability. It also brings real problems: multiple denoising steps, discrete-token difficulty, weaker tooling, and uncertain reasoning behavior at small scale.

The experiment in this repository supports a cautious but optimistic conclusion. Even a small masked diffusion chatbot learned local word structure quickly and produced early English-like behavior. But making it genuinely useful required better data, bigger models, GPU acceleration, and more training than the initial resources allowed.

The best interpretation is not "diffusion replaces LLMs tomorrow." It is:

> Diffusion is a promising language-modeling objective that may become valuable for editing, infilling, structured generation, and possibly future large-scale chat models. The idea is worth exploring, but it needs scale and engineering discipline to compete with autoregressive Transformers.

## References

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). arXiv, 2017.
2. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805). arXiv, 2018.
3. Jonathan Ho, Ajay Jain, and Pieter Abbeel. ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239). arXiv, 2020.
4. Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. ["Diffusion-LM Improves Controllable Text Generation"](https://arxiv.org/abs/2205.14217). arXiv, 2022.
5. Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. ["DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models"](https://arxiv.org/abs/2210.08933). arXiv, 2022.
6. Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. ["Simple and Effective Masked Diffusion Language Models"](https://arxiv.org/abs/2406.07524). arXiv, 2024.
7. Justin Deschenaux and Caglar Gulcehre. ["Promises, Outlooks and Challenges of Diffusion Language Modeling"](https://arxiv.org/html/2406.11473v1). arXiv, 2024.
8. Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. ["Large Language Diffusion Models"](https://arxiv.org/abs/2502.09992). arXiv, 2025.
9. Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and Volodymyr Kuleshov. ["Mercury: Ultra-Fast Language Models Based on Diffusion"](https://arxiv.org/abs/2506.17298). arXiv, 2025.
10. Google DeepMind. ["Gemini Diffusion"](https://deepmind.google/models/gemini-diffusion/). Google DeepMind, 2025.
11. ConvLab. ["DailyDialog Dataset on Hugging Face"](https://huggingface.co/datasets/ConvLab/dailydialog).
12. Databricks. ["Databricks Dolly 15k Dataset on Hugging Face"](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
