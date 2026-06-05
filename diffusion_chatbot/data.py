import csv
import os
import random

import numpy as np

from .tokenizer import WordTokenizer


GREETINGS = ["hello", "hi", "hey", "good morning", "good evening"]
FEELINGS = ["happy", "sad", "tired", "confused", "stressed", "excited", "bored"]
TOPICS = ["python", "code", "writing", "ideas", "homework", "a project", "debugging"]
NAMES = ["nova", "echo", "diffuse", "tiny bot"]
THANKS = ["thanks", "thank you", "cheers"]


def generate_pairs(n=50000, seed=7):
    rng = random.Random(seed)
    builders = [
        _greeting_pair,
        _feeling_pair,
        _identity_pair,
        _help_pair,
        _thanks_pair,
        _goodbye_pair,
        _simple_question_pair,
        _coding_pair,
    ]
    pairs = []
    for _ in range(n):
        pairs.append(rng.choice(builders)(rng))
    return pairs


def _greeting_pair(rng):
    greeting = rng.choice(GREETINGS)
    reply = rng.choice([
        "hello how are you",
        "hi how are you today",
        "hello i am here",
        "hey what can i help with",
    ])
    return greeting, reply


def _feeling_pair(rng):
    feeling = rng.choice(FEELINGS)
    prompt = rng.choice([
        f"i feel {feeling}",
        f"i am {feeling}",
        f"today i feel {feeling}",
    ])
    if feeling in {"sad", "tired", "confused", "stressed", "bored"}:
        reply = rng.choice([
            f"i am sorry you feel {feeling}",
            "that sounds hard but i can listen",
            "i hope things get better soon",
        ])
    else:
        reply = rng.choice([
            f"i am glad you feel {feeling}",
            "that sounds good",
            "nice i am happy for you",
        ])
    return prompt, reply


def _identity_pair(rng):
    name = rng.choice(NAMES)
    prompt = rng.choice([
        "what are you",
        "who are you",
        "what is your name",
        "are you a bot",
    ])
    reply = rng.choice([
        f"i am {name}",
        "i am a small diffusion chatbot",
        "i am a tiny language model",
        "yes i am a bot",
    ])
    return prompt, reply


def _help_pair(rng):
    topic = rng.choice(TOPICS)
    prompt = rng.choice([
        f"can you help me with {topic}",
        f"i need help with {topic}",
        f"help me with {topic}",
    ])
    reply = rng.choice([
        f"yes i can try to help with {topic}",
        "tell me what part is hard",
        "i can give a simple answer",
        "i will try to help",
    ])
    return prompt, reply


def _thanks_pair(rng):
    prompt = rng.choice(THANKS)
    reply = rng.choice([
        "you are welcome",
        "no problem",
        "glad i could help",
        "any time",
    ])
    return prompt, reply


def _goodbye_pair(rng):
    prompt = rng.choice(["bye", "goodbye", "see you", "talk later"])
    reply = rng.choice([
        "goodbye",
        "see you later",
        "bye have a good day",
        "talk to you later",
    ])
    return prompt, reply


def _simple_question_pair(rng):
    prompt, reply = rng.choice([
        ("how are you", "i am doing okay"),
        ("are you okay", "yes i am okay"),
        ("what can you do", "i can make short replies"),
        ("do you understand me", "i understand a little"),
        ("can you talk", "yes i can talk a little"),
        ("tell me something", "small steps can build big things"),
    ])
    return prompt, reply


def _coding_pair(rng):
    prompt, reply = rng.choice([
        ("my code is broken", "show me the error message"),
        ("python error", "read the traceback first"),
        ("how do i debug", "check the smallest failing case"),
        ("write python code", "i can help write simple python"),
        ("what is a loop", "a loop repeats code"),
    ])
    return prompt, reply


def write_pairs(path, pairs):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for prompt, response in pairs:
            writer.writerow([prompt, response])


def read_pairs(path):
    pairs = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Bad row in {path}: {row}")
            pairs.append((row[0].strip(), row[1].strip()))
    if not pairs:
        raise ValueError(f"No dialogue pairs found in {path}")
    return pairs


def build_arrays(pairs, config):
    texts = []
    for prompt, response in pairs:
        texts.append(prompt)
        texts.append(response)
    tokenizer = WordTokenizer.train(texts, vocab_size=config.vocab_size)
    prompt_ids, response_ids = encode_pairs(pairs, tokenizer, config)
    return tokenizer, prompt_ids, response_ids


def encode_pairs(pairs, tokenizer, config):
    prompts = []
    responses = []
    for prompt, response in pairs:
        prompts.append(tokenizer.encode_prompt(prompt, config.max_prompt_tokens))
        responses.append(tokenizer.encode_response(response, config.max_response_tokens))
    return np.asarray(prompts, dtype=np.int64), np.asarray(responses, dtype=np.int64)
