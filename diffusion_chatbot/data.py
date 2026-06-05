import csv
import os
import random

import numpy as np

from .tokenizer import WordTokenizer


GREETINGS = ["hello", "hi", "hey", "good morning", "good evening", "yo", "hiya"]
FEELINGS = [
    "happy",
    "sad",
    "tired",
    "confused",
    "stressed",
    "excited",
    "bored",
    "angry",
    "lonely",
    "worried",
    "calm",
    "proud",
]
TOPICS = [
    "python",
    "code",
    "writing",
    "ideas",
    "homework",
    "a project",
    "debugging",
    "math",
    "a game",
    "a website",
    "a story",
    "school",
    "work",
    "an error",
]
NAMES = ["nova", "echo", "diffuse", "tiny bot", "mask bot", "odd ai"]
THANKS = ["thanks", "thank you", "cheers", "thanks a lot", "nice one"]
OBJECTS = ["book", "phone", "laptop", "cup", "window", "chair", "table", "screen", "keyboard"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
ANIMALS = ["cat", "dog", "bird", "fish", "horse"]
PLACES = ["home", "school", "work", "the shop", "the park", "the train station"]
ACTIONS = ["read", "write", "think", "learn", "debug", "build", "draw", "plan", "test"]
TIMES = ["today", "tomorrow", "tonight", "this morning", "this evening", "later"]
LEVELS = ["small", "big", "quick", "simple", "clear", "careful", "better", "new", "rough", "first"]
PROBLEMS = [
    "it is confusing",
    "i do not know where to start",
    "it keeps failing",
    "i feel stuck",
    "there are too many choices",
    "i have little time",
    "the result looks wrong",
    "i forgot the next step",
    "it feels messy",
    "i need a simple version",
]
STEPS = [
    "write down the goal",
    "make a tiny example",
    "check one thing at a time",
    "remove the noisy parts",
    "read the error slowly",
    "try a smaller version",
    "test the first step",
    "ask one clear question",
    "save your work",
    "compare the expected result",
]
GOALS = [
    "learn faster",
    "finish the task",
    "fix the mistake",
    "understand the idea",
    "make it cleaner",
    "make it work",
    "feel less stuck",
    "choose a direction",
    "explain it better",
    "build confidence",
]
QUESTION_WORDS = ["why", "how", "when", "where", "what"]
MODES = ["slowly", "quickly", "carefully", "simply", "clearly", "again", "from scratch"]


def generate_pairs(n=50000, seed=7, unique=True, stable_prompts=False):
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
        _preference_pair,
        _planning_pair,
        _explain_pair,
        _small_fact_pair,
        _rewrite_pair,
        _math_pair,
        _weather_pair,
        _encouragement_pair,
        _clarify_pair,
        _memory_pair,
        _creative_pair,
        _instruction_pair,
        _advice_combo_pair,
        _why_how_pair,
        _compare_pair,
        _daily_task_pair,
        _mistake_pair,
        _step_by_step_pair,
        _short_answer_pair,
        _motivation_combo_pair,
        _learning_pair,
        _choice_pair,
    ]
    pairs = []
    seen = set()
    seen_prompts = set()
    attempts = 0
    max_attempts = max(n * 20, 1000)
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        pair = rng.choice(builders)(rng)
        if stable_prompts and pair[0] in seen_prompts:
            continue
        if unique and pair in seen:
            continue
        seen.add(pair)
        seen_prompts.add(pair[0])
        pairs.append(pair)
    return pairs


def _greeting_pair(rng):
    greeting = rng.choice(GREETINGS)
    prompt = rng.choice([
        greeting,
        f"{greeting} there",
        f"{greeting}, how are you",
        f"{greeting} bot",
        f"{greeting} can we talk",
    ])
    reply = rng.choice([
        "hello how are you",
        "hi how are you today",
        "hello i am here",
        "hey what can i help with",
        "hello it is nice to talk",
        "hi tell me what you need",
    ])
    return prompt, reply


def _feeling_pair(rng):
    feeling = rng.choice(FEELINGS)
    prompt = rng.choice([
        f"i feel {feeling}",
        f"i am {feeling}",
        f"today i feel {feeling}",
        f"i have been feeling {feeling}",
        f"why do i feel {feeling}",
        f"i feel very {feeling} today",
    ])
    if feeling in {"sad", "tired", "confused", "stressed", "bored", "angry", "lonely", "worried"}:
        reply = rng.choice([
            f"i am sorry you feel {feeling}",
            "that sounds hard but i can listen",
            "i hope things get better soon",
            "take a slow breath and tell me more",
            "you do not have to solve it all at once",
            "small steps can help",
        ])
    else:
        reply = rng.choice([
            f"i am glad you feel {feeling}",
            "that sounds good",
            "nice i am happy for you",
            "that is good to hear",
            "hold onto that feeling",
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
        "show me the details and i will help",
        "we can break it into small steps",
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
        ("what should i do", "tell me the problem first"),
        ("can you answer questions", "yes i can answer simple questions"),
        ("are you smart", "i am small but i can try"),
        ("do you know everything", "no i only know patterns from training"),
        ("why are you here", "i am here to chat and help"),
    ])
    return prompt, reply


def _coding_pair(rng):
    prompt, reply = rng.choice([
        ("my code is broken", "show me the error message"),
        ("python error", "read the traceback first"),
        ("how do i debug", "check the smallest failing case"),
        ("write python code", "i can help write simple python"),
        ("what is a loop", "a loop repeats code"),
        ("what is a function", "a function is reusable code"),
        ("what is a variable", "a variable stores a value"),
        ("what is a list", "a list stores many values"),
        ("what is an if statement", "an if statement chooses what code runs"),
        ("my program is slow", "measure the slow part first"),
        ("how do i fix a bug", "find the smallest example that fails"),
        ("what is numpy", "numpy helps python work with arrays"),
    ])
    return prompt, reply


def _preference_pair(rng):
    thing = rng.choice(TOPICS + OBJECTS + COLORS)
    prompt = rng.choice([
        f"do you like {thing}",
        f"what do you think about {thing}",
        f"is {thing} good",
        f"tell me about {thing}",
    ])
    reply = rng.choice([
        f"i think {thing} can be interesting",
        f"{thing} sounds useful",
        f"i do not have feelings but {thing} is okay",
        f"it depends what you want from {thing}",
    ])
    return prompt, reply


def _planning_pair(rng):
    action = rng.choice(ACTIONS)
    topic = rng.choice(TOPICS)
    when = rng.choice(TIMES)
    prompt = rng.choice([
        f"help me {action} {topic}",
        f"i need to {action} {topic} {when}",
        f"how do i start {topic}",
        f"make a plan for {topic}",
    ])
    reply = rng.choice([
        "start with the smallest useful step",
        f"first decide what you want from {topic}",
        f"write down one goal for {when}",
        "make a short list and do the first item",
        "break it into three small steps",
    ])
    return prompt, reply


def _explain_pair(rng):
    concept, explanation = rng.choice([
        ("diffusion", "diffusion slowly turns noise into an answer"),
        ("masking", "masking hides tokens so the model learns to fill them"),
        ("training", "training adjusts numbers so predictions improve"),
        ("a chatbot", "a chatbot replies to messages"),
        ("a model", "a model learns patterns from data"),
        ("a token", "a token is a small piece of text"),
        ("loss", "loss measures how wrong the prediction is"),
        ("an embedding", "an embedding is a number vector for a token"),
    ])
    prompt = rng.choice([
        f"what is {concept}",
        f"explain {concept}",
        f"can you explain {concept}",
        f"what does {concept} mean",
    ])
    return prompt, explanation


def _small_fact_pair(rng):
    subject, fact = rng.choice([
        ("the sky", "the sky often looks blue in daytime"),
        ("water", "water can be liquid ice or steam"),
        ("a keyboard", "a keyboard is used to type"),
        ("a phone", "a phone can call and send messages"),
        ("a book", "a book stores written words"),
        ("python", "python is a programming language"),
        ("the sun", "the sun gives light and heat"),
        ("rain", "rain is water falling from clouds"),
    ])
    prompt = rng.choice([
        f"tell me about {subject}",
        f"what do you know about {subject}",
        f"say a fact about {subject}",
    ])
    return prompt, fact


def _rewrite_pair(rng):
    phrase = rng.choice([
        "hello friend",
        "i need help",
        "this is hard",
        "thank you very much",
        "i am learning python",
        "please explain it",
    ])
    prompt = rng.choice([
        f"rewrite {phrase}",
        f"say this differently {phrase}",
        f"make this simple {phrase}",
    ])
    reply = rng.choice([
        phrase,
        f"you can say {phrase}",
        f"a simple version is {phrase}",
    ])
    return prompt, reply


def _math_pair(rng):
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    op = rng.choice(["plus", "minus"])
    if op == "plus":
        answer = a + b
        symbol = "+"
    else:
        if b > a:
            a, b = b, a
        answer = a - b
        symbol = "-"
    prompt = rng.choice([
        f"what is {a} {op} {b}",
        f"{a} {symbol} {b}",
        f"calculate {a} {op} {b}",
    ])
    return prompt, f"{a} {op} {b} is {answer}"


def _weather_pair(rng):
    weather = rng.choice(["sunny", "rainy", "cold", "warm", "windy", "cloudy"])
    prompt = rng.choice([
        f"it is {weather} today",
        f"the weather is {weather}",
        f"what should i do when it is {weather}",
    ])
    reply = rng.choice([
        f"{weather} weather can change your plans",
        "check outside before you go",
        "dress for the weather",
        "that sounds like normal weather",
    ])
    return prompt, reply


def _encouragement_pair(rng):
    task = rng.choice(TOPICS + ACTIONS)
    prompt = rng.choice([
        f"i cannot do {task}",
        f"{task} is too hard",
        f"i want to give up on {task}",
        f"i am bad at {task}",
    ])
    reply = rng.choice([
        "try one small step first",
        "you can improve with practice",
        "it is okay to be slow at first",
        "ask for help and keep going",
        "small progress still counts",
    ])
    return prompt, reply


def _clarify_pair(rng):
    topic = rng.choice(TOPICS)
    prompt = rng.choice([
        f"help {topic}",
        f"{topic} please",
        f"i need something with {topic}",
        "i do not know what to say",
    ])
    reply = rng.choice([
        "tell me more so i can help",
        "what part should we start with",
        "can you give one more detail",
        f"what do you want to do with {topic}",
    ])
    return prompt, reply


def _memory_pair(rng):
    thing = rng.choice(OBJECTS + ANIMALS + COLORS)
    prompt = rng.choice([
        f"remember {thing}",
        f"can you remember {thing}",
        f"my word is {thing}",
    ])
    reply = rng.choice([
        f"i can repeat {thing} in this reply",
        f"you said {thing}",
        "i do not have long term memory",
    ])
    return prompt, reply


def _creative_pair(rng):
    subject = rng.choice(ANIMALS + OBJECTS + TOPICS)
    prompt = rng.choice([
        f"write a line about {subject}",
        f"make a tiny story about {subject}",
        f"say something creative about {subject}",
    ])
    reply = rng.choice([
        f"the {subject} waited for a bright morning",
        f"a small {subject} can start a big idea",
        f"today the {subject} found a new path",
        f"the {subject} made the quiet room feel alive",
    ])
    return prompt, reply


def _instruction_pair(rng):
    action = rng.choice(ACTIONS)
    obj = rng.choice(OBJECTS + TOPICS)
    prompt = rng.choice([
        f"how do i {action} {obj}",
        f"teach me to {action} {obj}",
        f"steps to {action} {obj}",
    ])
    reply = rng.choice([
        f"first choose a small part of {obj}",
        f"practice how to {action} one simple example",
        "start simple then improve it",
        "do one step then check the result",
    ])
    return prompt, reply


def _advice_combo_pair(rng):
    level = rng.choice(LEVELS)
    topic = rng.choice(TOPICS)
    problem = rng.choice(PROBLEMS)
    step = rng.choice(STEPS)
    goal = rng.choice(GOALS)
    prompt = rng.choice([
        f"i need {level} help with {topic} because {problem}",
        f"give me {level} advice for {topic}",
        f"{topic} is hard because {problem}",
        f"i want to {goal} with {topic}",
    ])
    reply = rng.choice([
        f"start with {step}",
        f"to {goal} try to {step}",
        f"keep it {level} and {step}",
        f"for {topic} first {step}",
    ])
    return prompt, reply


def _why_how_pair(rng):
    qword = rng.choice(QUESTION_WORDS)
    action = rng.choice(ACTIONS)
    topic = rng.choice(TOPICS)
    mode = rng.choice(MODES)
    prompt = rng.choice([
        f"{qword} should i {action} {topic}",
        f"{qword} do people {action} {topic}",
        f"{qword} can i {action} {topic} {mode}",
    ])
    reply = rng.choice([
        f"you can {action} {topic} {mode} by starting small",
        f"people {action} {topic} to learn and improve",
        f"try to {action} one simple part first",
        f"the best start is to {action} {topic} simply",
    ])
    return prompt, reply


def _compare_pair(rng):
    left = rng.choice(TOPICS + OBJECTS + ACTIONS)
    right = rng.choice(TOPICS + OBJECTS + ACTIONS)
    prompt = rng.choice([
        f"compare {left} and {right}",
        f"is {left} better than {right}",
        f"what is different about {left} and {right}",
    ])
    reply = rng.choice([
        f"{left} and {right} are useful in different ways",
        f"choose {left} if it fits your goal",
        f"choose {right} if it solves the problem better",
        "the better choice depends on what you need",
    ])
    return prompt, reply


def _daily_task_pair(rng):
    place = rng.choice(PLACES)
    time = rng.choice(TIMES)
    action = rng.choice(ACTIONS)
    obj = rng.choice(OBJECTS + TOPICS)
    prompt = rng.choice([
        f"i need to {action} {obj} {time}",
        f"i am at {place} and need to {action}",
        f"remind me to {action} {obj}",
        f"what should i do at {place} {time}",
    ])
    reply = rng.choice([
        f"make time to {action} {obj}",
        f"at {place} focus on one small task",
        f"{time} do the first useful step",
        "write it down so you do not forget",
    ])
    return prompt, reply


def _mistake_pair(rng):
    topic = rng.choice(TOPICS)
    problem = rng.choice(PROBLEMS)
    step = rng.choice(STEPS)
    prompt = rng.choice([
        f"i made a mistake with {topic}",
        f"{topic} went wrong because {problem}",
        f"how do i recover from a mistake in {topic}",
    ])
    reply = rng.choice([
        f"check what changed before {topic} broke",
        f"try to {step}",
        "mistakes are useful if you inspect them",
        "undo one change and test again",
    ])
    return prompt, reply


def _step_by_step_pair(rng):
    action = rng.choice(ACTIONS)
    topic = rng.choice(TOPICS)
    step1 = rng.choice(STEPS)
    step2 = rng.choice(STEPS)
    prompt = rng.choice([
        f"give me steps to {action} {topic}",
        f"step by step for {topic}",
        f"walk me through how to {action} {topic}",
    ])
    reply = rng.choice([
        f"first {step1} then {step2}",
        f"step one {step1} step two {step2}",
        f"start when you {step1}",
        "do one step then check the result",
    ])
    return prompt, reply


def _short_answer_pair(rng):
    topic = rng.choice(TOPICS + OBJECTS + COLORS + ANIMALS)
    mode = rng.choice(MODES)
    prompt = rng.choice([
        f"answer shortly about {topic}",
        f"give a short reply about {topic}",
        f"say one sentence about {topic}",
        f"explain {topic} {mode}",
    ])
    reply = rng.choice([
        f"{topic} can be simple if you start small",
        f"{topic} is worth thinking about",
        f"i can talk about {topic} a little",
        f"keep {topic} simple first",
    ])
    return prompt, reply


def _motivation_combo_pair(rng):
    goal = rng.choice(GOALS)
    topic = rng.choice(TOPICS)
    level = rng.choice(LEVELS)
    prompt = rng.choice([
        f"motivate me to {goal}",
        f"i need motivation for {topic}",
        f"help me feel {level} about {topic}",
        f"i want to {goal} but feel stuck",
    ])
    reply = rng.choice([
        f"one {level} step can help you {goal}",
        f"you can work on {topic} one bit at a time",
        "progress can be messy and still count",
        f"focus on the next step not all of {topic}",
    ])
    return prompt, reply


def _learning_pair(rng):
    topic = rng.choice(TOPICS + ACTIONS)
    mode = rng.choice(MODES)
    prompt = rng.choice([
        f"i am learning {topic}",
        f"teach me {topic} {mode}",
        f"how can i learn {topic}",
        f"what is the best way to learn {topic}",
    ])
    reply = rng.choice([
        f"practice {topic} with tiny examples",
        f"learn {topic} {mode} and test yourself",
        "repeat the basics until they feel normal",
        "build one small thing and improve it",
    ])
    return prompt, reply


def _choice_pair(rng):
    topic = rng.choice(TOPICS)
    action = rng.choice(ACTIONS)
    goal = rng.choice(GOALS)
    prompt = rng.choice([
        f"should i {action} or wait",
        f"should i start {topic} today",
        f"help me choose how to {goal}",
        f"what option should i pick for {topic}",
    ])
    reply = rng.choice([
        "pick the option with the smallest useful next step",
        f"choose what helps you {goal}",
        f"if {topic} matters start with a tiny version",
        "write two options and compare the cost",
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
