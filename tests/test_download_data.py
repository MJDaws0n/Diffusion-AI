import json

from diffusion_chatbot.download_data import pairs_from_dailydialog, pairs_from_dolly_lines


def test_daily_dialog_parser_uses_user_to_system_pairs():
    dialogues = [
        {
            "data_split": "train",
            "turns": [
                {"speaker": "user", "utterance": "Hello there."},
                {"speaker": "system", "utterance": "Hi, how are you?"},
                {"speaker": "user", "utterance": "I am okay."},
            ],
        }
    ]
    pairs = pairs_from_dailydialog(dialogues)
    assert pairs == [("Hello there.", "Hi, how are you?")]


def test_dolly_parser_builds_instruction_response_pairs():
    line = json.dumps({
        "instruction": "Explain a loop",
        "context": "",
        "response": "A loop repeats code.",
    })
    pairs = pairs_from_dolly_lines([line])
    assert pairs == [("Explain a loop", "A loop repeats code.")]
