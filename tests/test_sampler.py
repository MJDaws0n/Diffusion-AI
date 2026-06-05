import numpy as np

from diffusion_chatbot.config import ModelConfig
from diffusion_chatbot.sampler import sample_response
from diffusion_chatbot.tokenizer import WordTokenizer


class FakeModel:
    def __init__(self, tokenizer):
        self.config = ModelConfig(vocab_size=tokenizer.vocab_size, max_prompt_tokens=8, max_response_tokens=4, diffusion_steps=4)
        self.ids = [
            tokenizer.token_to_id["hello"],
            tokenizer.token_to_id["how"],
            tokenizer.token_to_id["are"],
            tokenizer.token_to_id["you"],
        ]

    def predict_logits(self, prompt_ids, response_ids, timestep):
        logits = np.full((1, 4, self.config.vocab_size), -100.0, dtype=np.float32)
        for pos, token_id in enumerate(self.ids):
            logits[0, pos, token_id] = 100.0 + pos
        return logits


def test_sampler_reveals_tokens_and_final_has_no_masks():
    tokenizer = WordTokenizer.train(["hello how are you"], vocab_size=16)
    model = FakeModel(tokenizer)
    text, stages = sample_response(model, tokenizer, "hello", rng=np.random.default_rng(1), show_steps=True)
    assert text == "hello how are you"
    assert stages[0][1].tolist() == [tokenizer.mask_id] * 4
    assert tokenizer.mask_id not in stages[-1][1]
