from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    max_prompt_tokens: int = 24
    max_response_tokens: int = 18
    embed_dim: int = 64
    hidden_dim: int = 128
    diffusion_steps: int = 16
    seed: int = 7

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})
