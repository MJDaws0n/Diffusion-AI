from diffusion_chatbot.data import generate_pairs


def test_generate_many_unique_pairs():
    pairs = generate_pairs(n=5000, seed=7)
    assert len(pairs) == 5000
    assert len(set(pairs)) == 5000


def test_allow_duplicates_still_returns_requested_count():
    pairs = generate_pairs(n=1000, seed=7, unique=False)
    assert len(pairs) == 1000
