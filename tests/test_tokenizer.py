from diffusion_chatbot.tokenizer import WordTokenizer


def test_tokenizer_round_trip_simple_text():
    tokenizer = WordTokenizer.train(["hello how are you", "i am okay"], vocab_size=32)
    ids = tokenizer.encode_response("hello how are you", max_len=8)
    assert tokenizer.decode_ids(ids) == "hello how are you"


def test_prompt_has_bos_and_sep():
    tokenizer = WordTokenizer.train(["hello"], vocab_size=16)
    ids = tokenizer.encode_prompt("hello", max_len=5)
    assert ids[0] == tokenizer.bos_id
    assert tokenizer.sep_id in ids
