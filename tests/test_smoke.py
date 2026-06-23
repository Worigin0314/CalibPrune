from calibprune.data.loaders import build
import pytest

from calibprune.models.loader import (
    _collapse_repeated_image_tokens,
    _first_token_candidates,
    _format_prompt,
    _qwen_image_token_span,
    _qwen_keep_positions_after_pruning,
    _qwen_verbalizer_mode,
    load_model,
)


def test_fixture_dataset_builds_samples():
    samples = build("pope", "test", n=4, offline_fixture=True)
    assert len(samples) == 4
    assert samples[0].image.size == (32, 32)
    assert samples[0].gold_answer in samples[0].answer_choices


def test_toy_vlm_forward_has_answer_logits():
    sample = build("pope", "test", n=1, offline_fixture=True)[0]
    vlm = load_model("toy_vlm")
    bundle = vlm.forward_with_logits(sample.image, sample.question, sample.answer_choices)
    assert bundle.answer_set_logits is not None
    assert set(bundle.answer_set_logits) == set(sample.answer_choices)


def test_prompt_instruction_matches_answer_space():
    yes_no = _format_prompt("Is there a cat?", ("no", "yes"))
    mcq = _format_prompt("Question: pick one\nChoices:\nA. red\nB. blue", ("A", "B"))
    assert "Answer with yes or no." in yes_no
    assert "Answer with the option letter only." in mcq



class _FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "yes": [1],
            " yes": [2],
            "Yes": [3],
            " Yes": [4],
            "no": [5],
            "No": [6],
        }

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            pass

        out = Encoded()
        out.input_ids = self.vocab.get(text, [])
        return out


def test_qwen_yes_no_uses_canonical_verbalizer(monkeypatch):
    monkeypatch.delenv("CALIBPRUNE_QWEN_VERBALIZER", raising=False)
    assert _qwen_verbalizer_mode(("no", "yes")) == "capital_no_space"
    assert _qwen_verbalizer_mode(("A", "B")) == "as_written"
    monkeypatch.setenv("CALIBPRUNE_QWEN_VERBALIZER", "lower_space")
    assert _qwen_verbalizer_mode(("no", "yes")) == "lower_space"


def test_first_token_candidates_supports_canonical_modes():
    tokenizer = _FakeTokenizer()
    assert _first_token_candidates(tokenizer, "yes", mode="all_variants") == [1, 2, 3, 4]
    assert _first_token_candidates(tokenizer, "yes", mode="capital_no_space") == [3]
    assert _first_token_candidates(tokenizer, "no", mode="lower_no_space") == [5]
    with pytest.raises(ValueError, match="Unknown answer verbalizer mode"):
        _first_token_candidates(tokenizer, "yes", mode="bad_mode")

def test_collapse_repeated_image_tokens_keeps_one_placeholder():
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[10, 32000, 32000, 32000, 11, 12]])
    attention_mask = torch.ones_like(input_ids)

    collapsed_ids, collapsed_mask = _collapse_repeated_image_tokens(
        input_ids,
        attention_mask,
        image_token_index=32000,
    )

    assert collapsed_ids.tolist() == [[10, 32000, 11, 12]]
    assert collapsed_mask.tolist() == [[1, 1, 1, 1]]


def test_qwen_image_token_span_requires_contiguous_tokens():
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[1, 151655, 151655, 2]])

    positions = _qwen_image_token_span(input_ids, image_token_id=151655)

    assert positions.tolist() == [1, 2]

    with pytest.raises(RuntimeError, match="not contiguous"):
        _qwen_image_token_span(torch.tensor([[1, 151655, 2, 151655]]), image_token_id=151655)


def test_qwen_keep_positions_after_pruning_keeps_text_and_selected_image_tokens():
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[10, 151655, 151655, 151655, 11, 12]])
    image_positions = _qwen_image_token_span(input_ids, image_token_id=151655)

    keep = _qwen_keep_positions_after_pruning(input_ids, image_positions, torch.tensor([0, 2]))

    assert keep.tolist() == [0, 1, 3, 4, 5]



