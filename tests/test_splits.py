import json

from PIL import Image
import pytest

import calibprune.data.loaders as loaders
from calibprune.data.loaders import (
    _aokvqa_sample_from_row,
    _mmbench_cn_sample_from_row,
    _mmbench_cn_samples,
    _scienceqa_sample_from_row,
    build,
    dataset_population_size,
)
from calibprune.data.splits import (
    default_split_path,
    ensure_split_file,
    indices_hash,
    load_split_indices,
    make_split_indices,
    write_split_file,
)


def test_make_split_indices_is_deterministic():
    a = make_split_indices(n_total=20, n_cal=5, seed=123)
    b = make_split_indices(n_total=20, n_cal=5, seed=123)
    assert a["cal"] == b["cal"]
    assert a["test"] == b["test"]
    assert len(set(a["cal"]).intersection(a["test"])) == 0


def test_split_file_round_trip(tmp_path):
    payload = make_split_indices(n_total=12, n_cal=4, seed=20260616)
    payload["dataset"] = "pope"
    path = write_split_file(tmp_path / "pope_cal4.json", payload)
    indices, loaded = load_split_indices(path, "cal")
    assert indices == payload["cal"]
    assert loaded["dataset"] == "pope"
    assert indices_hash(indices) == indices_hash(payload["cal"])


def test_default_split_path_includes_nondefault_seed():
    assert str(default_split_path("pope", 500, 20260616)) == "data\\splits\\pope_cal500.json"
    assert str(default_split_path("pope", 500, 20260617)) == "data\\splits\\pope_cal500_seed20260617.json"


def test_ensure_split_file_rejects_seed_mismatch(tmp_path):
    path = tmp_path / "pope_cal4.json"
    ensure_split_file(path, dataset="pope", n_total=12, n_cal=4, seed=1)
    with pytest.raises(ValueError, match="seed mismatch"):
        ensure_split_file(path, dataset="pope", n_total=12, n_cal=4, seed=2)


def test_fixture_loader_respects_indices():
    samples = build("pope", "test", n=3, offline_fixture=True, indices=[7, 2, 9])
    assert [sample.sample_index for sample in samples] == [7, 2, 9]
    assert [sample.gold_answer for sample in samples] == ["yes", "no", "yes"]


def test_scienceqa_row_conversion_uses_letter_choices():
    sample = _scienceqa_sample_from_row(
        {
            "image": Image.new("RGB", (16, 16), color=(20, 30, 40)),
            "hint": "The object is in the sky.",
            "question": "Which option is brightest?",
            "choices": ["stone", "sun", "leaf"],
            "answer": 1,
        },
        "test",
        source_index=42,
        sample_index=7,
    )
    assert sample is not None
    assert sample.answer_choices == ("A", "B", "C")
    assert sample.gold_answer == "B"
    assert "A. stone" in sample.question
    assert sample.sample_index == 7


def test_aokvqa_row_conversion_uses_letter_choices():
    sample = _aokvqa_sample_from_row(
        {
            "image": Image.new("RGB", (16, 16), color=(20, 30, 40)),
            "question": "What is the person holding?",
            "choices": ["cup", "phone", "book", "bag"],
            "correct_choice_idx": 2,
        },
        "validation",
        source_index=7,
        sample_index=7,
    )

    assert sample.dataset == "aokvqa"
    assert sample.answer_choices == ("A", "B", "C", "D")
    assert sample.gold_answer == "C"
    assert "C. book" in sample.question


def test_mmbench_cn_row_conversion_uses_letter_choices():
    sample = _mmbench_cn_sample_from_row(
        {
            "image": Image.new("RGB", (16, 16), color=(20, 30, 40)),
            "hint": "请根据图像选择答案。",
            "question": "图中主体是什么？",
            "A": "猫",
            "B": "狗",
            "C": "车",
            "D": "树",
            "answer": "B",
        },
        "dev",
        source_index=11,
        sample_index=11,
    )

    assert sample.answer_choices == ("A", "B", "C", "D")
    assert sample.gold_answer == "B"
    assert "A. 猫" in sample.question
    assert sample.dataset == "mmbench_cn"


def test_mmbench_cn_row_conversion_normalizes_repeated_answer_letter():
    sample = _mmbench_cn_sample_from_row(
        {
            "image": Image.new("RGB", (16, 16), color=(20, 30, 40)),
            "question": "Which option matches the image?",
            "A": "cat",
            "B": "dog",
            "C": "car",
            "D": "tree",
            "answer": "CC",
        },
        "test",
        source_index=17,
        sample_index=17,
    )

    assert sample.gold_answer == "C"


def test_mmbench_cn_population_size_counts_stream(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        assert args == ("lmms-lab/MMBench", "cc")
        assert kwargs["split"] == "test"
        assert kwargs["streaming"] is True
        return iter([{"index": 0}, {"index": 1}, {"index": 2}])

    monkeypatch.setattr(loaders, "_load_dataset", fake_load_dataset)

    assert dataset_population_size("mmbench_cn", "test") == 3


def test_mmbench_cn_streaming_loader_preserves_requested_index_order(monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (16, 16), color=(20 + i, 30, 40)),
            "question": f"Question {i}?",
            "A": "cat",
            "B": "dog",
            "answer": "A",
        }
        for i in range(5)
    ]

    def fake_load_dataset(*args, **kwargs):
        assert args == ("lmms-lab/MMBench", "cc")
        assert kwargs["split"] == "test"
        assert kwargs["streaming"] is True
        return iter(rows)

    monkeypatch.setattr(loaders, "_load_dataset", fake_load_dataset)

    samples = _mmbench_cn_samples("test", n=3, indices=[3, 1, 4])

    assert [sample.sample_index for sample in samples] == [3, 1, 4]
    assert [sample.question.splitlines()[0] for sample in samples] == [
        "Question: Question 3?",
        "Question: Question 1?",
        "Question: Question 4?",
    ]
