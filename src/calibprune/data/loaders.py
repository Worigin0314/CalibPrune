"""Dataset loaders and offline fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

from PIL import Image


@dataclass(frozen=True)
class VQASample:
    image: Image.Image
    question: str
    gold_answer: str
    answer_choices: tuple[str, ...]
    dataset: str
    split: str
    source: str
    sample_index: int | None = None


def _dataset_cache_dir() -> str:
    project_root = Path.cwd()
    os.environ.setdefault("HF_HOME", str(project_root / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(project_root / "data" / "hf_cache"))
    return os.environ["HF_DATASETS_CACHE"]


def _load_dataset(*args: object, **kwargs: object) -> object:
    cache_dir = _dataset_cache_dir()
    kwargs.setdefault("cache_dir", cache_dir)
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _aokvqa_local_arrow(split: str) -> object | None:
    split_key = {"val": "validation", "valid": "validation"}.get(split, split)
    cache_root = Path(_dataset_cache_dir()) / "HuggingFaceM4___a-okvqa"
    if not cache_root.exists():
        return None
    arrow_paths = sorted(cache_root.glob(f"**/a-okvqa-{split_key}*.arrow"))
    if not arrow_paths:
        return None
    from datasets import Dataset, concatenate_datasets

    datasets = [Dataset.from_file(str(path)) for path in arrow_paths]
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def _fixture_samples(dataset: str, split: str, n: int, indices: list[int] | None = None) -> list[VQASample]:
    choices = ("no", "yes")
    out: list[VQASample] = []
    selected = list(range(n)) if indices is None else indices[:n]
    for i in selected:
        answer = choices[i % 2]
        color = (40 + (i * 37) % 180, 80 + (i * 53) % 150, 120 + (i * 29) % 120)
        image = Image.new("RGB", (32, 32), color=color)
        out.append(
            VQASample(
                image=image,
                question=f"Fixture {dataset} question {i}: is the salient object present?",
                gold_answer=answer,
                answer_choices=choices,
                dataset=dataset,
                split=split,
                source="offline_fixture",
                sample_index=i,
            )
        )
    return out


def _letter_labels(n_choices: int) -> tuple[str, ...]:
    if not 1 < n_choices <= 26:
        raise ValueError(f"Expected 2..26 answer choices, got {n_choices}.")
    return tuple(chr(ord("A") + idx) for idx in range(n_choices))


def _scienceqa_gold_label(answer: object, labels: tuple[str, ...]) -> str:
    if isinstance(answer, int):
        return labels[answer]
    text = str(answer).strip()
    if text.isdigit():
        return labels[int(text)]
    upper = text.upper()
    if upper in labels:
        return upper
    raise ValueError(f"Cannot map ScienceQA answer {answer!r} to labels {labels}.")


def _scienceqa_sample_from_row(
    row: dict[str, object],
    split: str,
    *,
    source_index: int,
    sample_index: int,
) -> VQASample | None:
    image = row.get("image")
    if image is None:
        return None
    if not hasattr(image, "convert"):
        raise ValueError(f"ScienceQA row {source_index} has unsupported image type {type(image)!r}.")
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"ScienceQA row {source_index} has no choices list.")
    labels = _letter_labels(len(choices))
    question = str(row.get("question", "")).strip()
    hint = str(row.get("hint", "") or "").strip()
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    if hint:
        question_text = f"Context: {hint}\nQuestion: {question}\nChoices:\n{choice_lines}"
    else:
        question_text = f"Question: {question}\nChoices:\n{choice_lines}"
    return VQASample(
        image=image.convert("RGB"),
        question=question_text,
        gold_answer=_scienceqa_gold_label(row.get("answer"), labels),
        answer_choices=labels,
        dataset="scienceqa",
        split=split,
        source="hf:derek-thomas/ScienceQA:image-subset",
        sample_index=sample_index,
    )


def _scienceqa_samples(split: str, n: int, indices: list[int] | None = None) -> list[VQASample]:
    try:
        ds = _load_dataset("derek-thomas/ScienceQA", split=split, streaming=True)
    except ImportError as exc:
        raise RuntimeError("Install datasets before loading real ScienceQA samples.") from exc
    selected_order = {int(idx): rank for rank, idx in enumerate(indices[:n])} if indices is not None else None
    samples: list[VQASample] = []
    samples_by_rank: dict[int, VQASample] = {}
    image_position = -1
    for source_index, row in enumerate(ds):
        if row.get("image") is None:
            continue
        image_position += 1
        if selected_order is not None and image_position not in selected_order:
            continue
        sample = _scienceqa_sample_from_row(
            row,
            split,
            source_index=source_index,
            sample_index=image_position,
        )
        if sample is not None and selected_order is None:
            samples.append(sample)
        elif sample is not None and selected_order is not None:
            samples_by_rank[selected_order[image_position]] = sample
        if len(samples) >= n or len(samples_by_rank) >= n:
            break
    if selected_order is not None:
        samples = [samples_by_rank[rank] for rank in range(n) if rank in samples_by_rank]
    if len(samples) < n:
        raise RuntimeError(f"ScienceQA image subset yielded {len(samples)} samples, expected {n}.")
    return samples


def _mmbench_gold_label(answer: object, labels: tuple[str, ...]) -> str:
    text = str(answer).strip().upper()
    if text in labels:
        return text
    if len(text) > 1 and len(set(text)) == 1 and text[0] in labels:
        return text[0]
    if text.isdigit():
        return labels[int(text)]
    raise ValueError(f"Cannot map MMBench answer {answer!r} to labels {labels}.")


def _row_image(row: dict[str, object], *, dataset: str, source_index: int) -> Image.Image:
    image = row.get("image", row.get("img"))
    if image is None:
        raise ValueError(f"{dataset} row {source_index} has no image field.")
    if not hasattr(image, "convert"):
        raise ValueError(f"{dataset} row {source_index} has unsupported image type {type(image)!r}.")
    return image.convert("RGB")


def _mmbench_cn_sample_from_row(
    row: dict[str, object],
    split: str,
    *,
    source_index: int,
    sample_index: int,
) -> VQASample:
    labels = tuple(label for label in _letter_labels(8) if str(row.get(label, row.get(label.lower(), ""))).strip())
    if len(labels) < 2:
        raise ValueError(f"MMBench-CN row {source_index} has fewer than two answer choices.")
    question = str(row.get("question", "")).strip()
    hint = str(row.get("hint", "") or row.get("context", "") or "").strip()
    choice_lines = "\n".join(f"{label}. {row.get(label, row.get(label.lower()))}" for label in labels)
    if hint:
        question_text = f"Context: {hint}\nQuestion: {question}\nChoices:\n{choice_lines}"
    else:
        question_text = f"Question: {question}\nChoices:\n{choice_lines}"
    return VQASample(
        image=_row_image(row, dataset="MMBench-CN", source_index=source_index),
        question=question_text,
        gold_answer=_mmbench_gold_label(row.get("answer"), labels),
        answer_choices=labels,
        dataset="mmbench_cn",
        split=split,
        source="hf:lmms-lab/MMBench:cc",
        sample_index=sample_index,
    )


def _mmbench_cn_samples(split: str, n: int, indices: list[int] | None = None) -> list[VQASample]:
    try:
        ds = _load_dataset("lmms-lab/MMBench", "cc", split=split, streaming=True)
    except ImportError as exc:
        raise RuntimeError("Install datasets before loading real MMBench-CN samples.") from exc
    selected_order = {int(idx): rank for rank, idx in enumerate(indices[:n])} if indices is not None else None
    samples: list[VQASample] = []
    samples_by_rank: dict[int, VQASample] = {}
    for source_index, row in enumerate(ds):
        if selected_order is not None and source_index not in selected_order:
            continue
        sample = _mmbench_cn_sample_from_row(
            row,
            split,
            source_index=source_index,
            sample_index=source_index,
        )
        if selected_order is None:
            samples.append(sample)
        else:
            samples_by_rank[selected_order[source_index]] = sample
        if len(samples) >= n or len(samples_by_rank) >= n:
            break
    if selected_order is not None:
        samples = [samples_by_rank[rank] for rank in range(n) if rank in samples_by_rank]
    if len(samples) < n:
        raise RuntimeError(f"MMBench-CN yielded {len(samples)} samples, expected {n}.")
    return samples


def _aokvqa_sample_from_row(
    row: dict[str, object],
    split: str,
    *,
    source_index: int,
    sample_index: int,
) -> VQASample:
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError(f"A-OKVQA row {source_index} has no multiple-choice options.")
    labels = _letter_labels(len(choices))
    correct_idx = row.get("correct_choice_idx")
    if not isinstance(correct_idx, int):
        correct_idx = int(str(correct_idx))
    if correct_idx < 0 or correct_idx >= len(labels):
        raise ValueError(f"A-OKVQA row {source_index} has invalid correct_choice_idx={correct_idx!r}.")
    question = str(row.get("question", "")).strip()
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return VQASample(
        image=_row_image(row, dataset="A-OKVQA", source_index=source_index),
        question=f"Question: {question}\nChoices:\n{choice_lines}",
        gold_answer=labels[correct_idx],
        answer_choices=labels,
        dataset="aokvqa",
        split=split,
        source="hf:HuggingFaceM4/A-OKVQA",
        sample_index=sample_index,
    )


def _aokvqa_samples(split: str, n: int, indices: list[int] | None = None) -> list[VQASample]:
    ds = _aokvqa_local_arrow(split)
    if ds is None:
        try:
            ds = _load_dataset("HuggingFaceM4/A-OKVQA", split=split)
        except ImportError as exc:
            raise RuntimeError("Install datasets before loading real A-OKVQA samples.") from exc
    if indices is not None:
        selected = indices[:n]
        ds = ds.select(selected)
    else:
        selected = list(range(min(n, len(ds))))
        ds = ds.select(selected)
    samples = [
        _aokvqa_sample_from_row(row, split, source_index=int(source_index), sample_index=int(source_index))
        for source_index, row in zip(selected, ds)
    ]
    if len(samples) < n:
        raise RuntimeError(f"A-OKVQA yielded {len(samples)} samples, expected {n}.")
    return samples


def dataset_population_size(dataset: str, split: str = "test", offline_fixture: bool = False) -> int:
    if offline_fixture:
        return 1000
    if dataset == "pope":
        return 9000
    if dataset == "scienceqa":
        try:
            ds = _load_dataset("derek-thomas/ScienceQA", split=split, streaming=True)
        except ImportError as exc:
            raise RuntimeError("Install datasets before sizing real ScienceQA samples.") from exc
        return sum(1 for row in ds if row.get("image") is not None)
    if dataset == "mmbench_cn":
        try:
            ds = _load_dataset("lmms-lab/MMBench", "cc", split=split, streaming=True)
        except ImportError as exc:
            raise RuntimeError("Install datasets before sizing real MMBench-CN samples.") from exc
        return sum(1 for _ in ds)
    if dataset == "aokvqa":
        ds = _aokvqa_local_arrow(split)
        if ds is None:
            try:
                ds = _load_dataset("HuggingFaceM4/A-OKVQA", split=split)
            except ImportError as exc:
                raise RuntimeError("Install datasets before sizing real A-OKVQA samples.") from exc
        return int(len(ds))
    raise RuntimeError(f"Population size is not implemented for {dataset!r}.")


def build(
    dataset: str,
    split: str = "test",
    n: int | None = None,
    offline_fixture: bool = False,
    indices: list[int] | None = None,
) -> Sequence[VQASample]:
    n = 4 if n is None else n
    if offline_fixture:
        return _fixture_samples(dataset, split, n, indices)
    if dataset == "pope":
        try:
            split_expr = split if indices is not None or "[" in split else f"{split}[:{n}]"
            ds = _load_dataset("lmms-lab/POPE", split=split_expr)
        except ImportError as exc:
            raise RuntimeError("Install datasets before loading real POPE samples.") from exc
        if indices is not None:
            selected = indices[:n]
            ds = ds.select(selected)
        else:
            selected = list(range(len(ds)))
        samples: list[VQASample] = []
        for source_index, row in zip(selected, ds):
            samples.append(
                VQASample(
                    image=row["image"].convert("RGB"),
                    question=str(row["question"]),
                    gold_answer=str(row["answer"]).strip().lower(),
                    answer_choices=("no", "yes"),
                    dataset=dataset,
                    split=split,
                    source="hf:lmms-lab/POPE",
                    sample_index=int(source_index),
                )
        )
        return samples
    if dataset == "scienceqa":
        return _scienceqa_samples(split, n, indices)
    if dataset == "mmbench_cn":
        return _mmbench_cn_samples(split, n, indices)
    if dataset == "aokvqa":
        return _aokvqa_samples(split, n, indices)
    raise RuntimeError(
        f"Real dataset loading is not implemented for {dataset!r}. "
        "Use offline_fixture=True for smoke tests, or add the real loader before paper experiments."
    )
