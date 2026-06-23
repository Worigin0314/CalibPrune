"""Model loader registry.

Real VLM loading is intentionally explicit. The local `toy_vlm` is only for
smoke tests and must not be used for paper claims.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

import torch

from calibprune.models.vlm_wrapper import LogitsBundle, extract_answer_logits
from calibprune.pruners.base import build_pruner, kept_count
from calibprune.pruners.fastv import select_fastv_indices
from calibprune.pruners.pyramiddrop import select_pyramiddrop_attention_tokens, select_pyramiddrop_tokens
from calibprune.pruners.sparsevlm import select_sparsevlm_tokens
from calibprune.pruners.vtw import select_vtw_attention_tokens, select_vtw_proxy_tokens
from calibprune.pruners.visionzip import clip_key_metric_from_hidden_states, select_visionzip_tokens


def _midlayer_index(env_name: str, default_layer: int, n_layers: int) -> int:
    value = os.environ.get(env_name)
    if value is None or value.strip() == "":
        return min(max(0, default_layer), n_layers - 1)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer layer index, got {value!r}.") from exc
    if parsed < 0:
        parsed = n_layers + parsed
    return min(max(0, parsed), n_layers - 1)


class VLM(NamedTuple):
    name: str
    model: torch.nn.Module | None
    processor: Any
    visual_token_layer_idx: int
    num_visual_tokens: int
    forward_with_logits: Callable[[Any, str, tuple[str, ...]], LogitsBundle]


def _toy_forward(
    image: Any,
    query: str,
    answer_choices: tuple[str, ...],
    pruner_name: str = "none",
    retention: float = 1.0,
) -> LogitsBundle:
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    bias = (digest[0] / 255.0) - 0.5
    pixel_mean = sum(image.resize((1, 1)).getpixel((0, 0))) / (3 * 255.0)
    logits = torch.tensor([0.2 - pixel_mean - bias, pixel_mean + bias], dtype=torch.float32)
    if len(answer_choices) != 2:
        logits = torch.linspace(-0.5, 0.5, steps=len(answer_choices))
    pred_idx = int(torch.argmax(logits).item())
    answer_set_logits = {choice: float(logits[i].item()) for i, choice in enumerate(answer_choices)}
    return LogitsBundle(
        logits=logits,
        pred_token=pred_idx,
        pred_text=answer_choices[pred_idx],
        answer_set_logits=answer_set_logits,
        metadata={"num_visual_tokens_original": 16, "num_visual_tokens_kept": 16},
    )


def load_model(tag: str) -> VLM:
    if tag == "toy_vlm":
        return VLM(
            name="toy_vlm",
            model=None,
            processor=None,
            visual_token_layer_idx=0,
            num_visual_tokens=16,
            forward_with_logits=_toy_forward,
        )
    if tag == "llava15_7b_4bit":
        return _load_llava15_7b_4bit()
    if tag == "qwen2vl_2b":
        return _load_qwen2vl_2b()
    raise RuntimeError(
        f"Model {tag!r} is not downloaded or wired yet. "
        "Run scripts/download_models.py after dependencies are installed."
    )


def _first_token_candidates(tokenizer: Any, answer: str, mode: str = "all_variants") -> list[int]:
    """Return first-token verbalizer ids for a closed-set answer.

    ``all_variants`` preserves the original LLaVA-style behavior. Qwen-style
    chat prompts can be sensitive to yes/no prior bias when all casing and
    leading-space variants are pooled, so the Qwen loader uses a canonical mode
    by default and records that mode in run metadata.
    """
    clean = answer.strip()
    if mode == "all_variants":
        candidates = [clean, " " + clean, clean.capitalize(), " " + clean.capitalize()]
    elif mode == "lower_no_space":
        candidates = [clean.lower()]
    elif mode == "lower_space":
        candidates = [" " + clean.lower()]
    elif mode == "capital_no_space":
        candidates = [clean.capitalize()]
    elif mode == "capital_space":
        candidates = [" " + clean.capitalize()]
    elif mode == "as_written":
        candidates = [answer]
    else:
        raise ValueError(f"Unknown answer verbalizer mode: {mode}")
    ids: list[int] = []
    for text in candidates:
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def _qwen_verbalizer_mode(answer_choices: tuple[str, ...]) -> str:
    override = os.environ.get("CALIBPRUNE_QWEN_VERBALIZER", "").strip()
    if override:
        return override
    normalized = {choice.strip().lower() for choice in answer_choices}
    if normalized == {"yes", "no"}:
        return "capital_no_space"
    return "as_written"


def _to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _format_prompt(query: str, answer_choices: tuple[str, ...]) -> str:
    normalized = tuple(choice.strip().lower() for choice in answer_choices)
    if os.environ.get("CALIBPRUNE_OPEN_ENDED_RERANK") == "1":
        instruction = "Answer the question with a short phrase."
    elif set(normalized) == {"yes", "no"}:
        instruction = "Answer with yes or no."
    elif all(len(choice.strip()) == 1 and choice.strip().isalpha() for choice in answer_choices):
        instruction = "Answer with the option letter only."
    else:
        options = ", ".join(answer_choices)
        instruction = f"Answer with one of: {options}."
    return f"USER: <image>\n{query}\n{instruction}\nASSISTANT:"


def _answer_instruction(answer_choices: tuple[str, ...]) -> str:
    normalized = tuple(choice.strip().lower() for choice in answer_choices)
    if os.environ.get("CALIBPRUNE_OPEN_ENDED_RERANK") == "1":
        return "Answer the question with a short phrase."
    if set(normalized) == {"yes", "no"}:
        return "Answer with yes or no."
    if all(len(choice.strip()) == 1 and choice.strip().isalpha() for choice in answer_choices):
        return "Answer with the option letter only."
    options = ", ".join(answer_choices)
    return f"Answer with one of: {options}."


def _format_qwen_user_text(query: str, answer_choices: tuple[str, ...]) -> str:
    return f"{query}\n{_answer_instruction(answer_choices)}"


def _qwen_image_token_span(input_ids: torch.Tensor, *, image_token_id: int) -> torch.Tensor:
    """Return the contiguous Qwen2-VL image-token positions for one sample."""

    if input_ids.shape[0] != 1:
        raise ValueError("The local Qwen2-VL hook currently expects batch size 1.")
    positions = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("Qwen2-VL prompt contains no image tokens.")
    expected = torch.arange(
        int(positions[0].item()),
        int(positions[0].item()) + int(positions.numel()),
        device=positions.device,
        dtype=positions.dtype,
    )
    if not torch.equal(positions, expected):
        raise RuntimeError("Qwen2-VL image tokens are not contiguous; pruning hook cannot safely slice them.")
    return positions


def _qwen_keep_positions_after_pruning(
    input_ids: torch.Tensor,
    image_positions: torch.Tensor,
    kept_visual_indices: torch.Tensor,
) -> torch.Tensor:
    """Build sorted sequence indices after keeping a subset of Qwen2-VL image tokens."""

    device = input_ids.device
    seq_len = int(input_ids.shape[1])
    image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    image_mask[image_positions] = True
    text_positions = torch.nonzero(~image_mask, as_tuple=False).flatten()
    kept_image_positions = image_positions[kept_visual_indices.to(device=device, dtype=torch.long)]
    return torch.cat([text_positions, kept_image_positions]).sort().values

def _collapse_repeated_image_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    image_token_index: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Collapse processor-expanded image placeholders for custom LLaVA merging."""

    if input_ids.shape[0] != 1:
        raise ValueError("The local LLaVA hook currently expects batch size 1.")
    ids = input_ids[0]
    keep: list[int] = []
    previous_was_image = False
    for idx, token_id in enumerate(ids.tolist()):
        is_image = int(token_id) == int(image_token_index)
        if is_image and previous_was_image:
            continue
        keep.append(idx)
        previous_was_image = is_image
    if len(keep) == int(ids.shape[0]):
        return input_ids, attention_mask
    keep_tensor = torch.as_tensor(keep, dtype=torch.long, device=input_ids.device)
    collapsed_ids = input_ids.index_select(1, keep_tensor)
    collapsed_mask = attention_mask.index_select(1, keep_tensor) if attention_mask is not None else None
    return collapsed_ids, collapsed_mask


def _load_llava15_7b_4bit() -> VLM:
    project_root = Path.cwd()
    os.environ.setdefault("HF_HOME", str(project_root / ".hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_root / ".hf-cache" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(project_root / ".torch-cache"))

    try:
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Install transformers and bitsandbytes before loading LLaVA.") from exc

    local_dir = project_root / "models" / "llava15_7b_4bit"
    source = str(local_dir) if (local_dir / "config.json").exists() else "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(source, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        source,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = processor.tokenizer
    answer_token_ids_cache: dict[tuple[str, ...], dict[str, list[int]]] = {}

    def forward_with_logits(
        image: Any,
        query: str,
        answer_choices: tuple[str, ...],
        pruner_name: str = "none",
        retention: float = 1.0,
    ) -> LogitsBundle:
        prompt = _format_prompt(query, answer_choices)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = _to_device(inputs, device=device, dtype=torch.float16)

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")
        input_ids, attention_mask = _collapse_repeated_image_tokens(
            input_ids,
            attention_mask,
            image_token_index=model.config.image_token_index,
        )

        def image_features_from_pixels(active_pruner: str, active_retention: float) -> tuple[torch.Tensor, dict[str, Any]]:
            image_outputs = model.vision_tower(
                pixel_values,
                output_hidden_states=True,
                output_attentions=active_pruner == "visionzip",
            )
            vision_layer = model.config.vision_feature_layer
            select_strategy = model.config.vision_feature_select_strategy
            selected = image_outputs.hidden_states[vision_layer]
            metadata: dict[str, Any] = {
                "num_visual_tokens_original": int(selected.shape[1] - (1 if select_strategy == "default" else 0)),
            }
            if active_pruner == "visionzip":
                if not image_outputs.attentions:
                    raise RuntimeError("VisionZip requires CLIP vision attentions, but the vision tower returned none.")
                metric = clip_key_metric_from_hidden_states(
                    model.vision_tower,
                    image_outputs.hidden_states,
                    select_layer=vision_layer,
                )
                packed = select_visionzip_tokens(
                    selected,
                    image_outputs.attentions[vision_layer],
                    metric,
                    active_retention,
                )
                selected = packed.tokens
                packed_indices = torch.cat([packed.dominant_indices, packed.contextual_target_indices], dim=1)[0]
                patch_indices = packed_indices[packed_indices > 0] - 1
                relative_indices = patch_indices.detach().cpu().numpy().astype(int).tolist()
                metadata.update(
                    {
                        "kept_indices": relative_indices,
                        "visionzip_contextual_count": packed.contextual_count,
                        "visionzip_dominant_count": int(packed.dominant_indices.shape[1]),
                        "visionzip_includes_cls": True,
                    }
                )
                return model.multi_modal_projector(selected), metadata
            if select_strategy == "default":
                selected = selected[:, 1:]
            elif select_strategy != "full":
                raise ValueError(f"Unexpected vision feature select strategy: {select_strategy}")
            return model.multi_modal_projector(selected), metadata

        def merged_language_forward(
            image_features: torch.Tensor,
            output_attentions: bool = False,
        ) -> tuple[Any, torch.Tensor, torch.Tensor]:
            local_inputs_embeds = model.get_input_embeddings()(input_ids)
            local_inputs_embeds = local_inputs_embeds.to(image_features.dtype)
            merged_embeds, merged_mask, _, merged_position_ids = model._merge_input_ids_with_image_features(
                image_features,
                local_inputs_embeds,
                input_ids,
                attention_mask,
                labels=None,
            )
            outputs = model.language_model(
                attention_mask=merged_mask,
                position_ids=merged_position_ids,
                inputs_embeds=merged_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs, merged_embeds, merged_mask

        def merged_midlayer_pruned_language_forward(
            image_features: torch.Tensor,
            active_pruner: str,
            active_retention: float,
            original_tokens: int,
        ) -> tuple[Any, dict[str, Any], int, list[int] | None]:
            image_token_positions = torch.nonzero(
                input_ids[0] == model.config.image_token_index,
                as_tuple=False,
            )
            if image_token_positions.numel() != 1:
                raise RuntimeError(f"{active_pruner} mid-layer hook expects exactly one image token in the prompt.")
            image_start = int(image_token_positions[0].item())
            local_inputs_embeds = model.get_input_embeddings()(input_ids)
            local_inputs_embeds = local_inputs_embeds.to(image_features.dtype)
            hidden_states, active_mask, _, active_position_ids = model._merge_input_ids_with_image_features(
                image_features,
                local_inputs_embeds,
                input_ids,
                attention_mask,
                labels=None,
            )
            language_core = model.language_model.model
            lm_head = model.language_model.lm_head
            n_layers = int(language_core.config.num_hidden_layers)
            current_visual_count = int(original_tokens)
            current_global_indices = torch.arange(current_visual_count, device=hidden_states.device)
            metadata: dict[str, Any] = {
                "pruner_evidence_type": "literature-midlayer-hook",
                "midlayer_sequence_surgery": True,
                "midlayer_original_sequence_length": int(hidden_states.shape[1]),
            }
            stage_records: list[dict[str, Any]] = []
            default_sparse_vtw_layer = max(2, n_layers // 4)
            sparse_layer = _midlayer_index("CALIBPRUNE_SPARSEVLM_LAYER", default_sparse_vtw_layer, n_layers)
            vtw_layer = _midlayer_index("CALIBPRUNE_VTW_LAYER", default_sparse_vtw_layer, n_layers)
            pyramid_layers = tuple(sorted({min(idx, n_layers - 1) for idx in (8, 16, 24)}))
            pyramid_final_keep = kept_count(current_visual_count, active_retention)
            pyramid_targets = {
                layer: max(
                    pyramid_final_keep,
                    min(
                        original_tokens,
                        int(round(original_tokens - (original_tokens - pyramid_final_keep) * (stage + 1) / len(pyramid_layers))),
                    ),
                )
                for stage, layer in enumerate(pyramid_layers)
            }

            def layer_device(decoder_layer: torch.nn.Module) -> torch.device:
                return next(decoder_layer.parameters()).device

            def replace_visual_span(
                new_visual_states: torch.Tensor,
                new_visual_position_ids: torch.Tensor,
                new_global_indices: torch.Tensor,
            ) -> None:
                nonlocal hidden_states, active_mask, active_position_ids, current_visual_count, current_global_indices
                old_count = int(current_visual_count)
                new_count = int(new_visual_states.shape[1])
                before = slice(0, image_start)
                after = slice(image_start + old_count, hidden_states.shape[1])
                hidden_states = torch.cat(
                    [hidden_states[:, before, :], new_visual_states, hidden_states[:, after, :]],
                    dim=1,
                )
                if active_mask is not None:
                    visual_mask = active_mask.new_ones((active_mask.shape[0], new_count))
                    active_mask = torch.cat(
                        [active_mask[:, before], visual_mask, active_mask[:, after]],
                        dim=1,
                    )
                active_position_ids = torch.cat(
                    [active_position_ids[:, before], new_visual_position_ids, active_position_ids[:, after]],
                    dim=1,
                )
                current_visual_count = new_count
                current_global_indices = new_global_indices.to(device=hidden_states.device, dtype=torch.long)

            for layer_idx, decoder_layer in enumerate(language_core.layers[:n_layers]):
                device_for_layer = layer_device(decoder_layer)
                hidden_states = hidden_states.to(device_for_layer)
                active_position_ids = active_position_ids.to(device_for_layer)
                current_global_indices = current_global_indices.to(device_for_layer)
                if active_mask is not None:
                    active_mask = active_mask.to(device_for_layer)
                cache_position = torch.arange(hidden_states.shape[1], device=device_for_layer)
                causal_mask = language_core._update_causal_mask(
                    active_mask,
                    hidden_states,
                    cache_position,
                    None,
                    True,
                )
                position_embeddings = language_core.rotary_emb(hidden_states, active_position_ids)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=active_position_ids,
                    past_key_value=None,
                    output_attentions=True,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
                layer_attention = layer_outputs[1]

                should_prune = False
                if active_pruner == "sparsevlm" and layer_idx == sparse_layer:
                    should_prune = True
                elif active_pruner == "vtw" and layer_idx == vtw_layer:
                    should_prune = True
                elif active_pruner == "pyramiddrop" and layer_idx in pyramid_targets:
                    should_prune = pyramid_targets[layer_idx] < current_visual_count
                if not should_prune:
                    continue

                visual_states = hidden_states[:, image_start : image_start + current_visual_count, :]
                visual_position_ids = active_position_ids[:, image_start : image_start + current_visual_count]
                if active_pruner == "sparsevlm":
                    sparse = select_sparsevlm_tokens(
                        visual_states,
                        layer_attention[0],
                        merged_seq_len=int(hidden_states.shape[1]),
                        image_start=image_start,
                        n_visual_tokens=current_visual_count,
                        retention_ratio=active_retention,
                    )
                    kept_local = sparse.kept_indices.to(device_for_layer, dtype=torch.long)
                    kept_positions = visual_position_ids.index_select(1, kept_local)
                    if int(sparse.tokens.shape[1]) > int(kept_local.numel()):
                        kept_positions = torch.cat([kept_positions, visual_position_ids[:, -1:]], dim=1)
                    replace_visual_span(sparse.tokens, kept_positions, current_global_indices.index_select(0, kept_local))
                    metadata.update(
                        {
                            "sparsevlm_recycled_count": sparse.recycled_count,
                            "sparsevlm_recycled_max_weight": sparse.recycled_weight,
                            "sparsevlm_hook_type": "midlayer_text_guided_recycling",
                            "sparsevlm_prune_layer": int(layer_idx),
                        }
                    )
                    stage_records.append({"layer": int(layer_idx), "tokens": int(current_visual_count)})
                elif active_pruner == "pyramiddrop":
                    target_keep = int(pyramid_targets[layer_idx])
                    stage_retention = target_keep / max(1, current_visual_count)
                    pyramid = select_pyramiddrop_attention_tokens(
                        visual_states,
                        (layer_attention,),
                        merged_seq_len=int(hidden_states.shape[1]),
                        image_start=image_start,
                        n_visual_tokens=current_visual_count,
                        retention_ratio=stage_retention,
                        layer_indices=(0,),
                    )
                    kept_local = pyramid.kept_indices.to(device_for_layer, dtype=torch.long)
                    new_positions = visual_position_ids.index_select(1, kept_local)
                    replace_visual_span(pyramid.tokens, new_positions, current_global_indices.index_select(0, kept_local))
                    stage_records.append(
                        {
                            "layer": int(layer_idx),
                            "tokens": int(current_visual_count),
                            "target": target_keep,
                            "mean_redundancy": pyramid.mean_redundancy,
                        }
                    )
                    metadata.update(
                        {
                            "pyramiddrop_hook_type": "midlayer_progressive_decoder_attention_diversity",
                            "pyramiddrop_midlayer_stagewise": True,
                            "pyramiddrop_stage_layers": [int(item) for item in pyramid_layers],
                        }
                    )
                else:
                    vtw = select_vtw_attention_tokens(
                        visual_states,
                        layer_attention[0],
                        merged_seq_len=int(hidden_states.shape[1]),
                        image_start=image_start,
                        n_visual_tokens=current_visual_count,
                        retention_ratio=active_retention,
                    )
                    kept_local = vtw.kept_indices.to(device_for_layer, dtype=torch.long)
                    new_positions = visual_position_ids.index_select(1, kept_local)
                    replace_visual_span(vtw.tokens, new_positions, current_global_indices.index_select(0, kept_local))
                    metadata.update(
                        {
                            "vtw_boundary_count": vtw.boundary_count,
                            "vtw_attention_count": vtw.attention_count,
                            "vtw_hook_type": "midlayer_decoder_attention_withdrawal",
                            "vtw_withdrawal_layer": int(layer_idx),
                            "vtw_midlayer_withdrawal": True,
                        }
                    )
                    stage_records.append({"layer": int(layer_idx), "tokens": int(current_visual_count)})

            norm_device = next(language_core.norm.parameters()).device
            hidden_states = language_core.norm(hidden_states.to(norm_device))
            logits = lm_head(hidden_states.to(lm_head.weight.device))
            metadata.update(
                {
                    "midlayer_final_sequence_length": int(hidden_states.shape[1]),
                    "midlayer_stage_records": stage_records,
                }
            )
            kept_indices = current_global_indices.detach().cpu().numpy().astype(int).tolist()
            return SimpleNamespace(logits=logits), metadata, int(current_visual_count), kept_indices

        with torch.inference_mode():
            pruner_metadata: dict[str, Any] = {}
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_features, pruner_metadata = image_features_from_pixels(pruner_name, retention)
                original_tokens = int(pruner_metadata.pop("num_visual_tokens_original"))
                kept_indices: list[int] | None = pruner_metadata.pop("kept_indices", None)
                if pruner_name == "fastv":
                    full_outputs, merged_embeds, _ = merged_language_forward(image_features, output_attentions=True)
                    if not full_outputs.attentions:
                        raise RuntimeError("FastV requires attentions, but the language model returned none.")
                    layer_idx = min(2, len(full_outputs.attentions) - 1)
                    attn = full_outputs.attentions[layer_idx][0]  # [heads, query, key]
                    image_token_positions = torch.nonzero(
                        input_ids[0] == model.config.image_token_index,
                        as_tuple=False,
                    )
                    if image_token_positions.numel() != 1:
                        raise RuntimeError("FastV hook expects exactly one image token in the prompt.")
                    image_start = int(image_token_positions[0].item())
                    kept = select_fastv_indices(
                        attn,
                        merged_seq_len=int(merged_embeds.shape[1]),
                        image_start=image_start,
                        n_visual_tokens=original_tokens,
                        retention_ratio=retention,
                    )
                    image_features = image_features[:, kept, :]
                    kept_indices = kept.detach().cpu().numpy().astype(int).tolist()
                    kept_tokens = int(image_features.shape[1])
                    outputs, _, _ = merged_language_forward(image_features, output_attentions=False)
                elif pruner_name in {"sparsevlm", "pyramiddrop", "vtw"}:
                    outputs, midlayer_metadata, kept_tokens, kept_indices = merged_midlayer_pruned_language_forward(
                        image_features,
                        pruner_name,
                        retention,
                        original_tokens,
                    )
                    pruner_metadata.update(midlayer_metadata)
                elif pruner_name not in {"none", "visionzip"}:
                    pruner = build_pruner(pruner_name)
                    pruned = pruner(image_features[0], attentions=None, retention_ratio=retention)
                    image_features = pruned.tokens.unsqueeze(0)
                    kept_indices = pruned.kept_indices.astype(int).tolist()
                    kept_tokens = int(image_features.shape[1])
                    outputs, _, _ = merged_language_forward(image_features, output_attentions=False)
                else:
                    kept_tokens = int(image_features.shape[1])
                    outputs, _, _ = merged_language_forward(image_features, output_attentions=False)
            else:
                original_tokens = 0
                kept_tokens = 0
                kept_indices = None
                outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :].detach()
        if answer_choices not in answer_token_ids_cache:
            answer_token_ids_cache[answer_choices] = {
                choice: _first_token_candidates(tokenizer, choice) for choice in answer_choices
            }
        answer_scores = extract_answer_logits(next_token_logits, answer_token_ids_cache[answer_choices])
        pred_text = max(answer_scores, key=answer_scores.get)
        return LogitsBundle(
            logits=next_token_logits,
            pred_token=int(torch.argmax(next_token_logits).detach().cpu()),
            pred_text=pred_text,
            answer_set_logits=answer_scores,
            metadata={
                "num_visual_tokens_original": original_tokens,
                "num_visual_tokens_kept": kept_tokens,
                "kept_indices": kept_indices,
                **pruner_metadata,
            },
        )

    return VLM(
        name="llava15_7b_4bit",
        model=model,
        processor=processor,
        visual_token_layer_idx=0,
        num_visual_tokens=576,
        forward_with_logits=forward_with_logits,
    )


def _load_qwen2vl_2b() -> VLM:
    project_root = Path.cwd()
    os.environ.setdefault("HF_HOME", str(project_root / ".hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_root / ".hf-cache" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(project_root / ".torch-cache"))

    try:
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers>=4.46 and qwen-vl-utils before loading Qwen2-VL."
        ) from exc

    local_dir = project_root / "models" / "qwen2vl_2b"
    source = str(local_dir) if (local_dir / "config.json").exists() else "Qwen/Qwen2-VL-2B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(source)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        source,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = processor.tokenizer
    answer_token_ids_cache: dict[tuple[str, ...], dict[str, list[int]]] = {}

    def forward_with_logits(
        image: Any,
        query: str,
        answer_choices: tuple[str, ...],
        pruner_name: str = "none",
        retention: float = 1.0,
    ) -> LogitsBundle:
        if pruner_name not in {"none", "fastv"}:
            raise RuntimeError(
                "Qwen2-VL currently supports only the unpruned path and a real FastV-style "
                "language-attention visual-token pruning hook."
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _format_qwen_user_text(query, answer_choices)},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = _to_device(dict(inputs), device=device, dtype=torch.float16)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        kept_indices: list[int] | None = None
        pruner_metadata: dict[str, Any] = {}
        with torch.inference_mode():
            if pruner_name == "fastv":
                if pixel_values is None or image_grid_thw is None:
                    raise RuntimeError("Qwen2-VL FastV pruning requires image pixel values and image_grid_thw.")
                image_positions = _qwen_image_token_span(input_ids, image_token_id=model.config.image_token_id)
                original_tokens = int(image_positions.numel())

                inputs_embeds = model.get_input_embeddings()(input_ids)
                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                if int(image_embeds.shape[0]) != original_tokens:
                    raise RuntimeError(
                        "Qwen2-VL image features and image tokens do not match: "
                        f"tokens={original_tokens}, features={int(image_embeds.shape[0])}."
                    )
                image_mask = (
                    (input_ids == model.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                position_ids, _ = model.get_rope_index(input_ids, image_grid_thw, None, attention_mask)

                full_outputs = model.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
                if not full_outputs.attentions:
                    raise RuntimeError("Qwen2-VL FastV requires decoder attentions, but none were returned.")
                layer_idx = min(2, len(full_outputs.attentions) - 1)
                attn = full_outputs.attentions[layer_idx][0]
                kept = select_fastv_indices(
                    attn,
                    merged_seq_len=int(inputs_embeds.shape[1]),
                    image_start=int(image_positions[0].item()),
                    n_visual_tokens=original_tokens,
                    retention_ratio=retention,
                )
                keep_positions = _qwen_keep_positions_after_pruning(input_ids, image_positions, kept)
                pruned_embeds = inputs_embeds.index_select(1, keep_positions)
                pruned_mask = attention_mask.index_select(1, keep_positions) if attention_mask is not None else None
                pruned_position_ids = position_ids.index_select(2, keep_positions)
                outputs = model.model(
                    input_ids=None,
                    position_ids=pruned_position_ids,
                    attention_mask=pruned_mask,
                    inputs_embeds=pruned_embeds,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = model.lm_head(outputs[0])
                kept_indices = kept.detach().cpu().numpy().astype(int).tolist()
                kept_tokens = int(len(kept_indices))
                pruner_metadata.update(
                    {
                        "qwen2vl_pruning_supported": True,
                        "qwen2vl_hook_type": "fastv_language_attention",
                        "qwen2vl_pruning_layer": int(layer_idx),
                        "qwen2vl_sequence_length_original": int(input_ids.shape[1]),
                        "qwen2vl_sequence_length_pruned": int(pruned_embeds.shape[1]),
                    }
                )
            else:
                outputs = model(**inputs, use_cache=False)
                logits = outputs.logits
                try:
                    image_positions = _qwen_image_token_span(input_ids, image_token_id=model.config.image_token_id)
                    original_tokens = int(image_positions.numel())
                except ValueError:
                    original_tokens = 0
                kept_tokens = original_tokens
                pruner_metadata["qwen2vl_pruning_supported"] = True

        next_token_logits = logits[0, -1, :].detach()
        verbalizer_mode = _qwen_verbalizer_mode(answer_choices)
        cache_key = (*answer_choices, f"verbalizer={verbalizer_mode}")
        if cache_key not in answer_token_ids_cache:
            answer_token_ids_cache[cache_key] = {
                choice: _first_token_candidates(tokenizer, choice, mode=verbalizer_mode) for choice in answer_choices
            }
        answer_scores = extract_answer_logits(next_token_logits, answer_token_ids_cache[cache_key])
        pred_text = max(answer_scores, key=answer_scores.get)
        return LogitsBundle(
            logits=next_token_logits,
            pred_token=int(torch.argmax(next_token_logits).detach().cpu()),
            pred_text=pred_text,
            answer_set_logits=answer_scores,
            metadata={
                "num_visual_tokens_original": original_tokens,
                "num_visual_tokens_kept": kept_tokens,
                "kept_indices": kept_indices,
                "answer_verbalizer_mode": verbalizer_mode,
                **pruner_metadata,
            },
        )

    return VLM(
        name="qwen2vl_2b",
        model=model,
        processor=processor,
        visual_token_layer_idx=0,
        num_visual_tokens=-1,
        forward_with_logits=forward_with_logits,
    )




