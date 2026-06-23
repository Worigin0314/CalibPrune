from scripts.guarded_adaptive_from_logits import select_guarded_candidate


def _candidate(name: str, score: float) -> dict[str, object]:
    return {"name": name, "score": score, "model": object()}


def test_select_guarded_candidate_falls_back_to_ts_for_small_validation_gap():
    candidates = [
        _candidate("temperature_scaling", 0.100),
        _candidate("adaptive_calibprune", 0.096),
    ]

    selected = select_guarded_candidate(candidates, selection_margin=0.01)

    assert selected["name"] == "temperature_scaling"


def test_select_guarded_candidate_accepts_clear_validation_gain():
    candidates = [
        _candidate("temperature_scaling", 0.100),
        _candidate("adaptive_calibprune", 0.080),
    ]

    selected = select_guarded_candidate(candidates, selection_margin=0.01)

    assert selected["name"] == "adaptive_calibprune"