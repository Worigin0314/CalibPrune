import subprocess
import sys

import numpy as np


def write_npz(path, subset):
    np.savez_compressed(
        path,
        logits=np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.2]], dtype=float),
        labels=np.array([0, 1, 0], dtype=np.int64),
        answer_choices=np.array(["no", "yes"], dtype=object),
        sample_indices=np.array([0, 1, 2], dtype=np.int64),
        subset_name=np.asarray(subset, dtype=object),
        sample_indices_hash=np.asarray("abc", dtype=object),
    )


def test_calibrate_script_rejects_test_as_calibration(tmp_path):
    cal = tmp_path / "wrong_cal.npz"
    test = tmp_path / "test.npz"
    out = tmp_path / "out.json"
    write_npz(cal, "test")
    write_npz(test, "test")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_from_logits.py",
            "--cal",
            f"0.5={cal}",
            "--test",
            f"0.5={test}",
            "--calibrator",
            "temperature_scaling",
            "--output",
            str(out),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "subset_name=cal" in completed.stderr

