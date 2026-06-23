from calibprune.pipelines.run_phenomenon import iter_commands


def test_iter_commands_uses_current_runnable_gate():
    commands = list(
        iter_commands(
            python_exe="python",
            model="llava15_7b_4bit",
            datasets={"pope"},
            seeds=(20260616,),
            retentions=(0.5,),
            output_root="results/raw",
            include_adaptive=True,
        )
    )

    assert len(commands) == 1
    assert "scripts/calibrate_lite_grid.py" in commands[0].command
    assert "--include-adaptive-calibprune" in commands[0].command
    assert any("pope_lite_llava_128_512_adaptive_log_margin" in part for part in commands[0].command)

