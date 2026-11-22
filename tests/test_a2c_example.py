def test_a2c_example_writes_metrics_and_checkpoint(tmp_path):
    from examples.train_a2c_cartpole import main

    main(episodes=2, seed=3, run_dir=str(tmp_path), eval_interval=1)
    metrics_path = tmp_path / "metrics.csv"
    checkpoint_path = tmp_path / "latest.pt"
    assert metrics_path.exists()
    assert checkpoint_path.exists()
    rows = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3  # header + two episodes
