def test_tiny_llm_rl_pipeline_smoke_runs():
    from examples.tiny_llm_rl_pipeline import run_demo

    # A single small CPU update per objective verifies the complete toy path.
    run_demo(steps=1, seed=3)
