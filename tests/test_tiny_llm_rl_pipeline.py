def test_tiny_llm_rl_pipeline_smoke_runs():
    from examples.tiny_llm_rl_pipeline import run_demo

    # A small CPU update per stage verifies the Reward Model, DPO, GRPO and LLM-PPO path.
    run_demo(steps=1, seed=3)
