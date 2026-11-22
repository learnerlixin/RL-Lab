"""Run independent A2C CartPole experiments and aggregate seed metrics."""
import argparse
from pathlib import Path

from examples.train_a2c_cartpole import main as train_a2c
from utils import run_seed_sweep


def main(seeds, episodes, run_dir, eval_interval):
    root = Path(run_dir)

    def run_one_seed(seed):
        metrics = train_a2c(
            episodes=episodes,
            seed=seed,
            run_dir=str(root / f"seed_{seed}"),
            eval_interval=eval_interval,
        )
        return {
            "mean_eval_return": metrics["mean_eval_return"],
            "mean_train_return": metrics["mean_train_return"],
            "total_steps": metrics["total_steps"],
        }

    results = run_seed_sweep(seeds, run_one_seed)
    print("Per-seed results:")
    for seed, metrics in results["runs"].items():
        print(f"  seed={seed:4d} eval={metrics['mean_eval_return']:7.2f} "
              f"train={metrics['mean_train_return']:7.2f}")
    print("Aggregate (mean ± sample std):")
    for name, stats in results["summary"].items():
        print(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"(n={stats['n_seeds']}, 95% CI "
              f"[{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}])")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--run-dir", type=str, default="runs/a2c_seed_sweep")
    parser.add_argument("--eval-interval", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0 or args.eval_interval <= 0:
        parser.error("--episodes and --eval-interval must be positive")
    main(args.seeds, args.episodes, args.run_dir, args.eval_interval)
