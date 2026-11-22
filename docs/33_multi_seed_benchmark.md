# 多随机种子实验与结果汇总

强化学习训练对初始化、环境采样和 replay sampling 都较敏感。单次 seed 结果不足以代表算法表现。仓库增加轻量 seed sweep 工具，负责重置 Python/NumPy/PyTorch RNG，运行调用方训练/评估回调，并汇总标量结果。

## 使用方式

```python
from utils import run_seed_sweep

def train_and_evaluate(seed):
    # 在 callback 内创建 env，并单独调用 env.reset(seed=seed)
    # train agent, evaluate it on held-out episodes
    return {"mean_return": 180.0, "mean_length": 190.0}

result = run_seed_sweep([0, 1, 2, 3, 4], train_and_evaluate)
print(result["summary"]["mean_return"])
```

## 汇总字段

`summarize_seed_results` 为每项指标给出 mean、sample std、median、normal-approximation 95% confidence interval 和 seed 数量。只有一个 seed 时，std 报为 0，区间退化为观测均值；这不代表统计确定性。

`run_seed_sweep` 顺序执行每个 seed，不管理 GPU 资源、训练时长、输出目录或并行作业。建议 callback 在每个 seed 独立创建 env/agent、将每个 run 的 checkpoint 和原始 episode metrics 保存到不同子目录，并使用独立评估 seed。比较算法时应尽量使用相同的 seed 集合和评估协议。

该工具是结果聚合器，不是标准 benchmark 套件。报告论文级比较时，还需预先确定数据集、训练预算、评估 episode 数、超参数搜索预算，并对所有方法使用一致协议。


## 可运行示例：A2C CartPole

仓库提供 `examples/sweep_a2c_cartpole.py`，将真实 A2C 训练/评估接入 seed sweep：

```bash
python examples/sweep_a2c_cartpole.py --seeds 0 1 2 3 4 --episodes 300
```

每个 seed 使用独立 run directory，分别保留 CSV 训练日志与 checkpoint；最后打印每个 seed 的评估回报以及跨 seed 的均值、标准差和 95% 近似置信区间。不同算法比较时可沿用相同 seeds 和评估协议。
