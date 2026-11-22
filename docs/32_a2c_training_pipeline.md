# A2C CartPole 端到端训练示例

`examples/train_a2c_cartpole.py` 将 A2C Agent、RolloutBuffer 和实验工具串成可运行流程：

- 使用 Gymnasium CartPole 采样离散动作 rollout；
- 将 terminated 与 truncated 分开保存；
- 在 episode 终点根据 next observation 的 value bootstrap；
- 通过 RolloutBuffer 计算 GAE(return target)，再用 A2C 更新；
- 使用 CSVLogger 记录训练回报、评估回报、policy/value loss 和 entropy；
- 周期性用独立 eval environment 评估，并用 checkpoint 保存网络、optimizer、step 和 metadata。

运行：

```bash
python examples/train_a2c_cartpole.py --episodes 500 --seed 7
```

可通过 `--run-dir` 指定输出目录，日志写入 `metrics.csv`，最新模型写入 `latest.pt`。这里采用单环境、每个 episode 更新一次的简单 A2C baseline，目的是展示完整训练工程接口，并不等同于 vectorized A2C/A3C 或论文级 benchmark。
