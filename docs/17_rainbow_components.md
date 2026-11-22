# 17. Rainbow DQN 的核心构件

Rainbow 将多种 DQN 改进组合起来，经典组件包括 Double DQN、Prioritized Replay、Dueling Network、multi-step return、Noisy Networks 和 Distributional RL（C51）。

本仓库已经分别提供 Double DQN、Dueling 网络、PER 基础结构，并在本节新增：
- **NoisyLinear**：因子化高斯噪声，用参数噪声探索替代纯 epsilon-greedy。
- **C51**：预测回报分布而非单个期望 Q 值，将分布投影回固定 support。
- **n-step return**：用多步奖励构造 target，增加短期真实奖励信息。

C51 的投影把 Bellman 变换后的 atom 位置线性分配到相邻 support atom。当前仓库提供 Rainbow 的主要构件，但尚未把所有构件集成为一个完整、统一训练的 RainbowAgent；组件级实现不能等同于完整 Rainbow 复现。
