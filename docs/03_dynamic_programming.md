# 03. 动态规划：已知模型时如何求策略

动态规划（DP）假设状态转移和奖励模型已知。它通过反复应用 Bellman 方程做规划，不是从未知环境采样学习的无模型算法。

## 策略评估
给定策略 \(\pi\)：
\[
V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)
[r+\gamma V^\pi(s')].
\]
用右侧更新当前值，直到最大变化小于阈值 \(\theta\)。

## 价值迭代
\[
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r|s,a)
[r+\gamma V_k(s')].
\]
每轮做 Bellman optimality backup；收敛后对动作价值取 argmax 得到贪心策略。

## 策略迭代
1. 评估当前策略得到 \(V^\pi\)。
2. 对每个状态做贪心策略改进。
3. 策略稳定则结束，否则回到第 1 步。

实现中的 `transitions[s][a]` 是 `(probability, next_state, reward, terminated)` 列表。终止转移不 bootstrap。表格 DP 不适合连续/超大状态空间。
