# 03. 动态规划：已知模型时如何求策略

动态规划假设状态转移与奖励模型已知，通过 Bellman 方程反复 backup 做规划。

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
3. 策略稳定则结束，否则重复。

代码使用 `transitions[s][a]` 存储 `(probability, next_state, reward, terminated)`。终止转移不 bootstrap。表格 DP 不适合连续或超大状态空间。
