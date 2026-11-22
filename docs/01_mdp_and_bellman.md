# 01. MDP 与 Bellman 方程

## 1. 为什么用 MDP 描述强化学习？

马尔可夫决策过程（Markov Decision Process, MDP）把智能体与环境的交互抽象为
\(\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)\)。

- \(\mathcal{S}\)：状态集合。
- \(\mathcal{A}\)：动作集合。
- \(P(s'\mid s,a)\)：采取动作后转移到下一状态的概率。
- \(R(s,a,s')\)：这次转移得到的期望即时奖励（也常写作 \(r\)）。
- \(\gamma\in[0,1]\)：折扣因子，控制未来奖励的权重。

马尔可夫性表示：给定当前状态和动作后，未来与更早的历史条件独立：

\[
P(S_{t+1}\mid S_t,A_t,S_{t-1},\ldots)=P(S_{t+1}\mid S_t,A_t).
\]

状态必须尽可能包含预测未来所需的信息；如果遗漏关键历史，问题就不再满足这个假设。

## 2. 回报与价值函数

从时刻 \(t\) 开始的折扣回报：

\[
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
\]

给定策略 \(\pi(a\mid s)\)，状态价值函数是：

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
\]

动作价值函数是：

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
\]

二者关系：

\[
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
\]

## 3. Bellman expectation equation

把回报拆成“即时奖励 + 折扣后的下一时刻回报”：

\[
G_t=R_{t+1}+\gamma G_{t+1}.
\]

对策略和环境转移取期望，得到：

\[
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^\pi(s')\right].
\]

这不是额外假设，而是回报定义与条件期望的展开。

## 4. Bellman optimality equation

最优状态价值满足：

\[
V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^*(s')\right].
\]

对应动作价值：

\[
Q^*(s,a)=\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right].
\]

关键差异：期望方程评估一个给定策略；最优方程则在动作选择处取最大值。

## 5. 从 Bellman 方程到 TD 控制

若环境模型未知，可以用一次采样构造目标。Q-learning 使用：

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[
R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)].
\]

SARSA 使用实际按当前行为策略选出的下一动作：

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[
R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)].
\]

因此 Q-learning 是 off-policy，SARSA 是 on-policy。二者的差别不在于是否使用 TD，
而在于 bootstrap target 中下一动作价值的定义。

终止状态没有未来回报，因此终止转移的 target 只能是即时奖励。时间限制截断（truncation）
是否应 bootstrap，取决于环境语义：若只是人为截断而底层任务仍继续，理论上通常应保留下一状态价值。
本仓库的最小示例把 terminated 与 truncated 都当作 episode 边界，并在边界处停止更新；
更严格的任务会单独处理 time-limit truncation。

## 6. 推荐阅读顺序

1. 先手算一个两状态 MDP 的 \(V^\pi\)。
2. 再推导 Bellman optimality equation。
3. 最后对照 `algorithms/tabular/td_control.py` 中的两个 target，定位 on-policy/off-policy 的差异。
