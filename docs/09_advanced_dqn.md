# 09. Double DQN、Dueling DQN 与 Prioritized Replay

## Double DQN
普通 DQN 的 target 同时用 max 选择并估计动作价值，可能产生 max operator 的过估计。Double DQN 将选择和评估拆开：
\[
a^*=\arg\max_a Q_\theta(s',a),\quad
y=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).
\]
代码中的 `double_dqn_targets` 用 online 网络选动作，再由 target 网络读取该动作的值。

## Dueling Network
将 Q 分解为状态价值与动作优势：
\[
Q(s,a)=V(s)+A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a').
\]
减去优势均值可缓解 V/A 分解的不可辨识问题。Dueling 架构是网络结构，不是独立的 TD target。

## Prioritized Experience Replay
按 TD-error 绝对值定义优先级：
\[
P(i)=\frac{p_i^\alpha}{\sum_k p_k^\alpha},\quad
w_i=\left(\frac{1}{N P(i)}\right)^\beta.
\]
重要性采样权重用于修正非均匀抽样带来的偏差。仓库实现是清晰的 O(N) 教学版本，生产规模建议使用 sum tree。
