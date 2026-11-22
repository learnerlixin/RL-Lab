# 04. Monte Carlo 与 TD(0)

## Monte Carlo
完整 episode 结束后，使用实际折扣回报：
\[
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
\]
First-visit MC 对每个 episode 中状态第一次出现的回报做样本平均。它不需要模型，也不 bootstrap，但必须等 episode 结束，且回报方差可能较大。

## TD(0)
TD(0) 使用一步采样构造 bootstrap target：
\[
V(S_t)\leftarrow V(S_t)+\alpha[
R_{t+1}+\gamma V(S_{t+1})-V(S_t)].
\]
它可在线更新，不必等待完整 episode；代价是 target 使用当前估计值，可能带来 bootstrap bias。

## Bias-variance
MC 使用完整回报，通常无 bootstrap bias（在正确采样下）但方差可能较高。TD 使用估计值 bootstrap，更新及时且方差通常较低，但受当前估计偏差影响。TD(λ) 通过 eligibility trace 在多步回报间折中。
