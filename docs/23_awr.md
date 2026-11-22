# 23. Advantage-Weighted Regression（AWR）

AWR 是离线策略学习方法：用数据集中的状态-动作对做加权最大似然，优势越高的动作得到越大的拟合权重。

## 优势与权重

给定 Critic 估计与数据回报，可以构造：

\[
A(s,a)=R(s,a)-V_\phi(s)
\]

将优势映射为非负权重：

\[
w(s,a)=\min\left(\exp(A(s,a)/\beta),w_{max}\right)
\]

其中 \(\beta\) 控制权重对优势的敏感程度，\(w_{max}\) 用于限制极端样本的影响。

## Actor 目标

\[
L_{AWR}(\theta)=-\mathbb E_{(s,a)\sim D}
[w(s,a)\log\pi_\theta(a|s)]
\]

Critic 可使用数据回报做回归：

\[
L_V(\phi)=\mathbb E_{s\sim D}[(V_\phi(s)-R)^2]
\]

## 与 IQL 的关系

IQL 使用 expectile regression 拟合状态价值，并通过 advantage-weighted behavior cloning 提取策略。AWR 的核心思想相近，但具体 Critic 目标、回报估计与权重定义可因算法变体而不同。本仓库将 AWR 的加权策略损失与价值回归作为独立教学组件，不宣称覆盖所有 AWR 变体。

## 实现边界

- awr_weights：对优势做 detach、指数变换和上限裁剪。
- awr_policy_loss：支持有效样本 mask 的加权行为克隆。
- awr_value_loss：支持 mask 的价值回归。
- 数据集加载、回报/优势计算、策略分布与训练循环由调用方负责。
