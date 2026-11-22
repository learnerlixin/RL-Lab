# 14. Offline RL 与 IQL

Offline RL 只使用固定数据集 \(\mathcal D\)，训练期间不再向环境主动采样。核心困难是分布外动作的 Q 值可能被错误高估。

## IQL 的 expectile value regression
IQL 不在训练时对动作空间直接取 max，而是对数据动作的 Q 值拟合状态价值：
\[
L_V=\mathbb E_{(s,a)\sim\mathcal D}
[|\tau-\mathbb{1}(Q(s,a)-V(s)<0)|
(Q(s,a)-V(s))^2].
\]
代码用双 Q 的较小值作为 target，expectile 参数控制对正负残差的不对称权重。

## Q learning 与策略提取
\[
y=r+\gamma(1-d)V(s'),\quad
L_Q=\mathbb E[(Q(s,a)-y)^2].
\]
策略提取阶段使用 advantage-weighted behavior cloning：
\[
L_\pi=-\mathbb E_{(s,a)\sim\mathcal D}
[\exp(A(s,a)/\beta)\log\pi(a|s)].
\]
仓库提供 IQL 的核心目标函数组件，不包含数据集 loader、完整网络结构或 benchmark runner。
