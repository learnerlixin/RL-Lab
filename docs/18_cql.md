# 18. Conservative Q-Learning（CQL）

CQL 面向 offline RL，核心目标是让数据集动作的 Q 值相对更可信，同时压低分布外动作的过高 Q 估计。离散动作的简化保守正则项：
\[
L_{CQL}=\mathbb E_{s\sim\mathcal D}
\left[\log\sum_a\exp(Q(s,a))-Q(s,a_{data})\right].
\]
实现提供带 temperature 的 log-sum-exp 版本。完整 CQL 算法还需与 TD critic loss 结合，并按具体论文变体处理动作采样、重要性修正和双 Q 网络。
