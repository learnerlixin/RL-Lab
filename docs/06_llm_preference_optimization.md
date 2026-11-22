# 06. LLM 偏好优化：Reward Model、DPO 与 GRPO

SFT 是监督微调，不等于强化学习。偏好优化利用 chosen/rejected 数据；在线 RL 还需要策略采样、奖励计算和策略更新。

## Bradley–Terry 奖励模型
对 prompt \(x\)、chosen \(y_w\) 和 rejected \(y_l\)，常见偏好模型：
\[
P(y_w\succ y_l|x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
\]
负对数似然：
\[
L_{RM}=-\log\sigma(r_w-r_l).
\]

## DPO
在特定 KL-regularized RL 与 Bradley–Terry 偏好假设下：
\[
L_{DPO}=-\log\sigma\left(\beta[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
]\right).
\]
函数输入 chosen/rejected 的序列 log probability。真实 LM 包装层必须正确处理 padding mask，并明确 log-prob 是 token 求和还是长度归一化。

## GRPO 组内相对优势
对同一 prompt 的一组 completion 奖励 \(r_i\)，常见标准化：
\[
A_i=\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)+\epsilon}.
\]
这只是 GRPO 的一个构件，不是完整 trainer。完整训练还涉及 token-level ratio clipping、KL、采样、mask 和长度处理。

## 风险
Reward hacking、奖励模型分布偏移、长度偏差、偏好噪声和评测集污染都可能导致虚假的优化收益。当前代码是目标函数级教学实现，不是完整大模型训练系统。
