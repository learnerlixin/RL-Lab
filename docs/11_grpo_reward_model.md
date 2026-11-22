# 11. GRPO 目标与奖励模型组件

## Reward Model head
奖励模型通常对 prompt-response 序列输出标量分数。仓库的 `RewardModelHead` 接收外部 encoder 的 token hidden states，做 attention-mask-aware mean pooling，再映射成标量。它不是完整 Transformer，也不包含 tokenizer 或偏好数据训练循环。

## GRPO
对同一 prompt 采样 G 个 completion，使用组内奖励标准化：
\[
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}.
\]
token 级 clipped surrogate 可写为：
\[
L=-\mathbb{E}_{i,t}\left[
\min(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i)
\right].
\]
实现支持 attention mask 和可选参考策略 KL 项。它仍然只是 loss 组件；完整 GRPO 系统必须实现模型生成、组采样、reward、old log-prob 缓存、token mask、优化器循环和评估。
