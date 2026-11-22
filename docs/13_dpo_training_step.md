# 13. 从 DPO 公式到一次真实优化步骤

DPO 训练除了目标函数，还需要从 causal LM logits 计算 completion token 的 log-probability。对 token 序列 \(y_1,\ldots,y_T\)：
\[
\log\pi_\theta(y|x)=\sum_{t=1}^T
\log\pi_\theta(y_t|x,y_{<t}).
\]
代码通过 logits 的 causal shift 与 label gather 计算 token log-prob，再用 completion mask 排除 prompt 与 padding。

## 一次训练步骤
1. policy 分别计算 chosen/rejected completion log-prob。
2. reference policy 在 no-grad 下计算对应 log-prob。
3. 构造 DPO margin 并最小化 \(-\log\sigma(\beta(m_\theta-m_{ref}))\)。
4. 反向传播更新 policy，不更新 reference。

这是可插入小型 causal LM 的教学训练步骤。真实大模型还需要 tokenizer collator、动态 padding、混合精度、梯度累积、分布式训练、checkpoint、验证集和数据治理。
