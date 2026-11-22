# 20. GRPO：从目标函数到实际 optimizer step

此前的 GRPO 模块提供了组内相对优势与 clipped objective。本次新增 `grpo_train_step`，把因果语言模型的 token log-prob 计算、completion mask、PPO-style clipping、可选 reference KL 和 optimizer update 串起来。

## 1. 因果 token 对齐

给定模型输出 logits (Linmathbb R^{B	imes T	imes |V|}) 和 token ids (x)，预测位置 (t) 的 logits 只能预测 (x_{t+1})：

[
log pi_	heta(x_{t+1}|x_{le t})
=logoperatorname{softmax}(L_t)_{x_{t+1}}.
]

因此 logits、labels、completion mask 必须同时右移一位。completion mask 在 token-id 的位置定义，shift 后的 mask 才与预测该 token 的 log-prob 对齐。prompt token 和 padding token 都不进入损失。

## 2. 组内优势

同一个 prompt 采样 (G) 个 completion，得到奖励 (r_{i,j})。按 prompt 分组标准化：

[
A_{i,j}=rac{r_{i,j}-mu_i}{sigma_i+epsilon}.
]

优势在一组内共享同一 prompt 的相对基准。实际应用中需关注组内奖励方差接近零的情况，以及 reward model / verifier 的尺度和偏差。

## 3. Clipped objective

对每个 completion token：

[
ho_t(	heta)=exp(logpi_	heta-logpi_{m old}),
]

[
mathcal L_{m clip}
=-mathbb E_tleft[
min(ho_t A,operatorname{clip}(ho_t,1-epsilon,1+epsilon)A)
ight].
]

可选 reference penalty 使用常见的非负近似 KL 项：

[
D(pi_	heta|pi_{m ref})
approx e^{logpi_{m ref}-logpi_	heta}
-(logpi_{m ref}-logpi_	heta)-1.
]

只在 completion mask 为 1 的 token 上平均。

## 4. 训练步骤边界

`grpo_train_step` 接受模型、optimizer、input ids、attention mask、completion mask、rollout old log-probs 和 group advantages。支持 `[B,L]` 与 `[B,G,L]` 两种输入。模型生成、奖励函数、old/reference log-probs 的 rollout 计算和数据缓存仍由调用方负责；这让训练核心可与不同 Transformers 模型和生成服务适配。

注意：训练时的 prompt token 必须与 rollout 计算 old log-probs 时一致；temperature、top-p 等采样设置会影响行为策略定义。不要将采样分布的 log-probs 与未经对应校正的目标策略 log-probs 混用。大规模训练还需要分布式 rollout、reference/reward 服务、checkpoint、评测和安全过滤。
