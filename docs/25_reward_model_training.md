# 25. Reward Model pairwise training step

RLHF 的奖励模型通常从偏好对学习：给定同一 prompt 下的 chosen 与 rejected completion，模型应给 chosen 更高的标量分数。

## Bradley–Terry 目标

令两个序列的预测奖励为 \(r_\phi(x,y_w)\) 与 \(r_\phi(x,y_l)\)：

\[
P(y_w\succ y_l)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
\]

负对数似然为：

\[
L_{RM}=-\mathbb E[\log\sigma(r_w-r_l)]
\]

## 训练步骤

reward_model_train_step 接收 transformer-style encoder、RewardModelHead、包含两者参数的 optimizer，以及 chosen/rejected 两组 token batch。

1. 分别运行 encoder，读取 last hidden state。
2. 使用 attention mask 做 masked mean pooling，得到 sequence reward。
3. 计算 Bradley–Terry loss。
4. 反向传播、梯度裁剪、optimizer step。
5. 返回 loss、preference accuracy、平均 reward margin 与梯度范数。

## 注意事项

- 该步骤只负责 pairwise reward model 优化；tokenizer、数据集、prompt 构造、验证集与 reward calibration 仍由调用方管理。
- chosen 与 rejected 的 batch size 必须相同。
- optimizer 必须包含 encoder 与 reward head 的所有可训练参数。
- 此处 masked mean pooling 是简单教学选择；实际 RM 常采用 EOS/最后有效 token pooling，并应与预训练模型的训练约定一致。
