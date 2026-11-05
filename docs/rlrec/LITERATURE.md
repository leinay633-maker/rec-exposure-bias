# 研究三的相关文献（生成式推荐的 RL 后训练与奖励偏差）

2026-11-03 由 Sol（gpt-6.1-sol，grok-agent，会话 `01a1a134-6b8b-7462-8ca2-edb2bfc549ef`）联网核实，摘句是它从原文页面抄回的，日期是 arXiv 首次提交的 UTC 日期。待核清单来自另一个模型的网页调研，核实后有几条要改口径，写在各条的“差异”里。没有找到和研究三完全相同的设定，但各个组成部分都有先例，不能写“首次”。

## 1. 待核清单

- **OneRec（快手）**：
  - 论文 [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965)，第一作者 Jiaxin Deng，2025-02-26。§3.1、算法 1 用平衡残差 k-means 做多层语义 ID；§3.3 用个性化奖励模型选偏好对、迭代 DPO（IPA），原文“We propose a novel self-hard negative samples selection strategy based on personalized reward model.”
  - 另一份 [OneRec Technical Report](https://arxiv.org/abs/2506.13695)，署名 OneRec Team，2025-06-16（v4 2025-09-16）。对齐算法换成 ECPO（Early Clipped GRPO），§4.3.1、表 9 讨论了轻微的 reward hacking。
  - 差异：两份报告的对齐算法不同（IPA/DPO 对 ECPO），引用时写明哪一份。正式发表处、专属代码都没确认。
  - 可以引用：OneRec 把平衡残差量化的语义 ID 生成和奖励模型引导的偏好对齐结合起来。
- **OneRec-V2（快手）**：[OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900)，2025-08-28（按 v4 2025-10-28 核对）。
  - §1、§3.1：“In OneRec-V2, we introduce RL based on user feedback signals as rewards.” 讨论只靠代理奖励的 hacking 风险，用真实反馈、按同长度视频比较的时长奖励塑形和 GBPO（Gradient-Bounded Policy Optimization）。
  - 差异：反馈来自 OneRec 和传统推荐流的实际曝光，不是随机曝光；旧链路的行为概率还做了近似。
  - 可以引用：V2 用实际曝光后的用户反馈代替单纯的模型代理奖励。不能说它用随机曝光消除了曝光偏差。
- **OpenOneRec（快手）**：[OpenOneRec Technical Report](https://arxiv.org/abs/2512.24762)，2025-12-31（v2 2026-02-04）。代码 [Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)，核实快照 `a969edcadd579a06c1966ae1db5984e02f48beff`；README 第 205 行写代码 Apache-2.0、权重另有许可，根目录没有 `LICENSE`。
  - 原文：“We release the OneRec-Foundation series, comprising models with 1.7B and 8B parameters.”
  - 差异：推荐 RL 用的是命中目标的规则奖励；RecIF-Bench 的用户随机切分不是物品随机曝光。
  - 可以引用：发布推荐基座模型、跨任务基准和训练流程。不当成奖励模型曝光纠偏的基准。
- **MiniOneRec（中科大、NUS）**：[MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation](https://arxiv.org/abs/2510.24431)，第一作者 Xiaoyu Kong，2025-10-28。代码 [AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)，Apache-2.0，核实快照 `526306229326013a7ca46f5b2d8f55ea669aa159`。
  - §3.2、§3.4 是 RQ-VAE → SFT → GRPO；默认奖励是规则命中加排序奖励。§5.5.3、图 4(c) 的消融里，冻结 SASRec 当协同奖励反而退化：“as recommendation accuracy declines, the reward continues to increase”。
  - 可以引用：开放的语义 ID 生成式推荐后训练流程；观察到协同代理奖励上升而准确率下降，作者解释为 reward hacking，但没有用随机曝光识别原因。
  - 约束解码问题见 §2。
- **ReRe（阿里淘天、NUS）**：[Reinforced Preference Optimization for Recommendation](https://arxiv.org/abs/2510.12211)，第一作者 Junfei Tan（与 Yuxin Chen 共同一作），2025-10-14。代码 [sober-clever/ReRe](https://github.com/sober-clever/ReRe)，快照 `e9ba6a29b2f3bd423361f4f4c467adc83bb988fc`，没有仓库级许可证。
  - §2.3、§4.2、图 3：“their values continue to rise even when recommendation accuracy declines”。GRPO、约束束搜索、规则奖励加排序惩罚，研究语义和 SASRec 协同代理奖励的失配。
  - 差异：和 MiniOneRec 有多位共同作者，但机构不完全相同；生成的是物品标题，不是语义 ID。准确率的裁判仍是常规日志目标。
- **Rank-GRPO（UVa、Netflix、Cornell）**：[Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](https://arxiv.org/abs/2510.20150)，第一作者 Yaochen Zhu，2025-10-23，ICLR 2026。代码 [yaochenzhu/Rank-GRPO](https://github.com/yaochenzhu/Rank-GRPO)，快照 `0808d82c0807396c07b4f4815a6f9bd9c4f03ae8`，没有仓库级许可证。
  - §2.3.2–2.3.3：“Rank-GRPO considers each rank as the unit for both advantage estimation and importance reweighting”。
  - 差异：对象是自然语言列表输出的对话推荐（Reddit-v2、ReDial），不是语义 ID，也不涉及随机曝光。
- **Exp-RSFT（Stanford、Meta、Netflix）**：[Robust Post-Training for Generative Recommenders: Why Exponential Reward-Weighted SFT Outperforms RLHF](https://arxiv.org/abs/2603.10279)，第一作者 Keertana Chidambaram，2026-03-10，没有可核实的代码。
  - 原文：“By never querying a learned reward model, the method sidesteps reward model unreliability at its root.” 方法是按 `exp(r/λ)` 加权的 SFT，r 是日志里已观测的奖励。PPO 和用奖励模型造偏好对的 DPO 在它的实现里出现代理分高、排序指标低。数据是 MovieLens、Amazon Books 和私有数据。
  - 差异：不是“纠正奖励模型”，而是绕开奖励模型；不能概括成所有 DPO 都会崩。
  - 同组 2026-08-01 的 [2608.00816](https://arxiv.org/abs/2608.00816) 和它高度重叠，多讨论了日志策略覆盖不足和反馈噪声两种成本，不是独立复现。
- **AdaGRPO（京东等）**：[Adaptive Loss Balancing for Noise-Robust GRPO in Generative Recommendation](https://arxiv.org/abs/2606.08480)，第一作者 Kewei Xu（与 Junbo Qi 共同一作），2026-06-07。正文带 `XXXX` DOI 和 RecSys 2025 模板，不能据此认定已发表；没有代码。
  - §4.1–4.4、§5：“The net contribution is negligible or negative.” 门控只让“策略觉得难、奖励模型又能区分目标”的样本贡献 GRPO 梯度，NLL 对全部样本保留。
  - **差异（必须改口径）**：表 1 的“接近零”是同一束搜索候选组里大模型和精排奖励模型给目标排的位次差（K = 50 时 +0.48 位，K = 128 时 −0.28 位），不是 GRPO 训练的平均净收益；作者也说明束搜索诊断和训练时的采样分布不同。
  - 可以引用：生产精排奖励的价值是有条件的，AdaGRPO 用样本级门控减少不可靠奖励的更新。不能说门控让奖励在随机曝光下变得校准。
- **CausalDPO（东北大学、中山大学）**：[Causal Direct Preference Optimization for Distributionally Robust Generative Recommendation](https://arxiv.org/abs/2603.22335)，第一作者 Chu Zhao，2026-03-21，ICML 2026。代码 [user683/CausalDPO](https://github.com/user683/CausalDPO)，快照 `e49b49f5eb4815e47126ee13277a23e84b954dbf`，没有仓库级许可证。
  - §3.1–3.3：“DPO tends to amplify spurious correlations caused by environmental confounders during the alignment process”。后门调整、软聚类伪环境和不变性约束。
  - 差异：Book-Crossing 的全曝光测试是合成的；总提升数字在 arXiv 摘要（17.17%）和会议页、README（24.10%）之间对不上，不引用这个数。

## 2. MiniOneRec 的约束解码问题

- 出处：[Issue #49](https://github.com/AkaliKong/MiniOneRec/issues/49)，2026-01-03 提出，Qwen2.5-3B-Instruct 复现，评估输出 `CC=30119`（[calc.py 第 62 行](https://github.com/AkaliKong/MiniOneRec/blob/526306229326013a7ca46f5b2d8f55ea669aa159/calc.py#L62)，统计生成的不在有效物品字典里的次数）。作者 2026-01-04 回复：“there are still a large number of invalid items being generated, and constrained decoding has failed.” 同日写进 README 第 24 行，只怀疑依赖版本，建议改用 base 模型。
- 修复：2026-01-31 的 [cc20eb70](https://github.com/AkaliKong/MiniOneRec/commit/cc20eb7075b9012babd0542d4b5c4d048cfbfd48) 加了 `top_k=None、top_p=None`；[PR #82](https://github.com/AkaliKong/MiniOneRec/pull/82) 于 2026-09-07 合并（合并提交 `0518daebf368319f9240a8ffc2c1239c841ae6e4`）。原因是 `transformers >= 4.50` 会把自定义 GenerationConfig 里等于全局默认的 `do_sample=False` 换成模型默认的 `True`，所以要在 `generate(..., do_sample=False)` 的参数里覆盖。修复者报告在 4.57.1 上验证过。
- 边界：独立评估（`evaluate.py` 第 192、204 行）已修；训练中的评估（`minionerec_trainer.py` 第 759 行）还只传 `test_generation_config`，这是静态代码观察，没有运行验证。研究三阶段 6 如果参考 MiniOneRec，固定到含修复的提交，并自己检查有效语义 ID 比例是 100%。

## 3. 相关先例

- **随机曝光或全观测数据上的推荐 RL**：
  - DORL：[Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation](https://arxiv.org/abs/2307.04571)，Chongming Gao 等（中科大、浙大、快手），SIGIR 2023，代码 MIT。“conservatism in offline RL can incur the Matthew effect in recommendation.” 用 KuaiRec 稠密矩阵和 KuaiRand 随机曝光派生的环境；KuaiRand 没观测到的对要补全，模拟回报不全是真实随机标签。
  - ROLeR：[Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems](https://arxiv.org/abs/2407.13163)，Yi Zhang 等（昆士兰大学、CSIRO），CIKM 2024，代码 MIT。“a non-parametric reward shaping method is designed to refine the reward model.” 只有 KuaiRec 能直接当稠密奖励，KuaiRand、Coat、Yahoo 的测试奖励由 DeepFM 补全。
  - DARLR：[Dual-Agent Offline Reinforcement Learning for Recommender Systems with Dynamic Reward](https://arxiv.org/abs/2505.07257)，同组，SIGIR 2025；仓库目前只有 README 和 LICENSE。比较训练中估计奖励和裁判奖励的差，动态修正奖励。
- **少量随机数据纠正有偏模型**：
  - AutoDebias：[Learning to Debias for Recommendation](https://arxiv.org/abs/2105.04170)，Jiawei Chen 等，SIGIR 2021，代码 MIT。“leverages another (small) set of uniform data to optimize the debiasing parameters”，用 Yahoo! R3、Coat 的随机数据做元学习去偏。
  - CausE：[Causal Embeddings for Recommendation](https://arxiv.org/abs/1706.07639)，Stephen Bonner 等（Criteo），RecSys 2018。“leverages a small sample of randomized recommendations outcomes”；MovieLens、Netflix 实验的随机曝光是重采样模拟的。
  - 倾向校准：[Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation](https://arxiv.org/abs/2303.12973)，TKDE 2025，用 KuaiRand，含 Platt；校准的是曝光倾向，不是用户效用奖励（研究一的 `docs/LITERATURE.md` 也引了它）。
- **DPO 和热门偏差**：SPRec：[Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243)，Chongming Gao 等，WWW 2025。“DPO exacerbates popularity bias in LRSs”，实验不是真实随机曝光。
- **解析最优策略和相同 KL 的比较**：
  - DPO 原论文 [2305.18290](https://arxiv.org/abs/2305.18290)（NeurIPS 2023）§4 式 4 给出 π_r ∝ π_ref·exp(r/β)，图 2 画奖励–KL 前沿：“achieving slightly higher reward but with much higher KL is not necessarily desirable.”
  - [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)（Leo Gao 等，ICML 2023）比较代理和真值奖励随 KL 的变化，同时提醒：“it should not be used to compare the amount of optimization between different optimization algorithms.” 研究三在同一种优化方式内按 KL 对齐，跨方式（解析解对 RL）只比价值–KL 曲线，不说“优化强度相同”。
- **奖励模型基准**：RecRM-Bench [2605.11874](https://arxiv.org/abs/2605.11874)（美团、复旦，2026-05-12），多维奖励模型基准，数据来自美团真实日志，不是随机曝光。

## 4. 2026 年 6 月以后的工作

- **RGD**：[Reward Guided Decoding for Generative Recommendation](https://arxiv.org/abs/2607.25344)（中科院信工所、国科大、快手、北大，2026-07-28）。推导 Q\* ∝ P·exp(R/β)，在解码时用奖励模型做局部策略改进，不重训生成器；公开实验是 Amazon。研究三的 O3（解析最优策略）和它思路相同，引用它，不说 O3 是新方法。
- **OxygenREC-v2**：[2607.24255](https://arxiv.org/abs/2607.24255)（京东，2026-07-27），可验证命中奖励加熵感知自蒸馏，“enabling reward-model-free policy optimization”，对照外部精排代理奖励，私有日志。
- **HRPO**：[Hierarchical Residual Policy Optimization for Generative Recommendations](https://arxiv.org/abs/2608.00750)（港城大、快手，KDD 2026），代码 Apache-2.0（v1.0.0 = `cef658bf6530762e40be02d10f882fbd9c4ce01d`）。在 KuaiRand 上做语义 ID 后训练、比较 DPO 和 GRPO，按用户簇平滑前缀级奖励；主要裁判是 KuaiSim 模拟反馈，仓库用的是 `log_standard_*`，不是随机曝光测试。**这是研究三最近的邻居**，研究三要在协议里写明差别：真实随机曝光当裁判，比较的是奖励而不是优化器。
- **GRP v0.1**：[2609.36688](https://arxiv.org/abs/2609.36688)（Snap，2026-09-29）。冻结的联合精排头当奖励，“vanilla GRPO frequently produced no gain or a net loss”，加参考策略锚定的目标 margin，并强调奖励头的校准。
- **EWM**：[Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning](https://arxiv.org/abs/2610.13504)（Spotify、剑桥，2026-10-09）。附录 B.1：“Each evaluation session includes a held-out target from the randomly exposed interactions.” 用 Qwen2.5-1.5B-Instruct 做解释加权 SFT，测试目标来自随机曝光（1,000 个会话）。**真实随机曝光目标参与生成模型后训练评测已有先例**，但它不是校准有偏精排奖励后比较 GRPO。
- **GR4AD-X**：[2610.38378](https://arxiv.org/abs/2610.38378)（快手，2026-10-28），前缀级奖励优化、价值引导的束剪枝和判别式重排；正文里的 WWW 2027 信息不能当已发表。
- **UAGR**：[Utility-Aware Generative Recommendation](https://arxiv.org/abs/2611.02981)（快手，2026-11-02），在 KuaiRand-27K 上比较 GRPO 训练时对齐和推理时效用引导，leave-one-out 评测，没有把随机曝光单独留作裁判。不能因为用了 KuaiRand 就说是无偏评测。
- **HERO**：[Diverse-Teacher On-Policy Distillation for Generative Recommendation](https://arxiv.org/abs/2611.03272)（中科大、NUS、阿里，2026-11-02）。“These two sources of bias reinforce each other and reduce recommendation diversity.” on-policy 蒸馏里教师偏热门和学生 rollout 覆盖不足互相放大，对象是蒸馏，不是独立奖励模型的曝光校准。

## 5. 研究三的定位

- 没找到同时满足这几条的工作：用真实随机曝光数据校准有偏的精排奖励模型 → 同一个语义 ID 生成策略做 RL 或 DPO → 在相同 KL 下比较不同奖励 → 用独立的随机曝光标签裁判。也没找到直接检验“少量随机数据做分组 Platt 校准能改善语义 ID 的 GRPO”的论文。
- 但每个部分都有先例：ROLeR、DARLR 的奖励修正，AutoDebias、CausE 的随机数据去偏，SPRec 的 DPO 热门放大，DPO 原论文和 RGD 的解析策略，HRPO 和 EWM 在 KuaiRand、随机曝光上的后训练。研究三写成对这个具体组合的实证检验，不写“首次”。
- 写作时把三种“裁判”分开：真实随机曝光标签、补全的奖励矩阵（ROLeR 等）、模拟器反馈（HRPO、DORL 等）。

## 6. 总表

| 简称 | 编号 | 首次提交 | 核实状态 |
| --- | --- | --- | --- |
| OneRec | [2502.18965](https://arxiv.org/abs/2502.18965) | 2025-02-26 | 已核实；对齐是 IPA/DPO |
| OneRec 技术报告 | [2506.13695](https://arxiv.org/abs/2506.13695) | 2025-06-16 | 已核实；对齐是 ECPO，和上一篇分开引 |
| OneRec-V2 | [2508.20900](https://arxiv.org/abs/2508.20900) | 2025-08-28 | 已核实；用户反馈不是随机曝光 |
| OpenOneRec | [2512.24762](https://arxiv.org/abs/2512.24762) | 2025-12-31 | 已核实；代码和权重许可分开 |
| MiniOneRec | [2510.24431](https://arxiv.org/abs/2510.24431) | 2025-10-28 | 已核实；解码问题和修复都找到 |
| ReRe | [2510.12211](https://arxiv.org/abs/2510.12211) | 2025-10-14 | 部分不符；机构、生成对象要分清 |
| Rank-GRPO | [2510.20150](https://arxiv.org/abs/2510.20150) | 2025-10-23 | 部分不符；是对话列表推荐 |
| Exp-RSFT | [2603.10279](https://arxiv.org/abs/2603.10279) | 2026-03-10 | 部分不符；绕开奖励模型，不是纠正 |
| AdaGRPO | [2606.08480](https://arxiv.org/abs/2606.08480) | 2026-06-07 | 部分不符；“接近零”是位次差，不是训练净收益 |
| CausalDPO | [2603.22335](https://arxiv.org/abs/2603.22335) | 2026-03-21 | 已核实；总提升数字版本冲突，不引 |
