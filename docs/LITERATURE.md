# 相关文献（预注册 §1.3 的出处）

2026-10-30 由 Sol（gpt-6.1-sol，grok-agent）联网核实，摘录是它从原文页面抄回的；页码和链接以原文为准。核实没有找到和本项目完全相同的设定，但这不能当作“首次”的证明。

## 有偏测试与无偏测试下的模型比较

- **Marlin & Zemel, RecSys 2009**：[Collaborative Prediction and Ranking with Non-Random Missing Data](https://www.cs.toronto.edu/~zemel/documents/acmrec2009-MarlinZemel.pdf)，DOI [10.1145/1639714.1639717](https://doi.org/10.1145/1639714.1639717)。
  - 数据：Yahoo! R3。
  - 结论：自选测试上 MM 和缺失机制模型接近，随机测试上后者更好，原文“outperform the baseline methods”（p1）。
- **Schnabel et al., ICML 2016**：[Recommendations as Treatments: Debiasing Learning and Evaluation](https://arxiv.org/abs/1602.05352)。
  - 数据：Yahoo! R3、Coat，以及 MovieLens 半合成数据。
  - 结论：原文“The naive estimator is severely biased”（p4），朴素估计会把模型排错。§6.4 改变过随机（MCAR）样本量。
- **Yang et al., RecSys 2018**：[Unbiased Offline Recommender Evaluation for Missing-Not-At-Random Implicit Feedback](https://www.cs.cornell.edu/~ylongqi/paper/YangCXWBE18.pdf)，DOI [10.1145/3240323.3240355](https://doi.org/10.1145/3240323.3240355)。
  - 数据：citeulike、Tradesy、Amazon book、Yahoo music。
  - 结论：原文“The unbiased evaluator may amplify, diminish, or flip the relative differences reported by AOA.”（p7）。
- **Carraro & Bridge**：
  - 会议版 SAC 2020：[Debiased Offline Evaluation of Recommender Systems: A Weighted-Sampling Approach](https://www.cs.ucc.ie/~dgb/papers/Carraro-Bridge-2020a.pdf)，DOI [10.1145/3341105.3375759](https://doi.org/10.1145/3341105.3375759)。
  - 期刊扩展版 JIIS 58:311–336, 2022：[A sampling approach to Debiasing the offline evaluation of recommender systems](https://link.springer.com/content/pdf/10.1007/s10844-021-00651-y.pdf)，DOI [10.1007/s10844-021-00651-y](https://doi.org/10.1007/s10844-021-00651-y)。
  - 数据：Yahoo! R3、Coat。
  - 结论：期刊版 p22 直接比较模型排序，有偏全量测试的 Kendall τ = 0.6，部分去偏设置“achieving perfect correlation (τ = 1)”。
- **Parajuli et al., RecSys 2026**：[arXiv 2607.25097](https://arxiv.org/abs/2607.25097)。
  - 数据：KuaiRec、MovieLens。
  - 做法和结论：比较 45 个模型在稀疏测试和稠密真值上的排序，原文“there is no uniformly best offline evaluation design”（p1）。

## KuaiRand 相关

- **Gao et al., CIKM 2022**：[KuaiRand](https://arxiv.org/abs/2208.08696)。原论文没有去偏模型的性能基准或排序实验，§4 是“POTENTIAL RESEARCH DIRECTIONS”。
- **Hu et al., TKDE 2025**：[Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation](https://arxiv.org/abs/2303.12973)，DOI [10.1109/TKDE.2025.3552658](https://doi.org/10.1109/TKDE.2025.3552658)。
  - 数据：R3、Coat、KuaiRand。
  - 做法：校准的是倾向分数，不是点击概率。KuaiRand 随机数据 10% 作验证、90% 作测试（p7）。
- **Wang & Rasiah, Discover Computing 2026**：DOI [10.1007/s10791-026-10629-5](https://doi.org/10.1007/s10791-026-10629-5)。在 KuaiRand-27K 上比较了常规和随机曝光下的性能下降，不是按 GAUC 审查模型排序。

## KuaiRand 的使用坑

- README L189：双列界面的 `is_click` 是点击，单列界面是有效播放（valid_play）。
- `tab` 是 0–14 的场景号，README 没给编号和界面的对应。作者在 [issue #3](https://github.com/chongminggao/KuaiRand/issues/3) 里说“值最多的就是单列数据，值第二多的就是双列数据”。
- 常规日志和随机日志同在 4/22–5/08，但不能逐行配对。Pure 版删了池外记录，序列不完整（README L116）。
- `video_features_statistic` 是“over one month”的月均统计（README L271），不能当作曝光前已知的特征。

## 本项目引用的方法

- logQ 采样偏差校正：Xinyang Yi、Ji Yang、Lichan Hong、Derek Zhiyuan Cheng、Lukasz Heldt、Aditee Kumthekar、Zhe Zhao、Li Wei、Ed Chi，“Sampling-bias-corrected neural modeling for large corpus item recommendations”，RecSys 2019，DOI [10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996)。2026-10-30 经 Semantic Scholar 接口按 DOI 核对了标题、作者和年份。

## 研究二（KuaiRec）相关

2026-11-01 由 Gemini（gemini-4-argon，grok-agent）联网核实数据集和 KuaiRec 上的评测做法，摘录是它从原文页面抄回的。方法出处一节按论文标题和 arXiv 编号列出。

- **Gao et al., CIKM 2022**：[KuaiRec](https://arxiv.org/abs/2202.10842)。
  - 小矩阵的来历（§3.1.1）：先挑出彼此看过很多的用户和视频，“For the missing values … we manipulate the online recommendation rule to insert these videos into the recommendation streaming … It takes 15 days for this exposure process.”；稠密度 99.6% 是因为“some users have blocked the videos from certain authors”。
  - 大小矩阵的关系（§3.1.2）：“all user-item interactions in the small matrix are excluded from the big matrix to separate the training and evaluation data.”
  - 正样本（§3.1.3）：累计观看时长超过视频时长两倍。README 的例子是 `watch_ratio > 2.0`。
  - 视频文本 `kuairec_caption_category.csv` 2024-06-02 加入，README 写的是作者添加的封面文字、标题和话题，没说采集时点。
- **Parajuli et al., RecSys 2026**：[arXiv 2607.25097](https://arxiv.org/abs/2607.25097)。整份大矩阵训练、小矩阵作稠密真值，正样本阈值 `r ≥ 0.85` 和 `r ≥ 2.0`，指标 NDCG@k、DCG@k；KuaiRec 上各种稀疏评测设计给出的排序和稠密评测负相关，原文“sparse evaluation inverts the dense ground-truth ranking regardless of any design choice”。
- **Wang et al., WWW Companion 2025**：[Debias Can Be Unreliable](https://arxiv.org/abs/2409.04810)。只用小矩阵，在矩阵内部分出有偏训练集和全曝光测试集。
- **KuaiRec 上的语义 ID 生成式召回**：[GLASS](https://arxiv.org/abs/2602.05663)、[DIG](https://arxiv.org/abs/2605.14853)、[Xie et al.](https://arxiv.org/abs/2608.21012)、[OGR](https://arxiv.org/abs/2608.17613) 都在单个矩阵内按时间留出（leave-one-out 或按日切），没有用小矩阵作稠密测试。检索不穷尽。

## 研究二引用的方法

- 语义 ID 生成式召回：Rajput et al.，“Recommender Systems with Generative Retrieval”，NeurIPS 2023，[arXiv 2305.05065](https://arxiv.org/abs/2305.05065)（TIGER：RQ-VAE 给物品生成语义 ID，再自回归生成下一个物品的码）。
- SASRec：Kang & McAuley，“Self-Attentive Sequential Recommendation”，ICDM 2018，[arXiv 1808.09781](https://arxiv.org/abs/1808.09781)。
- 文本编码器：[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，用的是提交 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
