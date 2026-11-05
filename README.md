# 短视频生成式召回、精排与曝光偏差评测

Does exposure bias in logged feedback change offline conclusions about recommender models? Retrieval, ranking and generative recommendation on Kuaishou's open KuaiRand and KuaiRec datasets, checked against random-exposure logs and a near-fully-observed user × video matrix. Studies 1 and 2 are pre-registered and each opens its test set once after the model zoo is frozen; Study 3 is a pre-registered extension evaluated on Study 1's already-opened random test set. Docs are in Chinese.

常规推荐日志里，用户只看得到旧推荐策略挑过的视频，离线测试集也是这样筛出来的。这个项目用快手开源的 KuaiRand、KuaiRec 两份数据，拿随机曝光日志和近乎全观测的用户 × 视频矩阵当对照，检验常规离线评测会不会把召回、精排模型的比较结论和点击率预估带偏。共三个研究：研究一、二都在训练正式模型前写好协议和判定规则，模型组冻结后测试集只开一次；研究三沿用研究一开过的随机测试集，是同一测试集上的扩展，不是新的盲测。

这是开发仓库的公开快照：代码、协议、结果文档、配置、测试和证据文件都在。开发日志、上机手册和仓库工作规则（`AGENTS.md`，代码注释和协议里有几处提到它）没有公开；原始数据、缓存、模型权重和完整的运行目录也不在仓库里，结果文档用到的运行记录和诊断副本在 `evidence/`。文档里引用的提交哈希指开发仓库，公开版不带开发历史。

## 结果

**研究一：KuaiRand-Pure 上的召回与精排**（`docs/PROTOCOL.md`、`docs/RESULTS.md`）

常规测试里（精排的共同用户），72.8% 的曝光落在最热的两成视频上；随机曝光日志从 7,583 个视频的池子里近似均匀地抽，当作没被策略筛过的对照。在常规日志上训练 LightGBM、DCN-v2、DIN、MMoE 等 7 个精排模型和 ItemCF、logQ 双塔等 4 个召回模型，两份测试集在同一批用户上、每人等权比较。

- 召回差距到随机曝光上大幅压缩：logQ 双塔对双塔的 Recall 比由 2.61 降到 1.23，logQ 双塔仍排第一。
- 精排模型的 UAUC 一起降了约 0.02，21 对差距的变化都没有检出显著。
- 点击率在随机曝光上整体高估，越冷门的视频高估越多；按热度分组做 Platt 校准后，7 个模型的 LogLoss 都低于全局 Platt。
- 开封后的补充实验：LightGBM 训练集混入约 5% 随机曝光数据，固定训练轮数后，随机曝光测试的 UAUC 提高 0.026。

**研究二：KuaiRec 上的语义 ID 生成式召回**（`docs/kuairec/`）

实现 RQ-VAE 语义 ID 加前缀树约束解码的生成式召回，和 SASRec、logQ 双塔比较。在有借用目标的 1,117 个测试用户上，同一份推荐列表改成只在别人实际被推过的视频里算命中（20 次供体分配平均），语义 ID 对双塔的 Recall@100 点估计由稠密测试上的领先 0.006 变为落后 0.032。正式分析（束宽 100、Recall@100）里，生成式召回在四种测试上都低于 SASRec。

**研究三：生成式推荐的奖励纠偏**（`docs/rlrec/`）

回到 KuaiRand-Pure，拿常规日志上训的点击率模型当奖励，从同一个 SFT 起点出发、在相同 KL 下优化生成式推荐，看策略在随机曝光上的价值会不会上升。优化分两种：解析最优在全池上对起点精确重新加权，不训练模型；RLOO 是真的后训练生成模型。下面都是主档 1 nat、随机曝光上前 500 的列表价值：

- 常规日志奖励自己的打分涨了，但解析最优策略的列表价值没有检出提升；RLOO 出现负迁移，比 SFT 起点低 1.3%（只限这个小模型、几十步的检查点）。
- 用混入随机曝光数据训的奖励，比加入同样多常规数据训的奖励，策略的列表价值更高：解析最优高 0.0057，RLOO 高 0.0045。四种奖励下的 RLOO 都没有检出超过 SFT 起点。

## 结论的边界

- 随机曝光数据仍是事后的离线数据，只去掉一种曝光偏差（旧策略只让用户看它挑过的视频），而且只在 7,583 个视频的随机池范围内成立。
- 测不到新模型上线后自己挑视频、用户反应再影响下一轮推荐的闭环，也测不到随机视频周围仍是旧策略内容带来的列表和位置影响，以及留存、时长这类长期指标。
- 结论只说离线评测里的曝光偏差有多大、会不会改变模型比较的结论，不代表线上效果。
- “未检出”只表示在本研究的样本量下没有检出，不等于两者一样；各检验的最小可检出差见各自的结果文档。

## 目录

| 路径 | 内容 |
|---|---|
| `src/receval/` | 研究一：数据封存与审计、特征、精排与召回模型、调参、校准、统计检验和正式分析，命令行入口 `receval` |
| `src/genrec/` | 研究二：视频文本编码、RQ-VAE 语义 ID、前缀树约束解码、SASRec 与双塔、稠密评测，入口 `python -m genrec <命令>` |
| `src/rlrec/` | 研究三：奖励模型、SFT、解析最优与 RLOO、列表价值评估，入口 `python -m rlrec <阶段>`，顺序见 `src/rlrec/__main__.py`（`sft` 和 `tables` 之间先跑 `scripts/rlrec_reward_models.py`） |
| `docs/` | 各研究的数据审计、协议、结果和文献 |
| `configs/` | 冻结记录和模型组清单 |
| `evidence/` | 结果文档引用的运行登记、配置、汇总指标、数据诊断和模型组清单的逐字节副本，不含原始交互记录和模型权重，可用 `evidence/SHA256SUMS` 核对 |
| `scripts/` | 开封后的补充分析、出表和出图脚本 |
| `tests/` | 合成数据上的单元测试 |

## 运行

需要 Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)；torch 按 `pyproject.toml` 走 PyTorch 的 CUDA 12.6 源，作者在一张 RTX 3080 上跑。

```bash
uv sync --extra test --extra genrec   # 只跑研究一可以不加 genrec
uv run pytest -q
```

数据都不在仓库里，要自己从 Zenodo 下载：

- 研究一：[KuaiRand-Pure.tar.gz](https://zenodo.org/records/10439422) 放到 `.data/raw/`，`uv run receval prepare` 先核对 md5 再解压。
- 研究二：[KuaiRec.zip](https://zenodo.org/records/18164998) 放到 `.data/kuairec/`，再把包里 `data/` 下的 CSV 解压到 `.data/kuairec/raw/`，然后运行 `uv run python -m genrec prepare`。
- 研究三：KuaiRand 作者另发的[视频标题和类目补充文件](https://zenodo.org/records/18159199)，两个 CSV 放到 `.data/kuairand_text/`。
- 研究二、三的文本编码器用 `uv run python -m genrec fetch-encoder` 下载。

冻结记录（`configs/freeze.json` 等）和研究三的代码会核对开发仓库里的协议、模型组冻结提交，公开版没有这些提交，所以研究一、二的开封测试集和正式分析、研究三的 `hold-report` 和 `test` 阶段在这里跑不通。三个研究的测试评估都已在开发仓库里按协议做完，结果以 `docs/` 下的结果文档和 `evidence/` 为准；要在公开版上重做，得重新训练，在本地重建协议和模型组的冻结提交并更新引用（研究三还要改 `src/rlrec/config.py` 的 `PROTOCOL_FREEZE`）。

## 许可

代码使用 MIT 许可。KuaiRand、KuaiRec 数据集归原作者所有，按 CC BY-SA 4.0 许可使用；KuaiRand 的视频标题和类目补充文件按 CC BY 4.0 许可使用。本仓库不转发数据文件。
