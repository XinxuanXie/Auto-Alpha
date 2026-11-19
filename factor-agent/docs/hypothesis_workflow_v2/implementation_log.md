# Hypothesis Workflow V2 实现记录

## 2026-11-14

### V2.1 已完成

- 将 Hypothesis Development 改为批判性审查：每个 Idea 必须同时给出支持、反驳、替代解释，并选择 ACCEPT、REVISE 或同方向 REPLACE。
- Hypothesis Expansion 改为每方向 1–4 个机制不同的假设，允许只保留一个高质量假设；每个假设新增核心机制、最强反驳、证伪标准和同批差异。
- 新增 Measurement Plan，将金融机制拆成可观测代理、字段、局限、Baseline 逻辑和 Full Mechanism 逻辑。
- FactorSpec 改为每假设 2–5 个实验，必须包含 Baseline 与 Full Mechanism，可选 Ablation/Robustness；移除仅靠周期变化扩充候选的设计。
- Experiment 新增 `measurement_plan_id`、`variant_type` 和 `represented_proxy_ids`，建立 Hypothesis → Measurement Plan → Formula 的可追踪谱系。
- Preflight 对首次和修正公式统一检查代理编号、字段覆盖和 Full Mechanism 完整性。
- 假设与实验修正均接收 Checker 的真实结构化问题；实验修正同时接收 Measurement Plan 和同假设已通过公式。
- `RealV2Tools` 改为延迟导入，Mock 和合同测试无需加载真实 LLM/数据依赖。
- 新增 Measurement Plan 代理编号及 Experiment 代理编号唯一性测试。
- 19 个 V2 专项测试通过；完整 Mock 工作流完成 20 个事件，包含 Measurement Plan、两类一次修正以及训练/验证/测试。

### Smoke 修复（repair_experiments 报错，Run `13ac24bc-e6c3-43ac-bc1e-891e1370f709`）

- 报错本身不是 JSON 语法错：`_validated_json_completion` 在本版本已同时 catch `JSONDecodeError` 和 `ValidationError`，真正失败的是 Pydantic 校验——DeepSeek 返回了 `{"op": "add", "field": "close", "args": []}`，`add` 需要恰好 2 个 args 且不能带 field。
- 根因是形状规则没讲全：生成 prompt 只讲了一元算子的 field-or-args，修复 prompt 连这条都没有，而 `FormulaNode.model_json_schema()` 表达不了 `validate_shape()` 的参数个数校验；重试反馈又用 `include_input=False` 去掉了出错节点，LLM 3 次都在重复同一个错误。
- 修复：新增共用的 `FORMULA_NODE_SHAPE_RULES`（一元 / 二元 / 三元算子的参数个数、field/value/window 禁止项、正确与错误示例），生成、修复、重试提示三处共用；重试反馈改为带截断后的出错节点 input（`_truncated_error_input`，上限 2,000 字符）。
- `_normalize_formula_node` 从只处理一元扩展到二元 / 三元：field + 1 个 args 时把 field 还原成 `{"op": "field", ...}` 补进 args，args 已满时去掉多余 field；args 为空时补不齐，留给校验报错走重试，不静默编造第二个操作数。
- `repair_experiments` 改为按实验隔离：单个实验 3 次重试仍失败时只丢弃该实验，其余继续修、继续进第二轮 Preflight 和 train，不再让一个坏公式拖垮整轮。丢弃是可观测的：该实验不会出现在新的 `experiment_specs` 里。
- Checkpoint：`__main__.py` 改为 `SqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=[(V2Stage.__module__, V2Stage.__name__)]))`，消除 `Deserializing unregistered type V2Stage` 警告；allowlist 只放这一个类型。
- 新增 `tests/test_hypothesis_v2_real_tools.py`，11 个新测试覆盖 normalize、形状规则、重试反馈、repair 隔离和 checkpoint 往返；V2 专项测试共 30 个通过。另验证 Mock + 该 serde 的完整流程可完成并可 `--resume` 恢复，无 V2Stage 警告。
- 未改动：Preflight 的语义检查仍是字段覆盖 / 代理编号 / 完全重复，未加自动语义对齐；`_valid_experiment_drafts` 仍未使用。

### Measurement Plan 可实现性修复（13ac24bc 跑完后的复盘，Run `13ac24bc-e6c3-43ac-bc1e-891e1370f709`）

- 现象：恢复跑完后 repair 5/5 修出、第二轮 Preflight 4/8 通过，train 0 通过。4 个没过第二轮 Preflight 的全死在 proxy 覆盖上：`incomplete_full_mechanism`（full_mechanism 没覆盖 Plan 全部 proxy）与 `measurement_proxy_not_implemented`（公式没用到某个已声明 proxy 的全部 required_fields）。
- 根因不在 repair：在 Plan 生成那一步。旧 Plan 一次列 7、8 个 proxy（合同上限就是 8），而 Preflight 两条都是硬性的——full 必须覆盖全部 proxy、公式必须用到每个 proxy 的全部字段。修公式的人两头堵：简化就掉 proxy，全声明又做不出来。实测证据：Reversal P4 的 required_fields 里有 open，但它自己的定义 CLV=(close-low)/(high-low) 根本用不到 open；Price-Volume P3 的定义要算“未来 K 日”收盘价，DSL 禁止用未来数据；Liquidity P1 要算全市场换手率中位数，DSL 里没有全市场统计量。
- 修复：新增 `MEASUREMENT_PROXY_RULES`（proxy 最多 5 个；只用给定字段和 DSL 算子、只用截至当日历史数据；定义里不准出现“未来”；不准用全市场统计量、买卖价差、真实订单流；required_fields 必须恰好是定义里实际用到的字段；分类型 proxy 必须给可计算的判定规则；观测不到的写进 unobservable_aspects；logic 只能引用已定义的 proxy），写进 `design_measurement_plans` 的 prompt；新增确定性检查 `_validate_measurement_plan`（proxy 数量、重复 ID、未来词、全市场短语、required_fields 与定义的一致性——含中文同义词如“日收益→close、换手率→volume+mktcap”和组合 proxy 对被引用 proxy 字段的继承、logic 引用未定义 proxy）；设计 Plan 时检查不通过则修一轮 Plan，修 Plan 本身失败也不让整轮崩掉；修完仍有问题时取违反最少的版本，并把剩余问题以 `PLAN_CHECK:` 记进 `unobservable_aspects`，下游能看到是 Plan 本身有问题。
- 注意：合同里的 `proxies` 上限仍是 8，没改——改了会让旧 Run 的 Plan 读不进来；5 个的上限只由 prompt 与检查器执行。`_validate_measurement_plan` 是启发式检查不是语义证明：字段一致性只在定义提到英文字段或中文同义词时才校验，纯中文且无同义词的定义跳过该项。
- 验证：新增 11 个测试（validator 各项 + 设计 Plan 的修一轮、修不好留档、修 Plan 失败不崩），V2 + DSL 共 54 通过、10 subtests 通过；用真实 DeepSeek 对 13ac24bc 的 4 个假设重跑 `design_measurement_plans`，新 Plan 全部是 5 个 proxy、0 违反（旧 Plan 是 7、8 个 proxy、各 2–4 项违反）。
- 未改动：Preflight 的两条硬性规则没放宽；DSL 没扩。

### Resume 丢 checkpoint 与 train IC 问题（13ac24bc 复盘之二）

- 先纠正一个结论：之前汇报的“13ac24bc 跑完、train 0 通过”测的不是原来那 8 个实验。实测证据：zip 里的 checkpoints.sqlite 只有 9 个 checkpoint，最新只到 design_measurement_plans（step 7）；而 experiment_specs / preflight_results 的时间戳是 15:53，比 sqlite 文件的 15:44 晚。原因是 SqliteSaver 开了 `PRAGMA journal_mode=WAL`，generate 之后的 checkpoint 都落在 `checkpoints.sqlite-wal` 里，zip 里只有主文件没带 -wal；恢复时最新只到设计 Plan，generate_experiments 被整个重跑（cache 也没发过来），8 个实验的公式、名字、周期全换了（Momentum 20 日→5 日，Reversal 基线 5 日→1 日）。repair 5/5 修出、第二轮 Preflight 4/8 的结论仍成立，但 train 的 4 个 IC（0.008–0.014）属于重跑出来的新实验，原来那 8 个从没进过 train。
- 修复：`__main__.py` 新增 `_checkpoint_wal`，`manager.invoke` 包在 try/finally 里，无论跑完还是崩掉，退出前都执行 `PRAGMA wal_checkpoint(TRUNCATE)`，把 WAL 并回主文件；失败只打警告，不掩盖原来的错误。以后只拷 checkpoints.sqlite 也能恢复到崩掉的那一步。验证：WAL 机制的单元测试（没并 WAL 之前只拷主文件连表都不存在，并过之后全在）+ mock 全流程跑完后只拷主文件能读到 stage=completed 的最终 checkpoint。
- train IC 低的另外两处实测原因与修复（`real_tools.py` 的 `generate_experiments`）：
  - Reversal 基线选了 1 日持有：turnover 1.0，10bps 成本一年吃掉约 50%，10 年毛收益 7.52 最后净收益 -0.31；信号本身 IC 0.0143 并不差，是成本杀掉的。新增 `MIN_PREDICTION_HORIZON_DAYS = 5`，prompt 里要求只从 ≥5 日的周期里选，代码里再强制夹住（hypothesis 里没有 ≥5 日的取最长）。
  - Momentum 完整机制 IC 0.0078 低于基线 0.0102：full 把全部 proxy 的项简单加总，稀释了核心信号。prompt 改为 full 必须以 baseline 公式为骨架，用机制项做乘法调节。
- 没动门槛：Mean Rank IC > 0.02 且净收益 > 0 的筛选门槛没改。V2.0 是 84 个实验里 10 个过 train（约 12%），4 个实验 0 通过按这个概率算并不意外（约 60% 的可能），这轮 smoke 的样本太小，结论要等 real-full 的分布。

### 两个完整机制公式编译失败（f1bfbaee 复盘，Run `f1bfbaee-c809-450c-b6c5-e8c2267b01ef`）

- 现象：第二轮 Preflight 6/8，被拒的 2 个都是完整机制：Reversal `rolling_corr() got an unexpected keyword argument 'ddof'`；Liquidity `invalid series dtype: expected Boolean, got f64`（where 的条件不是布尔序列）。
- 根因 1（Reversal，编译器的 bug，不是 LLM 的错）：`compiler.py` 里两处 `pl.rolling_corr` 都硬编码了 `ddof=1`，但 polars 2.x 的 `rolling_corr(a, b, window_size, min_samples)` 根本没有 ddof 参数——相关系数对 ddof 不敏感，分子分母约掉了。被拒公式里的 correlation 节点 ddof 本来就是 null，LLM 没写错，是编译器让任何用到 correlation / rsquare 的公式都编不过。修法：两处 `pl.rolling_corr` 调用去掉 `ddof=1`。副作用：之前全量测试里 16 个失败的 Alpha158 测试，有 14 个就是这个原因，修完后全量测试 160 通过、0 失败。
- 根因 2（Liquidity，LLM 的错）：`where` 的条件写成了 `elementwise_max` 的数值结果，而 `pl.when()` 要求布尔序列；本 DSL 里只有 `greater_than` / `less_than` 产布尔值。修法分三层：`spec.py` 的 `validate_shape` 里 where 的第一个参数必须是比较算子，否则解析时就报错，让生成 / repair 的 3 次重试有机会修正（而不是拖到 Preflight 之后——那时已经没有 repair 轮次了）；`FORMULA_NODE_SHAPE_RULES` 加两条（correlation 不带 ddof；where 第一个参数必须是布尔条件）；`_normalize_formula_node` 不再把 bare field 还原成 where 的第一个参数（field 永远当不了布尔条件，还原了也会被校验打回，改成直接去掉、让校验报缺参数）。
- 注意：旧 artifact 里如果已经存了非法的 where 公式（f1bfbaee 的 Liquidity 实验），新校验会让 `model_validate` 直接报错、读不进来——这些公式本来就编不过，只影响修复前的旧产物，新跑出来的不会有这个问题。
- 验证：用真实 DeepSeek 对 f1bfbaee 被拒的 2 个实验重跑 repair，再对 8 个实验做 Preflight，8/8 通过；修好的 Reversal 公式用了 correlation 且能编译，Liquidity 公式里的 where 条件全是比较算子。另：第一次重跑时 Liquidity 的 repair 曾因 LLM 连续 3 次校验不过被丢弃，第二次重跑成功——repair 本身是非确定性的，单次失败不等于修不好。

### Preflight 信号去重（首轮 real-full 复盘，Run `3d2eeec5-f84c-4f76-b462-e3e7632220b6`）

- 现象：40 个实验看着是 40 个信号，实际只有 5、6 个：Reversal 基线 `return(close,5)` 与 Liquidity 基线 `negate(return(close,5))`（train 方向一吸即同一信号，IC 同为 0.0236）；Momentum 基线与 Price-Volume 基线一字不差都是 `return(close,20)`；Reversal 完整机制与自己的基线 train 指标完全相同（IC 0.0102、净收益 -0.063）。Preflight 原来只查公式文本完全相同的 exact_duplicate，对取负、乘一个接近 1 的量比、where 套壳这类“换个写法、同一个信号”的情况完全没反应。
- 修法：Preflight 增加第二阶段信号去重——对通过编译的每个实验，在 2008–2010 样本（64 只）上算逐日横截面 rank，与同一批实验成对比较 Pearson 相关；|corr| ≥ 0.95 且共同观测 ≥ 500 时判为重复，列表中靠后的一个被拒、issue 代码为 `duplicate_signal`（带对照实验 id 与相关系数），进 repair 轮次；修不好第二轮再被拒就丢弃。rank 向量按实验 id 缓存，Round 1 重检已通过实验时直接复用。注意：只有对照实验本身没有其他 issue 时才作为参照——对照若已因别的原因被拒、修后可能变样，靠后的干净实验不应为复制一个注定要死的实验而被杀，Round 1 会重新检查这对组合。
- repair 的提示也补上：`existing_experiments` 原来只给同假设的已通过实验，跨假设的重复对照看不到、LLM 不知道要跟谁区分；现在把本次被拒实验的重复对照（preflight 记在 `_duplicate_pairs` 里）也加进去。
- 验证（未花 LLM 调用）：对 3d2eeec5 的 40 个真实实验 + 真实 panel 重跑新 Preflight，5.8 秒；17 个被判重复，4 组预期重复全部抓到（Reversal-02-01/Liquidity-02-01 corr -1.00、Price-Volume-01-01/02-03 corr 1.00、Reversal-01-01/01-02 完整机制 corr 1.00、Momentum-02-01/Price-Volume-03-01 corr 1.00）；原来就被拒的 7 个 -R1 实验仍以相同理由被拒；唯一过三关的 LIQUIDITY_VOLATILITY-03-04 未被误杀；5 日 vs 20 日收益这类合理相关的信号（corr 约 0.5）不受影响；Round 1 用缓存 2.9 秒、结果与 Round 0 完全一致。全量测试 166 通过、0 失败（新增 6 个：rank 相关 helper 3 个、Preflight 去重 2 个、repair 跨假设对照 1 个）。

### 完整机制申报派生（首轮 real-full 复盘之二，Run `3d2eeec5-f84c-4f76-b462-e3e7632220b6`）

- 现象：9 个完整机制实验 6 个被拒，5 个理由是 `incomplete_full_mechanism`（申报的 proxy 不全）。把公式逐个拆开看，根因不是公式造不出来，是 LLM 在申报上丢了 proxy：Price-Volume-02-02 只报 P1–P3，但公式里有 `return(close,5)`（P4）和 `std(1 日收益,20)`（P5）；Liquidity-03-02 只报 P1–P3，但 `volume / mean(volume,20)`（P4）就在 where 条件里；Momentum-02-02 只报 P1、P2，但 P3–P5 的计算（`negate(return5)`、`log(均成交额/市值)`、5 日与 20 日区间均值比较）全在公式里。第 6 个是反过来的：报全了 5 个，但公式漏了 `log(mktcap)`。repair 也修不好，因为报错只说“必须覆盖全部 proxy”、没点名缺哪个，LLM 得自己拿公式跟 Plan 对，Round 0 对错、repair 又对错。
- 修法 A（派生覆盖）：Preflight 不再信申报，改按公式实际用的字段派生覆盖（`_derived_proxy_ids`：proxy 的 required_fields 全被公式用上即算覆盖），完整机制按派生结果判；生成与 repair 之后把 full 的申报按派生结果重写（`_normalize_represented_proxies`，按 Plan 顺序），存档里的记录与公式一致。非 full 实验不动申报：对基线/消融来说申报是意图（这个变体 meant to 隔离哪些 proxy），公式顺带用了别的 proxy 的字段不改变意图；派生为空时不重写，留给 Preflight 报错。
- 修法 B（点名报错）：`incomplete_full_mechanism` 列出缺的 proxy id；`measurement_proxy_not_implemented` 列出每个申报了但没实现的 proxy 缺的字段（例如 “P3 needs mktcap”），repair 拿到就能直接补。
- 注意：派生只按字段判断，不证明公式真的按定义算出了 proxy（例如用均值代替中位数、字段对了但算式走样，仍会算覆盖）——这是 Preflight 本来就有的语义缺口，派生没有把它变大，只是把原来藏在申报里的不一致摆到了明处；公式与机制的语义对照仍靠人工检查。
- 验证（未花 LLM 调用）：对 3d2eeec5 的 40 个真实实验重跑新 Preflight，6.3 秒；3 个被错杀的 full 通过（Momentum-02-02-R1、Liquidity-03-02-R1、Price-Volume-02-02-R1），Momentum-01-02-R1 只剩编译错，Reversal-02-02-R1 与 Price-Volume-02-04-R1 仍被拒但报错点名到 P3/mktcap；被拒总数 24→22——另有 1 个是新通过的 full 成为参照后、其近似 twin Liquidity-03-03-R1 被去重拦下，属于去重与派生两处修法的正常联动；归一化后 9 个 full 的申报全部与公式一致（含 Reversal-02-02-R1 的 P3 被如实去掉）。全量测试 173 通过、0 失败（新增 7 个：派生覆盖 3 个、归一化 3 个、repair 归一化 1 个）。

### 文献锚定与防坍缩（第二轮 real-full 复盘，Run `4fa040cd-1c41-402c-83d5-e842faaf9776`）

- 现象：4fa040cd 只有 3 个 train 过关、验证 0 个；6 个被判重复的实验 repair 后仍是重复（rank corr 0.98–1.00），repair 提示只说“必须具有实质差异”，LLM 不知道具体改什么，只能改窗口、取负、乘量比——恰好都不算差异；另有一对漏网：REVERSAL-02-01 与 PRICE_VOLUME-02-04 的 train IC 同为 0.0236，实测去重样本 rank 相关 0.944，刚好在 0.95 阈值之下。
- 修法：1）给四个研究方向各加一份经典价格类异象文献（只列 OHLCV+市值能算的：Jegadeesh & Titman 1993 中期动量、Novy-Marx 2012、George & Hwang 2004 52 周高点；Jegadeesh 1990 / Lehmann 1990 短期反转、De Bondt & Thaler 1985 长期反转；Amihud 2002、Ang 等 2006 特质波动率、Bali 等 2011 MAX、Datar 等 1998 换手率；Lee & Swaminathan 2000、Gervais 等 2001、Parkinson 1980），研究 agent 的洞察必须锚定其中一个文献机制并点名区别；批判性审查对照文献判“是否只是已有因子的重新表述”；假设展开要求每个假设在 claim 或 core_mechanism 里点名文献机制（作者与年份）及与经典构造的区别。2）生成实验加防坍缩规则（`EXPERIMENT_DIVERSITY_RULES`）：非 baseline 实验的主信号必须与 baseline 不同，去掉机制变量后退化成过去收益本身的实验不合格；full_mechanism 除乘法调节外，也可用 where 做条件筛选，二选一。3）repair 的重复修法改成可执行的操作（`DUPLICATE_REPAIR_RULES`）：改用 Plan 里对照实验没用过的 proxy 作主信号，修完与对照的 rank 相关必须低于 0.90。4）去重阈值 0.95→0.90：4fa040cd 同轮 |corr|≥0.80 的 33 对里，0.93–1.00 是一簇，0.88 以下是相对独立的信号，0.90 落在断层上。
- 验证（未花 LLM 调用）：4fa040cd 的 37 个真实实验重跑新 Preflight，漏网那对里 PRICE_VOLUME-02-04 被判 duplicate_signal，REVERSAL-02-01 作为参照保留；3d2eeec5 的 40 个真实实验重跑，被拒 22→26，新增的都是 0.90–0.95 区间的收益类变体，唯一过三关的 LIQUIDITY_VOLATILITY-03-04 未被误杀（它的两个近似 twin 被拒）。全量测试 177 通过、0 失败（新增 4 个：文献覆盖、阈值、防坍缩规则、重复修法各 1 个）。
- 没改的东西：筛选门槛（Mean Rank IC 0.02 + 净收益 > 0）与数据分段（train 2001–2010、验证 2011–2012、测试 2013–2015）一律不动；放宽门槛或用测试集选因子等于自己骗自己，出不来“能用的因子”。

### 退化因子与 train 隔离（第三轮 real-full 复盘，Run `0f45e744-e721-4c38-98a5-d8bb011f77b4`）

- 现象：0f45e744 在 train 崩了，没有可用因子。REVERSAL-03-02 的公式里有一项 `mean(where(volume<1, 1, 0), 20)`（过去 20 日零成交量天数占比），但本 panel 全表 5,260,971 行里 `volume<1` 的有 0 行（最小成交量约 2.8），这一项恒为 0，乘法链把整个因子变成恒 0，横截面没有区分度，train 报 `ValueError: factor produced no valid Rank IC dates`。两个设计漏洞让它一路漏网：Preflight 只要求“至少算出 1 个有限值”，0 也是有限值；train 的 metrics 只在全部实验跑完后才落盘，一个坏因子崩掉，前面 10 个已算好的结果也跟着丢了。单因子重跑核实：30 个里只有这 1 个坏；MOMENTUM-01-05（63 日）train IC 0.0309、REVERSAL-01-04（10 日）0.0220，本来过了 IC 这一关。
- 修法：1）Preflight 加退化因子检查（`degenerate_signal`）：复用去重用的横截面 rank 向量，按日期数有区分度的日期——2008–2010 样本约 780 个交易日，正常因子几乎每天都有区分度；不到 10 天有区分度的判退化，进 repair（`DEGENERATE_SIGNAL_MIN_VARIED_DATES = 10`）。rank 是现成的，检查不增加新的数据计算。2）train 按实验隔离：`RealV2BacktestEngine.evaluate` 里每个实验的回测包在 try/except 里，失败只丢该实验（记进 `engine.failed_experiments` 并打印 warning），其余实验的 metrics 照常返回、整轮完成；验证/测试共用同一个 `evaluate`，同样受益。失败的实验不会进候选名单（filter 只看返回的 metrics），所以隔离不会把坏因子放进验证。3）生成提示加数据事实（`PANEL_ZERO_VOLUME_NOTE`）：本 panel 成交量从不为 0，全表 `volume<1` 有 0 行，不要用 `volume<1`、`volume==0` 或零成交量天数占比这类构造。
- 验证（未花 LLM 调用）：对 0f45e744 的 37 个真实实验 + 真实 panel 重跑新 Preflight，30 秒；4 个被判退化，逐个核实全是真的：REVERSAL-03-02（崩的那个，零成交量项在乘法链里）、REVERSAL-03-01-R1（公式本身就是 `mean(where(volume<1, 1, 0), 20)`）、REVERSAL-03-03-R1（同一项在乘法链里）——后两个本来就因别的原因被拒；新被拒的只有 REVERSAL-02-03-R1 一个：它的 `where(mean(volume/mktcap, 20) < 0.02, ...)` 里 0.02 这个“低换手”阈值错了约 10 倍——本 panel 的 volume/mktcap 中位是 0.23、均值 1.08，20 日均值 < 0.02 几乎从不成立，样本 40,592 行里只有 4 个非零值。MOMENTUM-01-05 与 REVERSAL-01-04 仍通过，无误杀。阈值 10 天不算拍脑袋：37 个实验的有区分度天数分布是 0、0、0、4、101、265、451……（样本共 757 天），0–4 是一簇、101 以上是另一簇，10 落在断层上。隔离的单元测试是 stub 级别的：真实流程里要等下一轮跑出坏因子才算真正演练过。全量测试 181 通过、0 失败（新增 4 个：退化计数 helper、退化 Preflight、生成提示零成交量、train 隔离各 1 个；测试在临时 venv 里跑，benchmark 文件因缺 qlib 未计入）。
- 已知的残留缺口（这次没修）：另两个 train 期中位覆盖率为 0.00 的稀疏因子 PRICE_VOLUME-02-01-R1（有区分度 265 天）与 PRICE_VOLUME-02-04-R1（101 天）仍能通过 Preflight——它们在 2008–2010 样本里有区分度，稀疏只在 train 期（2001–2010）才看得出来，Preflight 的小样本查不到；要拦它们得在 train 面板上算覆盖率，那是另一处改动，不在这次 Patch 8 里。
- 没改的东西：筛选门槛与数据分段一律不动；退化阈值 10 天是宽松下限，只拦恒为常数或几乎恒为常数的因子。

### 文献锚定改机制优先（Patch 9，用户决定，2026-11-15）

- 背景：Patch 7 硬要求每个假设锚定一篇文献（点名作者+年份、说清与经典构造的区别），治好了想法层塌成通用技术指标（0f45e744 是 9/9 点文献，0784f5f3 是 7/7）。但 0784f5f3 的 Amihud 假设说明锚定治的是想法层、不是公式层：假设点了 Amihud 2002，造出来的完整机制与基线的 rank 相关仍有 0.95。用户指出去重阈值与文献相关性不是一回事，决定去重阈值不动（仍 0.90）、不加因子库相关性检查，只把文献锚定改成机制优先。
- 修法：新增 `MECHANISM_FIRST_LITERATURE_RULES` 常量，研究想法与假设展开两处 prompt 改用它，删掉“必须锚定其中一个文献机制”“没有文献锚点的假设不准输出”两句硬要求。新规则：先把核心机制说精确（谁承担什么风险、什么摩擦或谁的行为导致横截面收益差异，以及在什么条件下成立或失效）；经典文献可以引用 0 到 2 篇，说明是沿用、组合还是不同于该文献的经典构造，引用了就要说清区别；没有单篇文献能概括的机制也可以输出，但不要为了凑一篇而硬锚定，也不要编造既无精确机制、又无文献或因子库依据的洞察。批判性审查不动：仍对照文献查“是否只是已有因子的重新表述”，方向文献简报（`_DIRECTION_LITERATURE`）保留作参考。
- 没改的东西：去重阈值仍是 0.90（`DUPLICATE_SIGNAL_MIN_CORRELATION`，测试仍断言 0.90）；没有加因子库相关性检查；筛选门槛与数据分段一律不动。
- 验证：全量测试 182 通过、0 失败（新增 1 个：机制优先规则文本 + 两处 prompt 不再含旧硬要求）。未花 LLM 调用，未跑新一轮，效果待下一轮真实流程检验。

### 完整机制去重豁免 + 覆盖率门槛 + 400 重试（Patch 10，2026-11-15）

- 背景：五轮 real-full 跑完，train 过关 18 个，validation 只出过 1 个还是运气；唯一三关全过且 test 显著的是手动拼的 Amihud 完整机制 M1/M2，但流水线自己的去重规则把它拦在 train 门外（与基线 rank 相关 0.95/0.96，阈值 0.90）。0d53bd4b 的数据还说明了两点：6 个完整机制里 4 个建在基线骨架上仍然输给基线（最惨 0.0339 → 0.0021），所以问题不在骨架、而在调节项本身；PRICE_VOLUME-01-04 以 train 覆盖率 0.00 过了 train 门槛，信号只在极少数股票上。另 0d53bd4b 曾因 HKUST 网关一时 400 崩在批判审查，原样重放同一请求全部通过。
- 诊断：去重规则在系统性地筛掉好完整机制。完整机制按定义就是基线加调节，调节温和时与基线的相关就高（Amihud 那种 0.95–0.96）；能进 train 的只剩相关 < 0.90 的剧烈调节——乘上一个横截面变化很大的项，排名被打乱、IC 也垮了。0.90 阈值本身没错，错的是把它用在“完整机制 vs 自己假设的基线”这一对上：这一对的高相关是设计意图，值不值得由回测判定，不是相关系数。
- 修法：(a) `preflight_experiments` 第二阶段跳过“完整机制 vs 同假设基线”这一对（`_is_same_hypothesis_full_vs_baseline`），阈值 0.90 不动，完整机制 vs 其它假设的基线、以及其它所有配对照常去重；(b) 筛选门槛加第三条腿：`contracts.py` 新增 `passes_selection_gate`（IC > 0.02 且净收益 > 0 且 `median_stock_coverage >= 0.5`），manager 的 train/validation 两处过滤与 `build_memory_summary` 的计数四处共用，0.5 落在实测分布的断层上（稀疏因子 0.00，正常因子 0.99–1.00）；(c) `_completion_content` 对 `BadRequestError` 做有界重试（最多 3 次、每次睡 5 秒，持续 400 仍原样抛出）；(d) real-full 的 `min_hypotheses_per_idea` 从 1 改为 2（0d53bd4b 一轮只有 6 个假设，太薄）。
- 没改的东西：去重阈值仍是 0.90；IC 门槛 0.02 与数据分段不动；没有放宽完整机制覆盖要求；没有加因子库相关性检查。
- 验证：全量测试 188 通过、0 失败（新增 6 个：去重豁免正反两例、400 重试两例、门槛覆盖率、manager 稀疏过滤）。离线验证（无 LLM 调用）：新 Preflight 下 Amihud M1 通过（之前被判 duplicate_signal），M2 仍被拒（缺 P4 是构造时就有意的，且与 M1 相关 0.99——这个拒绝是对的）。

### 新研究方向隔夜–日内 + 换手成本规则 + 审查加日历/换手职责（Patch 11，2026-11-15，manager 授权）

- 背景：六轮 real-full 之后，唯一过 validation 的流水线因子（9c25eb4a 的 03-02）在封存段 2016–2025 上 IC 0.0013、p=0.91，归零；逐日 IC 按月平均显示 12 月 -0.0997、非 12 月 0.0095，信号本质上是日历现象（12 月避税卖盘、1–3 月反弹）。四个老方向的点子池抽干了——六轮产出的全是 Amihud、动量、反转的变体。另外 07–08 块里完整机制、P2、P3 的 IC 全是正的，净收益却全是负的，换手成本吃掉信号，生成时没有任何换手意识。
- 诊断：问题不在筛选（Patch 10 已让完整机制进 train 并赢了基线），在想法层：(a) 四个方向共用同一批价格信息，`open` 字段六轮 213 个实验里只用了 10 次，隔夜/日内分解从没当过主信号；(b) 生成提示不提换手，LLM 可以造出每天翻转的信号；(c) 批判审查不查日历效应，12 月陷阱要到事后诊断才发现。
- 修法：(a) `ResearchDirection` 新增 `OVERNIGHT_INTRADAY`（隔夜收益与日内收益的分解、开盘缺口与隔夜信息到达），文献简报给 Lou, Polk & Skouras 2019 与 Heston, Korajczyk & Sadka 2010，并给出可编译的 DSL 写法（隔夜 = open/lag(close,1)−1，日内 = close/open−1）；state 从 enum 自动派生方向分支，mock 的 idea 描述、概念集与 allowed_concepts 同步加；(b) 新增 `TURNOVER_RULES` 常量并接入 `generate_experiments` 提示：调节项优先 20 日窗口、不用每天翻转的信号；(c) `analyze_insights` 审查多查两点：机制是否只是日历/月份效应、隐含换手频率能否扛住 10bps 成本。
- 没改的东西：去重阈值仍是 0.90；IC 门槛 0.02、覆盖率 0.5 与数据分段不动；封存段 2016–2025 已被 P3 检验用掉，不再用于选变体。
- 验证：全量测试 191 通过、0 失败（新增 3 个：隔夜方向简报含 DSL 写法、换手规则内容与接线、审查含日历与换手检查；另把 3 处按方向数硬编码的 mock 计数测试改成按 `len(ResearchDirection)` 计算）。

### return/lag 节点非法参数自动清洗（Patch 12，2026-11-15，manager 授权，第七轮崩溃复盘）

- 背景：第七轮 real-full `ec610c9d` 在 `generate_experiments` 崩溃——DeepSeek 三次返回的实验 JSON 里 `return` 节点都带了 `min_periods`（窗口 20 配 `min_periods: 15`，窗口 1 配 `min_periods: 1`），schema 拒绝（`return does not accept min_periods`），三次重试后整轮退出。已完成 5 个想法、5 个批判审查、15 个假设、15 个 Plan、15 个 hypothesis_checks；实验、Preflight、回测均未开始。
- 诊断：提示里已有“return 与 lag 只设置 window”，形状规则却从没写清哪些算子能带 `min_periods`——LLM 看到 rolling 算子“才可设置 min_periods”，把带 window 的 return 也当成 rolling 算子。重试反馈给了报错文本，但整批重生成时 LLM 在别的节点上重复同一个错误，三次都撞在同一点上。靠提示防不住这种系统性习惯。
- 修法：`_normalize_formula_node` 在递归后先按 `spec.validate_shape` 的规则清洗参数再走原有分支：field/constant 去掉全部窗口参数；一元算子里非滚动算子（return、lag 等）去掉 `min_periods`，非窗口算子去掉 `window`，非 quantile 去掉 `quantile`，非 std 去掉 `ddof`；滚动算子与 correlation 的 `min_periods` 超过 window 时钳到 window。`min_periods` 在 return/lag 节点上没有语义（return 是点对点比值，不是滚动窗口），去掉不改变算出的值。`FORMULA_NODE_SHAPE_RULES` 加一行点名哪些算子能带 `min_periods`、return/lag 不能带。
- 验证：全量测试 195 通过、0 失败（新增 4 个：return/lag 去 min_periods、field/constant 去窗口参数、滚动算子保留并钳制、非 std 去 ddof/非 quantile 去 quantile）。实测复现：崩溃形状的整批 payload（return 带 min_periods，嵌在 mean/multiply/zscore/std 里）经 `_normalize_experiment_batch` 后通过 `_ExperimentBatch` 校验，return 的 `min_periods` 为 None。

### proxy 声明按公式实际字段派生到全部 variant（Patch 13，2026-11-15，manager 授权，第七轮复盘）

- 背景：第七轮 Preflight 被拒 13 个里 9 个是 `measurement_proxy_not_implemented`，逐个核实发现全是修后的 -R1 ablation/robustness：repair 重写了公式、却原样复制了旧声明，声明与公式对不上——纯成交量公式还挂着 [P1..P4]、隔夜 ablation 的 repair 丢了 open 字段却还宣称代表缺口代理 P1。不是公式有问题，是标签过期了。
- 修法：`_normalize_represented_proxies` 的派生从只覆盖完整机制扩展到全部 variant：声明按公式实际引用的字段派生（plan 顺序），派生为空时不动、留给 Preflight 拒掉。完整机制的覆盖检查不动（派生集必须等于全部 proxy 才算过）。`generate_experiments` 与 `repair_experiments` 末尾本来就调这个函数，两处同时受益。
- 验证：全量测试 197 通过、0 失败（新增 3 个：非 full 声明被改写、子集公式派生子集、repair 的 ablation 继承派生声明；原 `test_non_full_declaration_is_left_alone` 按新行为改写）。离线重跑新 Preflight（`~/workspace/autoalpha-patch13-verify.py`，无 LLM）：第七轮 58 个真实实验，`measurement_proxy_not_implemented` 从 9 降到 0；9 个里 7 个被救回通过，另 2 个被别的规则正当拦下（LIQUIDITY_VOLATILITY-02-05-R1 与另一实验公式完全重复；OVERNIGHT_INTRADAY-01-03-R1 与 MOMENTUM-02-04-R1 跨方向信号重复——隔夜 repair 坍缩成了动量信号）；`incomplete_full_mechanism` 仍是 2 个，完整机制覆盖要求没松动；3 个 train 过关者与 test 候选 MOMENTUM-03-01 全部仍通过。

### where 两参数补全为三参数（Patch 14，2026-11-15，manager 授权，第八轮崩溃复盘）

- 背景：第八轮 real-full `a5a57aea` 在 `generate_experiments` 崩溃——DeepSeek 三次返回的实验 JSON 里一个 `where` 节点只有 2 个 args（schema 要求 3 个），三次重试后整轮退出。已完成 5 个想法、14 个假设、14 个 Plan；实验、Preflight、回测均未开始。无 400/402，余额无问题。
- 诊断：2 参数的 `where(条件, X)` 是 LLM 的“筛选”写法——条件成立取 X，否则缺失；代码库里惯用的 3 参数写法正是 `where(条件, X, 0.0)`。重试反馈给过报错文本，但整批重生成时 LLM 在同一个节点上重复同一个错误。截断后的报错只露出末尾一个 constant 0.0，看不到完整节点，无法逐个点名，只能按形状修复。
- 修法：`_normalize_formula_node` 的 where 分支在去掉 stray field 后，若剩 2 个 args 且第一个是 greater_than/less_than，就补一个 constant 0.0 作第三参数；第一个参数不是比较时不补——没有条件可补的 where 无法恢复，仍走校验报错与重试反馈。`FORMULA_NODE_SHAPE_RULES` 加一行：想写“条件成立取 X、否则取 0”必须写满 3 个 args。
- 验证：全量测试 200 通过、0 失败（新增 3 个：有条件的 2 参数 where 被补全、无条件的仍非法、field + 2 参数先去 field 再补全）。实测复现崩溃形状的整批 payload 经 normalize 后通过 `_ExperimentBatch` 校验，where 为 3 参数、else 为 0.0。

### JSON 语法错误确定性抢救（Patch 15，2026-11-15，manager 授权，第八轮恢复崩溃复盘）

- 背景：第八轮从断点恢复又崩了，仍在 `generate_experiments`——这次是 DeepSeek 返回的 JSON 本身缺逗号（`Expecting ',' delimiter: line 1 column 6436`），三次全是语法错误，整轮退出。5 个想法、14 个假设、14 个 Plan 仍在 checkpoint 里；无 400/402，余额无问题。
- 诊断：重试机制其实已把破损原文与报错位置发回给 LLM，但模型对 6KB+ 的整批 JSON 做外科手术式修补不在行——它重生成整批，又在别处断开。连续三轮 `generate_experiments` 崩在三种不同的输出缺陷上（min_periods、where 参数数、缺逗号），说明靠提示与重试兜不住这类错误，得在解析层兜底。
- 修法：新增 `_salvage_json`，在 `_validated_json_completion` 里 `json.loads` 失败时先试确定性修复再走 LLM 重试：用解析器自己的报错位置补缺失的逗号、冒号，删 `}`/`]` 前的多余逗号，截掉多余尾巴；每次修复后重解析，最多 8 次，修不好返回 None。抢救出的文本仍要过 schema 校验——修错形状过不了校验，不会把坏数据带进流水线，下游还有编译、覆盖、去重、退化检查。
- 验证：全量测试 207 通过、0 失败（新增 7 个：缺逗号、数组缺逗号、多余逗号、缺冒号、不可修返回 None、合法 JSON 原样通过、缺逗号的整批经抢救后通过 `_ExperimentBatch` 校验）。

### V2.1 封版（2026-11-19）

- 本版共 15 次修改（Patch 1–15）：公式形状校验与 repair 隔离、Measurement Plan 可实现性、checkpoint WAL、两个完整机制编译修复、Preflight 信号去重（阈值 0.90）、完整机制覆盖按公式字段派生、文献锚定改机制优先、完整机制去重豁免与覆盖率门槛、隔夜–日内新方向与换手成本规则、return/lag 与 where 参数清洗、JSON 确定性抢救。
- 正式运行：2 轮 smoke + 8 轮 real-full，共 331 个实验；全量测试 207 通过、0 失败。
- 结果：流水线自产因子通过 train + validation + test 三关（IC > 0.02、净收益 > 0、覆盖率 ≥ 0.5，test p < 0.05）为 0。手工拼的 Amihud 完整机制 M1/M2 在旧分段下三关全过且 test 显著，但不计入流水线成绩；新分段（训练 2001–2019、验证 2020–2023、测试 2024–2026）重测全部 328 个因子，三关全过仍为 0，M1/M2 在新分段下验证与测试期是干净的零。
- 封存段 2016–2025 已在 P3 检验中用掉一次，不再用于选变体。后续方案在新分支上做，main 保持 V2.1 封版状态。

### 版本边界

- V2.0 正式 Run `70778337-b7c6-4a41-a250-dc62a0748441` 保留为历史结果：14 个假设、84 个实验、10 个训练通过、0 个验证通过。
- V2.1 增加必填数据契约，旧 Run 不使用新代码恢复；下一次真实实验创建新 `run_id`。
- V1 与 EvoAlpha 代码和结果均保持独立。

### 下一步

- V2.1 已封版，不再往 main 上加修改；后续方案（新研究方向、DSL 扩展、筛选协议调整等）在新分支上验证，验证通过后再决定是否合入。

## 2026-11-13（V2.0 历史实现）

### 已完成

- CLI 新增 `real-full`：每方向 2–5 个假设，每个假设最多 10 个实验。
- 接入 LangGraph SQLite checkpoint，支持使用原 `run_id` 和 `--resume` 恢复。
- FactorSpec 按假设写入独立 cache，后续假设失败时复用已完成 API 结果。
- FactorSpec 改为逐候选校验，同一批中的非法公式不再使合法候选全部失效。
- 明确 `residual`、`slope`、`rsquare` 是单序列滚动趋势算子，并规范化一元算子输入。
- Preflight 先裁剪 2008–2010 和 64 个真实股票，再编译执行公式，避免每个候选重复扫描完整面板。
- 新增 `file_guide.md`，记录所有 V2 文件职责、调用关系和 checkpoint 恢复方法。

### 正式运行进度

Run：`70778337-b7c6-4a41-a250-dc62a0748441`

- 4 个 Research Idea 已完成；
- 4 份 Insight Analysis 已完成；
- 14 个 Developed Hypothesis 全部通过检查；
- 84 个 FactorExperimentSpec 已生成；
- 首次 Preflight：80 个通过，4 个进入一次修正；
- 后续训练、验证和测试结果以该 Run 的最终 Artifact 为准。

## 2026-11-12（V2.0 初始实现）

### 已完成

- 保留 V1 与 EvoAlpha 目录，新增独立 `hypothesis_workflow_v2/` 分支实现。
- 建立 Pydantic contracts、轻量 State、原子 Artifact Repository 与固定 LangGraph Manager。
- 接入四路 DeepSeek Research、学校 OpenAI-compatible 两阶段 Hypothesis Development、DeepSeek FactorSpec。
- 接入确定性 Hypothesis Checker 与 FactorSpec Preflight；失败项各允许修正一次。
- Preflight 同时检查字段、DSL/Pydantic 结构、样本容量、完全重复，并在 2008–2010 样本上实际执行公式，确认能产生有限值。
- 接入 2001–2010 训练、2011–2012 验证、2013–2015 测试的数据分段。
- 实现逐日横截面 Rank IC、训练期方向确定与冻结、Newey–West、BH-FDR、Top 20 日频重叠 cohort、10 bps 成本和组合指标。
- CLI 支持 `mock` 与 `real-smoke`；真实冒烟限制为 4 个方向 × 1 个假设 × 最多 2 个实验。
- 将 Manager 从 `build_mock_manager` 重命名为通用 `build_manager`。

### 运行中发现并修复

1. 学校 OpenAI 网关偶发返回 HTTP 成功但空正文：LLM 适配层将空正文纳入应用级重试。
2. 推理型学校模型在 2,048 token 上限下可能耗尽预算而不输出正文：Development 阶段统一使用 8,192 token 上限，并记录 `finish_reason`。
3. DeepSeek 可能生成语法上接近但不满足 `FormulaNode` 的 JSON，例如给 `return` 错误设置 `min_periods`：结构化输出先经 Pydantic 校验，失败时把精确错误反馈给同一模型并完整重生成一次；Prompt 同时明确算子参数规则。

### 真实冒烟结果

Run：`2625e3be-bf57-4b29-95d0-28c738e7c383`

- 4/4 Research 分支成功；
- 4/4 假设通过 Hypothesis Checker；
- 8/8 FactorSpec 通过 Preflight；
- 8/8 训练回测完成；
- 最高 Mean Rank IC：`0.01682`；
- 最高项训练成本后累计净收益：`0.7122`；
- 训练门槛通过：0；
- 验证与测试未触发，符合固定筛选协议。

### 后续事项

- 将 Artifact 索引和跨轮经验统一写入 SQLite Memory。
- 在正式规模运行前评估 LLM 并发、速率限制和成本预算。
- 分析本轮 8 个候选低于 IC 门槛的原因，再决定优化 Research/Development Prompt、概念可实现性约束或 FactorSpec 变体覆盖。
- 长期保留集 2016–2025 继续保持封存。
