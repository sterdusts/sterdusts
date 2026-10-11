"""Write column docs (columns.json) for stablecoins_daily and coinbase_premium(_daily)."""
import json, sys
OUTDIR = sys.argv[1]

stable = {
    "dataset": "stablecoins_daily",
    "source": "DefiLlama 免费接口 stablecoins.llama.fi：/stablecoincharts/all（总量）、/stablecoincharts/all?stablecoin=<id>（单币总量）、/stablecoin/<id>（分链）。下载时刻 2026-10-10 02:25 UTC。",
    "key": ["date"],
    "time_convention": (
        "全部 UTC。DefiLlama 的日期标签 D（时间戳 = D 00:00 UTC）对应的是 D 日【开始】附近的一次快照，不是 D 日收盘。"
        "实证：用以太坊归档节点逐 5 分钟/逐块比对 USDC(Ethereum) totalSupply 与 DefiLlama minted(D)，15 个探测日中精确/近似匹配时刻落在 D-1 23:55 ~ D 02:45 UTC"
        "（2021-06-15 00:00；2022-06-15 00:30；2023-06-15 00:25-01:15；2024-09-15 00:10；2025-03-15 00:05；2025-06-15 01:05；2026-05-15 00:45；2026-07-15 02:45；其余近似在 00:00-02:15），"
        "且当天记录写入后在当日内未被覆盖（2026-10-10 当日记录 02:25 与 02:39 两次下载一致，且对应约 00:05 的链上状态）。"
        "因此：value(D) ≈ D 日 00:00 左右的存量 = D-1 日结束时的存量；value(D)-value(D-1) ≈ 日历日 D-1 内的净流入。"
        "known_at = D 06:00 UTC（保守上界；观测到的最晚快照 02:45）。"
        "与 perp_daily（行 d 在 d+1 00:00 决策）对接的安全规则：按同一 date 直接 join（value(D) 接到 perp 行 D，决策时点 D+1 00:00，余量 ≥18h）。"
        "严禁把 value(D) 接到 perp 行 D-1 或更早（那等于在 D 00:00 使用 D 00:00~02:45 才写入的数据，前视 1-3 小时）。"
    ),
    "point_in_time_caveats": [
        "DefiLlama 历史不是点时数据：今天下载的历史值可能被事后修订/回填（新增链、新增稳定币、修 bug）。回测里的 value(D) 不一定等于当时实时能看到的 value(D)。",
        "era_pre_20220512=True 的区间（2022-05-12 之前）：单币总量 ≠ 分链之和（2022-05-11 USDT 总量 826.9 亿 vs 分链和 810.2 亿；USDC 484.5 亿 vs 429.7 亿），Solana/Tron(USDC)/Algorand 等链从 2022-05-12 才开始有分链数据——这段历史很可能是重建/回填，方法学与之后不一致。",
        "2022-05-12 之前 USDT 日变化多为整 1 亿美元倍数或 0（反映 Tether 国库铸币事件，含已授权未发行库存的口径问题），不是交易所真实入金节奏。",
        "已知方法学跳变：USDC 总量 2022-01-02 单日 +47 亿（38.0→42.8B，此前疑似少计）；total_usd 2022-05-09..12 大跌主要是 UST 脱锚的价格调整（totalCirculatingUSD 按价格计），不是资金流出。",
        "qa_vshape_total_LOOKAHEAD 标出 total_usd 单日>1% 且次日反向回补的 V 形日（2021-08-23/24、2021-12-09、2023-03-12(USDC 脱锚)、2024-04-18 等）；该标记用到 D+1，只能用于诊断，不能作为实时过滤器。",
        "BUSD 2023-02 起的清盘（-200 亿级）会在 total_usd 里表现为长期'流出'，实为稳定币之间转换；如要衡量净新增资金，建议用 usdt_total+usdc_total 或自行剔除 BUSD/USTC。",
        "分链 circulating 是 DefiLlama 口径：桥接到目标链的量计在目标链、从源链扣除（避免重复计数）；新链首次出现日（*_chain_entry_usd）不会造成单币总量跳变（已核对 2024-12-01 Hyperliquid、2025-09-24 Plasma、2026-09-17 Arc），但会造成'分链之和'或单链序列的结构性断点。",
        "最后一行 2026-10-09；2026-10-10 当天记录已存在但未纳入（保持完整日）。",
    ],
    "holdout_rule": "锁定验证集 = 2026-04-01 之后数据（in_holdout=True）。开发期只用 in_holdout==False 的行；注意滚动特征的窗口不得跨入 holdout 去评估。",
    "columns": {
        "date": "DefiLlama 日期标签 D（Date）。含义见 time_convention：D 日开始附近的快照。",
        "snapshot_ts_nominal": "名义快照时刻 D 00:00 UTC（实际快照在 D-1 23:55 ~ D 02:45 之间）。",
        "known_at": "保守可知时刻 D 06:00 UTC。",
        "in_holdout": "date >= 2026-04-01。",
        "era_pre_20220512": "date < 2022-05-12：分链明细不完整、疑似回填重建区间，见 caveats。",
        "total_usd": "全部美元锚定稳定币流通量，按价格折算为美元（stablecoincharts/all 的 totalCirculatingUSD.peggedUSD）。可知：known_at。",
        "total_usd_units": "同上但按枚数（totalCirculating.peggedUSD，不按价格折算；UST 脱锚期与 total_usd 差异大）。",
        "total_allpeg_usd": "所有锚定类型（USD/EUR/JPY/黄金等 VAR...）totalCirculatingUSD 之和。",
        "usdt_total": "USDT 全链流通量（枚，≈美元；stablecoincharts/all?stablecoin=1 的 totalCirculating.peggedUSD）。",
        "usdt_total_usd": "USDT 全链流通量按价格折算美元。",
        "usdc_total": "USDC 全链流通量（枚）。",
        "usdc_total_usd": "USDC 全链流通量按价格折算美元（2023-03 脱锚期 < 枚数）。",
        "busd_total": "BUSD(id=4) 流通量（枚），2020-04 起；2023-02 起清盘。",
        "fdusd_total": "FDUSD(id=119)，2023-05-27 起。",
        "tusd_total": "TUSD(id=7)。",
        "dai_total": "DAI(id=5)。",
        "usds_total": "USDS/Sky Dollar(id=209)，2024-09-18 起（DAI 迁移目标）。",
        "usde_total": "Ethena USDe(id=146)，2023-12-11 起。",
        "ustc_total": "UST/USTC(id=3)，2020-11-26 ~ 2022-06-22（之后 DefiLlama 不再更新）。",
        "usdt_eth / usdt_tron / usdt_sol / usdt_bsc / usdt_arb": "USDT 在 Ethereum / Tron / Solana / BSC / Arbitrum 的 circulating（DefiLlama 分链口径，枚）。首日：eth、tron 早于 2020；bsc 2020-11-11；arb 2021-09-01；sol 2022-05-12。null=该链尚未被跟踪。",
        "usdc_eth / usdc_tron / usdc_sol / usdc_bsc / usdc_arb": "USDC 分链 circulating。首日：bsc 2020-10-20；arb 2021-06-25；tron、sol 2022-05-12（Circle 2024 起停止 Tron 版 USDC）。",
        "usdt_chain_entry_usd / usdc_chain_entry_usd": "当天首次出现（circulating>0）的所有链的首日流通量之和（美元）；0=无新链。用于识别分链结构断点。",
        "usdt_chain_entry_names / usdc_chain_entry_names": "当天首次出现的链名（按首日量降序，逗号分隔）；无则 null。",
        "total_usd_d1 / usdt_total_d1 / usdc_total_d1": "value(D) - value(D-1)，≈ 日历日 D-1 内的净变化（美元/枚）。可知：known_at(D)。",
        "qa_vshape_total_LOOKAHEAD": "诊断标记（用到 D+1！）：total_usd 单日变化>1% 且次日反向、回补过半。",
    },
}

prem = {
    "dataset": "coinbase_premium (hourly) + coinbase_premium_daily",
    "files": {
        "coinbase_premium.parquet": "小时级，key=(ts, asset)，asset ∈ BTC/ETH/SOL，ts 网格 2021-01-01 00:00 ~ 2026-09-30 23:00（50376 小时 × 3）。",
        "coinbase_premium_daily.parquet": "日级，key=(date, asset)，2021-01-01 ~ 2026-09-30（2099 天 × 3）。",
        "coinbase_premium_aux_cb1h.parquet": "Coinbase Exchange 原始 1h K 线（product ∈ BTC-USD/ETH-USD/SOL-USD/USDT-USD；ts=开盘时刻；volume=基础币数量）。",
        "coinbase_premium_aux_bn1h.parquet": "币安现货原始 1h K 线（symbol ∈ BTCUSDT/ETHUSDT/SOLUSDT/USDCUSDT；含 quote_volume、taker_buy_quote_volume；已把 2025 起的微秒时间戳统一）。",
    },
    "source": "Coinbase Exchange 公共 REST /products/{id}/candles?granularity=3600（每请求 300 根，end 含端点，≤3 req/s）；币安现货 data.binance.vision/data/spot/monthly/klines/{SYM}/1h（2025-01 起时间戳为微秒，已归一）。下载于 2026-10-10。",
    "time_convention": (
        "全部 UTC。小时行 ts=h 描述 [h, h+1h)，所有列在 h+1h（该小时收盘，known_at）可知。"
        "日行 date=d 描述 [d 00:00, d+1 00:00)，所有列在 d+1 00:00 可知（与 perp_daily 同口径，可按 date 直接 join）。"
        "实时可得性：Coinbase REST 实时可用（K 线在收盘后数秒~1 分钟出现，最近一根可能被短暂修订）；币安现货在本环境 API 被封，只有归档（月文件次月初、日文件约次日发布）——实盘若要实时溢价需另找 USDT 报价源（如 OKX REST BTC-USDT）。"
    ),
    "definitions": {
        "prem_raw": "cb_close / bn_close - 1。Coinbase 美元价 vs 币安 USDT 价，同一小时收盘价。包含 USDT/USD 基差（USDT 溢价时 prem_raw 偏负，反之偏正）——这正是 CryptoQuant 风格'Coinbase Premium Index'的原始口径。",
        "prem_adj": "cb_close / (bn_close * usdt_usd_cb) - 1。用 Coinbase USDT-USD 同小时收盘价把币安价换成美元，剔除 USDT 基差，更接近'美国现货买盘相对离岸的溢价'。Coinbase USDT-USD 2021-05-04 01:00 才上线，此前为 null。",
        "prem_adj_usdc": "cb_close * usdcusdt_bn / bn_close - 1。备选修正：用币安 USDCUSDT（假设 USDC=1 美元）。2022-10~2023-02（币安 BUSD 自动转换期 USDCUSDT 下架）缺失；2023-03-11 前后 USDC 自身脱锚时不可用。",
    },
    "caveats": [
        "收盘价异步：两所 close 都是该小时最后一笔成交，时间差可达数秒（流动性差时更长，早期 SOL-USD），单小时溢价噪声大；日级建议用 prem_*_mean / median。",
        "USDT-USD 修正：usdt_usd_cb 为当小时 Coinbase USDT-USD 收盘价，无成交小时用此前最多 6 小时内最后收盘价前向填充（usdt_usd_cb_age_h 记录陈旧度，仅用过去数据）；USDT-USD 收盘价本身是成交价非中间价，含 ~0.5-1bp 买卖价差噪声。",
        "缺失：Coinbase BTC/ETH 各缺 13 小时（2023×3、2025×5、2026×5，交易所维护/API 无数据）；SOL-USD 2021-06-17 16:00 才在 Coinbase 上线（此前 SOL 溢价为 null），另缺 15 小时；USDT-USD 缺 15 小时。币安现货缺 14 小时（2021-02-11、03-06、04-20、04-25、08-13、09-29、2023-03-24 等维护）+ 2 根零成交占位 K 线（价格置 null）。",
        "极端值多与真实压力事件相符：2021-05-19 暴跌（BTC prem_adj +2.3%）、2022-11-08/09 FTX（SOL ±3%）、2025-10-10 暴跌（SOL +2.9%）；2023-03-24 11:00 ETH -2.4% 是币安现货停机（12:00、13:00 两小时无 K 线）前的陈旧收盘价造成的假溢价——币安维护/停机前一小时的 close 都可能陈旧（停机小时见缺失列表）。未做任何去极值。",
        "cb_volume_usd_approx = Coinbase 基础币成交量 × (H+L+C)/3，是近似美元成交额（Coinbase K 线不给 quote volume）；cb_vol_share 分母混合了 USD 与 USDT 计价。",
        "Coinbase 现货价格 ≠ 美国全部需求：Coinbase 机构/ETF 托管流（OTC/Prime）不一定在订单簿成交；2024-01 现货 ETF 上线后溢价的经济含义可能发生结构变化。",
    ],
    "holdout_rule": "锁定验证集 = 2026-04-01 之后（in_holdout=True）。开发期只用 in_holdout==False 的行。",
    "columns_hourly": {
        "ts": "小时开盘时刻 UTC。", "asset": "BTC/ETH/SOL。", "known_at": "ts+1h。", "in_holdout": "ts 日期 >= 2026-04-01。",
        "prem_raw / prem_adj / prem_adj_usdc": "见 definitions。",
        "cb_open/cb_high/cb_low/cb_close": "Coinbase {asset}-USD 1h OHLC（美元）；无成交小时为 null。",
        "cb_volume_base": "Coinbase 成交量（基础币）。", "cb_volume_usd_approx": "见 caveats。",
        "bn_close": "币安现货 {asset}USDT 收盘（USDT）；零成交占位 K 线置 null。",
        "bn_volume_base / bn_quote_volume / bn_taker_buy_quote_volume": "币安现货成交量（基础币）/ 成交额（USDT）/ 主动买入成交额（USDT）。",
        "cb_vol_share": "cb_volume_usd_approx / (cb_volume_usd_approx + bn_quote_volume)。",
        "usdt_usd_cb": "Coinbase USDT-USD（≤6h 前向填充），用于 prem_adj。", "usdt_usd_cb_raw": "当小时 USDT-USD 收盘（不填充）。",
        "usdt_usd_cb_age_h": "usdt_usd_cb 的陈旧小时数（0=当小时有成交）。", "usdcusdt_bn": "币安 USDCUSDT 收盘。",
    },
    "columns_daily": {
        "date": "UTC 日 d。", "asset": "BTC/ETH/SOL。", "known_at": "d+1 00:00 UTC。", "in_holdout": "date >= 2026-04-01。",
        "prem_raw_close / prem_adj_close / prem_adj_usdc_close": "d 日 23:00 那根小时 K 线的溢价（即 d+1 00:00 时点快照）。",
        "prem_raw_mean / prem_adj_mean / prem_adj_usdc_mean": "当日有效小时溢价的算术平均（只用非空小时）。",
        "prem_adj_median": "当日有效小时 prem_adj 中位数。",
        "n_hours": "网格小时数（恒 24）。", "n_hours_prem_raw / n_hours_prem_adj": "当日有效溢价小时数（<20 时日均值可靠性下降）。",
        "n_hours_cb / n_hours_bn": "当日 Coinbase / 币安有 K 线的小时数。",
        "cb_close_d / bn_close_d": "d 日 23:00 小时收盘价（= 日收盘）。",
        "usdt_usd_cb_close / usdt_usd_cb_mean": "d 日最后小时（填充后）/ 当日有成交小时均值的 Coinbase USDT-USD。",
        "cb_volume_usd_approx / bn_quote_volume / bn_taker_buy_quote_volume": "当日合计（全天无数据则 null）。",
        "cb_vol_share": "当日 Coinbase 近似成交额占两所合计之比。",
    },
}
json.dump(stable, open(f"{OUTDIR}/stablecoins_daily.columns.json", "w"), ensure_ascii=False, indent=2)
json.dump(prem, open(f"{OUTDIR}/coinbase_premium.columns.json", "w"), ensure_ascii=False, indent=2)
print("ok")
