"""Write upbit.columns.json (field docs + availability). Usage: python3 -I write_columns_json.py <out_path>"""
import json, os, sys
out = sys.argv[1]
assert not os.path.exists(out), "refusing to overwrite"
AV = "d+1 00:00 UTC"
doc = {
 "dataset": "upbit (韩国 KRW 资金流)",
 "built_utc": "2026-10-10",
 "source": "Upbit 公共 REST API: /v1/market/all (2026-10-10 02:27 UTC 快照), /v1/candles/days (count=200 向后翻页, ~8 req/s), /v1/candles/minutes/60 (上币首日); "
           "Binance data.binance.vision spot BTCUSDT 1h 月度归档 (SHA256 已校验); ECB eurofxref-hist; FRED DEXKOUS; /home/user/data/kl1d.parquet (只读, 永续映射)",
 "time_convention": "全部 UTC。【重要更正】Upbit 日 K 并不是按 KST 00:00 (=UTC 15:00) 切分，而是按 UTC 00:00 (=KST 09:00) 切分："
                    "API 返回 candle_date_time_utc 全部为 T00:00:00、candle_date_time_kst 全部为 T09:00:00；且在 2021-03-15、2021-11-10、2022-06-20、2023-08-01、2024-03-05、2025-04-07 及 2026-10-05..08 "
                    "对 KRW-BTC/KRW-XRP 用 24 根 60 分钟 K 线(UTC 00:00..23:00)求和，与日 K 成交额完全一致（按 KST 日求和则相差 −32%..+21%），日 K 收盘价 = UTC 23:00 小时 K 收盘价。"
                    "因此行 date=d 描述 UTC 区间 [d 00:00, d+1 00:00) = KST [d 09:00, d+1 09:00)，收盘时点 = d+1 00:00 UTC，与币安 1d K 线完全同窗。",
 "survivorship": "只含 2026-10-10 快照时仍在交易的 KRW 市场（294 个；日表 290 个，另 4 个 10 月才上币）。/v1/market/all 不含已下架币，且已下架市场的 candles 接口返回 404 'Code not found'"
                 "（已实测 KRW-LUNA、KRW-SRM、KRW-LOOM、KRW-XEM），旧 CRIX 接口同样 404；上币公告接口被 Cloudflare 拦截，无法补齐。后果：(1) 2021–2023 年上币事件严重缺失（存活上币数 2021:11, 2022:5, 2023:12；"
                 "实际上币数远多于此）；(2) 全市场成交额合计只覆盖存活币，越早越低估、且 BTC/大币占比被高估；(3) 任何“韩国热度→未来收益”检验都会因只看到存活者而偏乐观。",
 "files": {
  "upbit_daily.parquet": {
   "key": ["date", "market"], "rows_note": "2017-09-25..2026-09-30（任务要求 2021-01~2026-09；2021 之前的行保留作预热，可自行过滤）",
   "columns": {
    "date": "UTC 日期 d；K 线区间 [d 00:00, d+1 00:00) UTC。可知：—",
    "market": "Upbit 市场代码 KRW-<BASE>（仅存活市场）。可知：—",
    "base": "币种代码（Upbit 命名；同名冲突时 Upbit 用 '2' 后缀，如 META2/MET2/GAME2/UP2/FCT2）。可知：—",
    "candle_date_time_utc": "API 原字段，= d 00:00 UTC（tz-aware）。可知：—",
    "candle_date_time_kst": "API 原字段，= d 09:00 KST（naive，KST=UTC+9）。可知：—",
    "available_at": "= d+1 00:00 UTC。注意：Upbit 会把时间戳在 00:00 之后最多 ~138 秒的成交仍计入 d 日（见 close_staleness_min<0），实盘建议 d+1 00:05 再取数",
    "open_krw": "d 日开盘价 KRW。可知：d 00:00",
    "high_krw": "d 日最高价 KRW。可知：" + AV,
    "low_krw": "d 日最低价 KRW。可知：" + AV,
    "close_krw": "d 日收盘价 KRW（= d+1 00:00 前最后一笔成交价，API 的 trade_price）。Upbit 对改名/换币/缩股做了历史价格回溯调整（BTT 1000:1、STRAX 10:1 等可见）。可知：" + AV,
    "prev_close_krw_api": "API prev_closing_price（未回溯调整；2021 以来 99.97% 等于上一行 close_krw）。可知：d 00:00",
    "acc_trade_price_krw": "d 日 KRW 成交额（API candle_acc_trade_price）。可知：" + AV,
    "acc_trade_volume": "d 日成交量（币本位，随价格回溯调整）。可知：" + AV,
    "last_trade_ts_utc": "d 日最后一笔成交时间（API timestamp）。可知：" + AV,
    "close_staleness_min": "(d+1 00:00 − last_trade_ts_utc) 分钟；>0 说明收盘价陈旧（中位 0.2 分钟，p99 30 分钟，最大 1440）；<0 表示 00:00 之后的成交被计入 d 日（9808 行，绝大多数 <1 秒，最大 138 秒）。可知：" + AV,
    "days_since_prev_bar": "与该市场上一根日 K 的间隔天数；首根为 null；>1 为缺口（只出现在 2017–2018 维护期，2021 以来 0 个缺口）。可知：d 00:00",
    "first_bar_date": "该市场第一根日 K 的日期（见 upbit_listings）。可知：上币当日起",
    "age_days": "date − first_bar_date。可知：d 00:00",
    "is_stable": "base ∈ {USDT,USDC,USDE,USDS,USD1,PYUSD,RLUSD,USDG,EURC,JPYC,XAUT,...}（稳定币/法币/黄金代币）。可知：—",
    "usdkrw_ecb": "USD/KRW = ECB 参考汇率 EUR/KRW ÷ EUR/USD，取 ecb_date = ≤d 的最近一个 ECB 发布日；ECB 在 ecb_date 约 14:00–15:00 UTC 发布（参考时刻 14:15 CET），故在 d+1 00:00 已知；与 Upbit 收盘有 ~9–10 小时（周末/假日更长）时差，属于汇率噪声。可知：ecb_date ~15:00 UTC（≤ d+1 00:00）",
    "ecb_date": "所用 ECB 汇率日期。可知：同上",
    "fx_age_days": "d − ecb_date（0..4，周末/TARGET 假日前值填充）。可知：同上",
    "close_usd_ecb": "close_krw / usdkrw_ecb。可知：" + AV,
    "acc_trade_price_usd_ecb": "acc_trade_price_krw / usdkrw_ecb（美元口径韩国成交额）。可知：" + AV,
    "krw_usdt_close": "同日 Upbit KRW-USDT 收盘价（2024-06-07 起；之前 null）。可知：" + AV,
    "acc_trade_price_usdt_upbit": "acc_trade_price_krw / krw_usdt_close（含韩国 USDT 溢价的 USDT 口径；2024-06-07 前 null）。可知：" + AV,
   }},
  "kimchi.parquet": {
   "key": ["date"], "rows_note": "2021-01-01..2026-09-30 连续 2099 天；所有价格都指同一时点 snapshot_utc = d+1 00:00 UTC",
   "definitions": "prem_btc_vs_krwusdt 是任务给定公式 KRW-BTC/(KRW-USDT×币安BTCUSDT)−1；由于 KRW-USDT 本身就带着韩国溢价，该值≈0（1%–99% 分位 −0.21%..+0.19%），衡量的是韩国站内 BTC 相对 USDT 的偏离，不是传统泡菜溢价。"
                  "传统泡菜溢价 = kimchi_fx_ecb = KRW-BTC/(法币 USDKRW×币安BTCUSDT)−1（1%–99%: −2.3%..+12.1%，中位 +1.5%）。恒等式 (1+kimchi_fx_ecb) = (1+usdt_prem_fx_ecb)×(1+prem_btc_vs_krwusdt)（误差 <1e-15）。"
                  "KRW-USDT 2024-06-07 才上线，之前只能用 kimchi_fx_* 或站内交叉 usdtkrw_implied_upbit_cross。",
   "columns": {
    "date": "UTC 日 d。可知：—",
    "snapshot_utc": "价格时点 = d+1 00:00 UTC。可知：—",
    "available_at": "= snapshot_utc（kimchi_fx_fred 除外，见该列）",
    "krw_btc_close": "Upbit KRW-BTC 日收盘（d+1 00:00 前最后成交）。可知：" + AV,
    "krw_usdt_close": "Upbit KRW-USDT 日收盘（2024-06-07 起）。可知：" + AV,
    "btcusdt_bn_close": "币安现货 BTCUSDT 1h K 线（open_time = d 23:00 UTC）收盘价 = d+1 00:00 价格；与币安 1d 收盘在 2099/2099 天完全相等。可知：" + AV,
    "bn_bar_open_utc": "所用币安 1h K 线开盘时间（全部为 d 23:00）。可知：—",
    "usdkrw_ecb": "见 upbit_daily.usdkrw_ecb。可知：ecb_date ~15:00 UTC",
    "ecb_date": "ECB 汇率日期。", "fx_age_days": "d − ecb_date。",
    "usdkrw_fred": "FRED DEXKOUS（美联储 H.10 纽约中午买入价，12:00 ET≈16–17 UTC，fred_date ≤ d 前值填充）。可知：H.10 每周一发布，实时最多滞后 ~10 天 → 只作交叉校验，不可作实时特征",
    "fred_date": "所用 FRED 日期。",
    "prem_btc_vs_krwusdt": "任务公式 KRW-BTC/(KRW-USDT×btcusdt_bn_close)−1（≈0，见 definitions）；2024-06-07 前 null。可知：" + AV,
    "kimchi_fx_ecb": "传统泡菜溢价 KRW-BTC/(usdkrw_ecb×btcusdt_bn_close)−1；含 ECB 汇率时差噪声（与 FRED 版差异通常 <0.5%）。可知：" + AV,
    "usdt_prem_fx_ecb": "韩国 USDT 溢价 KRW-USDT/usdkrw_ecb−1（1%–99%: −2.2%..+7.8%）；2024-06-07 前 null。可知：" + AV,
    "kimchi_fx_fred": "同 kimchi_fx_ecb 但用 FRED 汇率。可知：滞后（见 usdkrw_fred），仅研究校验",
    "usdt_btc_upbit_close": "Upbit USDT-BTC 市场收盘（USDT/BTC；该市场很薄，2021 年日成交仅数个 BTC）。可知：" + AV,
    "usdtkrw_implied_upbit_cross": "KRW-BTC / USDT-BTC(Upbit) = 站内隐含 KRW/USDT，全期可得，是 2024-06 前 KRW-USDT 的替代（噪声大，看 usdt_btc_upbit_staleness_min）。可知：" + AV,
    "prem_upbit_usdtmkt_vs_bn": "USDT-BTC(Upbit)/btcusdt_bn_close − 1（Upbit USDT 市场相对币安的偏离，薄盘噪声）。可知：" + AV,
    "krw_btc_staleness_min": "KRW-BTC 收盘距 snapshot 的分钟数（<0 见 upbit_daily 说明）。",
    "krw_usdt_staleness_min": "同上，KRW-USDT。", "usdt_btc_upbit_staleness_min": "同上，USDT-BTC（2021–22 常为数十分钟到数小时）。",
    "krw_btc_turnover": "KRW-BTC 日成交额 KRW。可知：" + AV,
    "krw_usdt_turnover": "KRW-USDT 日成交额 KRW（法币↔USDT 出入金通道的直接度量）。可知：" + AV,
    "usdt_btc_upbit_turnover": "USDT-BTC 日成交额（USDT）。可知：" + AV,
    "upbit_krw_turnover_all": "当日所有（存活）KRW 市场成交额合计 KRW。幸存者偏差：越早越低估。可知：" + AV,
    "upbit_krw_turnover_nonstable": "剔除 is_stable 后合计。可知：" + AV,
    "upbit_krw_turnover_alts": "剔除稳定币、BTC、ETH 后合计（山寨热度）。可知：" + AV,
    "upbit_krw_turnover_stables": "稳定币类 KRW 市场合计。可知：" + AV,
    "upbit_turnover_all_usd_ecb": "upbit_krw_turnover_all / usdkrw_ecb。可知：" + AV,
    "upbit_turnover_alts_usd_ecb": "upbit_krw_turnover_alts / usdkrw_ecb。可知：" + AV,
    "n_krw_markets_traded": "当日有成交的存活 KRW 市场数（随存活上币机械上升）。可知：" + AV,
   }},
  "upbit_listings.parquet": {
   "key": ["market"], "rows_note": "294 个存活 KRW 市场；first_bar_date 2017-09-25..2026-10-09（含 10 月新上币）",
   "columns": {
    "market": "KRW-<BASE>。", "base": "币种代码。",
    "first_bar_date": "第一根 KRW 日 K 的 UTC 日期 = 韩国(Upbit)上币日。注意：(a) 16 个市场 = 2017-09-25 即 Upbit 数据起点 → 左删失(listing_left_censored)；(b) Upbit 改名/换币后沿用旧历史，first_bar_date 是前身的上币日，如 KRW-A=EOS(2018-03-22)、KRW-POL=MATIC(2021-10-15)、KRW-BTT(2019-01-31, 1000:1 缩股已回溯)、KRW-TOKAMAK=原 KRW-TON(2020-08-25)、KRW-MANTRA 早期为 OM(价格单位与 OMUSDT 差 ~4 倍)。可知：上币当日",
    "last_bar_date": "最后一根日 K（全部 = 2026-10-09，因都存活）。可知：静态元数据，含未来信息",
    "n_bars": "日 K 根数（全史）。可知：静态元数据，含未来信息",
    "first_day_turnover_krw": "上币首日（部分日）KRW 成交额。可知：first_bar_date+1 00:00",
    "first_day_close_krw": "上币首日收盘价。可知：first_bar_date+1 00:00",
    "days2to8_mean_turnover_krw": "上币第 2–8 天平均成交额。可知：first_bar_date+8d 00:00（作为特征使用时须遵守）",
    "korean_name": "韩文名（快照）。", "english_name": "英文名（快照）。",
    "snap_warning": "market_event.warning（투자유의/投资警告）——仅 2026-10-10 快照，无历史，不能回测使用；若要用需从现在起每日录制",
    "snap_caution_price_fluctuations": "market_event.caution 快照（同上，无历史）。",
    "snap_caution_trading_volume_soaring": "同上（成交量暴增注意）。",
    "snap_caution_deposit_amount_soaring": "同上（入金量暴增注意——最直接的“资金流入”标签，但只有当前快照）。",
    "snap_caution_global_price_differences": "同上（与海外价差注意）。",
    "snap_caution_concentration_of_small_accounts": "同上（少数账户集中注意）。",
    "first_hour_utc": "上币首日第一根有成交的 60 分钟 K 线开始时间（≈ 开放交易时刻；2021 年后上币多在 03:00–11:00 UTC = KST 12–20 点）。可知：该小时",
    "n_traded_hours_first_day": "首日有成交的小时数（首日覆盖比例）。可知：first_bar_date+1 00:00",
    "n_gaps": "日 K 缺口次数（全史）。", "n_gap_days": "缺口总天数。", "max_gap_days": "最长缺口。",
    "gaps_gt7d_resume_dates": ">7 天缺口后恢复日期（仅 KRW-ARDR 2018-01-17 +17d）。",
    "price_jumps_gt5x": "单日 |收盘比|>5 倍的日期（KRW-HBAR 2020-02-12、KRW-BORA 2021-02-18），可能是真实暴涨或单位调整，使用前核对。",
    "listing_left_censored": "first_bar_date = 2017-09-25（数据起点），上币日未知。",
    "is_stable": "稳定币/法币代币。",
    "market_all_snapshot_utc": "market/all 快照时间。",
    "survivor_only": "恒为 True：本表只有存活市场（见顶层 survivorship）。",
    "predecessor_perp_ok": "按改名别名匹配且价格校验通过的前身永续（A←EOSUSDT、BTT←1000BTTCUSDT、POL←MATICUSDT）；非空说明 Upbit 历史含前身。",
   }},
  "upbit_perp_map.parquet": {
   "key": ["market", "perp_symbol"], "rows_note": "每个 (Upbit 市场, 候选币安 U 本位永续) 一行；无候选的市场 perp_symbol=null、status=no_perp_name_match",
   "method_note": "候选来源：(1) name：永续 base 去掉 1000/10000/1000000/1M 前缀后与 Upbit base 相同；(2) alias_rename：已知改名表 BTT→BTTC, POL→MATIC, A→EOS, RENDER→RNDR, KAIA→KLAY, MANTRA→OM, SKY→MKR, TOKAMAK→TON（仅作候选）；"
                  "(3) alias_suffix2：Upbit 'X2' → 'X'。全部候选都用价格校验：在 2021-01-01..2026-09-30 同日都有成交的日子上，计算 lr_rel = log[(close_krw/usdkrw_ecb)/(perp_close/mult)] − log[(KRW-BTC/usdkrw_ecb)/BTCUSDT永续]，"
                  "自动识别 10 的整数次幂单位差(pow10)。status: ok = n≥10 且 |median lr_rel|<0.10 且 ≥80% 天数 |lr_rel|<0.10；price_mismatch = 不同资产同代码（如 TOKAMAK vs TONUSDT=Toncoin、META(Metadium) vs METAUSDT、EDGE、MANTRA vs OMUSDT 单位差 4 倍）；"
                  "no_overlap = 时间不重叠（多为 2026-10 新上币或永续已下架）。注意：status 用了全样本(含 2026-04 之后锁定期)价格，但只是代码同一性 QA，不涉及收益；严格 PIT 可改用 upbit_perp_daily.px_match_10pct（逐日、只用当日价格）。",
   "columns": {
    "base": "Upbit base。", "market": "KRW-<BASE>。",
    "perp_symbol": "币安 USDT 永续（来自 kl1d.parquet，含已下架）。",
    "mult_prefix": "永续代码前缀倍数（1000PEPE→1000，1MBABYDOGE→1e6）。",
    "method": "name / alias_rename / alias_suffix2。",
    "n_overlap_days": "价格校验用到的重叠天数。", "overlap_first": "重叠首日。", "overlap_last": "重叠末日。",
    "med_logprem_rel_btc": "lr_rel 中位数（已扣 pow10）。",
    "frac_days_rel_within_10pct": "|lr_rel|<0.10 的天数占比。",
    "med_logprem_fx": "log(Upbit 美元价/永续价) 中位数（含 BTC 泡菜溢价）。",
    "pow10": "识别出的单位差 10^pow10（BTT/BTTUSDT=−3，STRAX=−1）。",
    "mult_effective": "perp_close / mult_effective = 1 个 Upbit 单位的美元价（BTT/BTTUSDT=1000，STRAX=10，其余 = mult_prefix）。",
    "status": "ok / price_mismatch / no_overlap / no_perp_name_match。",
    "english_name": "Upbit 英文名（快照）。", "first_bar_date": "Upbit 上币日。",
   }},
  "upbit_perp_daily.parquet": {
   "key": ["date", "market", "perp_symbol"], "rows_note": "仅 status=ok 的映射；2021-01-01..2026-09-30；每个 (date, market) 至多 1 个永续（改名前后时间不重叠）",
   "columns": {
    "date": "UTC 日 d。", "market": "KRW-<BASE>。", "perp_symbol": "币安永续。",
    "map_status": "ok。", "mult_effective": "见 map。",
    "close_krw": "Upbit 收盘。可知：" + AV, "close_staleness_min": "Upbit 收盘陈旧度（分钟）。",
    "usdkrw_ecb": "ECB USD/KRW（见上）。",
    "perp_close": "币安永续 1d 收盘（kl1d，= d+1 00:00 价格，仅 quote_volume>0 的日子）。可知：" + AV,
    "prem_fx_ecb": "单币泡菜溢价 close_krw/(usdkrw_ecb×perp_close/mult_effective) − 1（含永续基差与汇率时差）。可知：" + AV,
    "prem_rel_btc": "相对 BTC 的单币溢价 (1+prem_fx_ecb)/(1+BTC 对 BTCUSDT 永续的溢价) − 1，扣掉了全市场溢价与汇率误差（中位 +0.06%，1%–99%: −0.65%..+2.1%）。可知：" + AV,
    "px_match_10pct": "|log(1+prem_rel_btc)|<0.10（逐日 PIT 代码一致性标志）。False 的 0.3% 行多为 Upbit 充提暂停导致的持续高溢价（如 KRW-KAVA 2022-12..2023-05 +40–55%、IOTA、GMT），不可套利，不等于资金流入。",
    "upbit_turnover_usd_ecb": "Upbit 当日成交额（美元）。可知：" + AV,
    "perp_quote_volume": "币安永续当日成交额 USDT。可知：" + AV,
    "upbit_to_perp_turnover": "upbit_turnover_usd_ecb / perp_quote_volume（韩国成交相对币安永续的倍数）。可知：" + AV,
    "available_at": "d+1 00:00 UTC。",
   }},
 },
 "not_done": "未计算任何信号与未来收益的统计；未构建 fwd 列（需要时与 perp_daily.parquet 的 fwd_* 按 (date, perp_symbol) 连接）。",
}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
