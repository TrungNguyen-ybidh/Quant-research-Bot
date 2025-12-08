# Phase 1: Preprocessing Pipeline Report

**Generated:** 2025-12-07 17:47:30

---

## Summary

| Metric | Value |
|--------|-------|
| Total Files Processed | 55 |
| Total Samples | 333,897 |
| Symbols | 11 |
| Timeframes | 5 |

### Symbols
AUDJPY, AUDUSD, CHFJPY, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY

### Timeframes
15m, 1d, 1h, 1m, 5m

---

## Top Predictive Features (Across All Assets)

| Rank | Feature | Frequency |
|------|---------|-----------|
| 1 | pivot_point | 53/55 |
| 2 | realized_vol_60m | 28/55 |
| 3 | hurst_exponent | 27/55 |
| 4 | skew | 24/55 |
| 5 | macd_signal | 22/55 |
| 6 | obv | 20/55 |
| 7 | adx | 19/55 |
| 8 | yield_curve_factor | 14/55 |
| 9 | value_factor | 9/55 |
| 10 | volatility_of_vol | 8/55 |

---

## Per-Asset Results

| Symbol | Timeframe | Samples | Features | Train | Val | Test |
|--------|-----------|---------|----------|-------|-----|------|
| AUDJPY | 15m | 7,629 | 30 | 5,340 | 1,144 | 1,145 |
| AUDJPY | 1d | 2,570 | 30 | 1,798 | 386 | 386 |
| AUDJPY | 1h | 8,398 | 30 | 5,878 | 1,260 | 1,260 |
| AUDJPY | 1m | 7,097 | 30 | 4,967 | 1,065 | 1,065 |
| AUDJPY | 5m | 7,160 | 30 | 5,012 | 1,074 | 1,074 |
| AUDUSD | 15m | 7,562 | 30 | 5,293 | 1,134 | 1,135 |
| AUDUSD | 1d | 2,569 | 30 | 1,798 | 385 | 386 |
| AUDUSD | 1h | 8,314 | 30 | 5,819 | 1,247 | 1,248 |
| AUDUSD | 1m | 6,549 | 30 | 4,584 | 982 | 983 |
| AUDUSD | 5m | 7,041 | 30 | 4,928 | 1,056 | 1,057 |
| CHFJPY | 15m | 7,375 | 30 | 5,162 | 1,106 | 1,107 |
| CHFJPY | 1d | 2,569 | 30 | 1,798 | 385 | 386 |
| CHFJPY | 1h | 8,291 | 30 | 5,803 | 1,244 | 1,244 |
| CHFJPY | 1m | 5,506 | 30 | 3,854 | 826 | 826 |
| CHFJPY | 5m | 6,578 | 30 | 4,604 | 987 | 987 |
| EURJPY | 15m | 7,319 | 30 | 5,123 | 1,098 | 1,098 |
| EURJPY | 1d | 2,559 | 30 | 1,791 | 384 | 384 |
| EURJPY | 1h | 8,343 | 30 | 5,840 | 1,251 | 1,252 |
| EURJPY | 1m | 5,382 | 30 | 3,767 | 807 | 808 |
| EURJPY | 5m | 6,478 | 30 | 4,534 | 972 | 972 |
| EURUSD | 15m | 7,301 | 30 | 5,110 | 1,095 | 1,096 |
| EURUSD | 1d | 2,566 | 30 | 1,796 | 385 | 385 |
| EURUSD | 1h | 8,140 | 30 | 5,698 | 1,221 | 1,221 |
| EURUSD | 1m | 4,701 | 30 | 3,290 | 705 | 706 |
| EURUSD | 5m | 6,115 | 30 | 4,280 | 917 | 918 |
| GBPJPY | 15m | 7,496 | 30 | 5,247 | 1,124 | 1,125 |
| GBPJPY | 1d | 2,556 | 30 | 1,789 | 383 | 384 |
| GBPJPY | 1h | 8,320 | 30 | 5,824 | 1,248 | 1,248 |
| GBPJPY | 1m | 6,082 | 30 | 4,257 | 912 | 913 |
| GBPJPY | 5m | 6,819 | 30 | 4,773 | 1,023 | 1,023 |
| GBPUSD | 15m | 7,275 | 30 | 5,092 | 1,091 | 1,092 |
| GBPUSD | 1d | 2,569 | 30 | 1,798 | 385 | 386 |
| GBPUSD | 1h | 8,150 | 30 | 5,705 | 1,222 | 1,223 |
| GBPUSD | 1m | 5,224 | 30 | 3,656 | 784 | 784 |
| GBPUSD | 5m | 6,488 | 30 | 4,541 | 973 | 974 |
| NZDUSD | 15m | 7,596 | 30 | 5,317 | 1,139 | 1,140 |
| NZDUSD | 1d | 2,562 | 30 | 1,793 | 384 | 385 |
| NZDUSD | 1h | 8,312 | 30 | 5,818 | 1,247 | 1,247 |
| NZDUSD | 1m | 5,691 | 30 | 3,983 | 854 | 854 |
| NZDUSD | 5m | 6,821 | 30 | 4,774 | 1,023 | 1,024 |
| USDCAD | 15m | 6,803 | 30 | 4,762 | 1,020 | 1,021 |
| USDCAD | 1d | 2,561 | 30 | 1,792 | 384 | 385 |
| USDCAD | 1h | 7,996 | 30 | 5,597 | 1,199 | 1,200 |
| USDCAD | 1m | 3,460 | 30 | 2,422 | 519 | 519 |
| USDCAD | 5m | 5,492 | 30 | 3,844 | 824 | 824 |
| USDCHF | 15m | 7,471 | 30 | 5,229 | 1,121 | 1,121 |
| USDCHF | 1d | 2,572 | 30 | 1,800 | 386 | 386 |
| USDCHF | 1h | 8,205 | 30 | 5,743 | 1,231 | 1,231 |
| USDCHF | 1m | 5,620 | 30 | 3,933 | 844 | 843 |
| USDCHF | 5m | 6,560 | 30 | 4,592 | 984 | 984 |
| USDJPY | 15m | 7,483 | 30 | 5,238 | 1,122 | 1,123 |
| USDJPY | 1d | 2,567 | 30 | 1,796 | 385 | 386 |
| USDJPY | 1h | 8,366 | 30 | 5,856 | 1,255 | 1,255 |
| USDJPY | 1m | 5,834 | 30 | 4,083 | 875 | 876 |
| USDJPY | 5m | 6,834 | 30 | 4,783 | 1,025 | 1,026 |

---

## Saved Artifacts

### Scalers
```
models/scalers/{SYMBOL}/{TIMEFRAME}_scaler.pkl
```

### Selectors
```
models/selectors/{SYMBOL}/{TIMEFRAME}_selector.pkl
```

---

## Next Steps

- **Phase 2:** Regime Analysis (HMM)
- **Phase 3:** Strategy Testing
- **Phase 4:** Research Reports per Asset

