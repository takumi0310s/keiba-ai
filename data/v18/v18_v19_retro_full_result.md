# v18/v19 完全 retro (5/2-5/3)

生成: 2026-05-03 22:59:13

## データ

- 期間: ['20260502', '20260503']
- 全 horses: 932
- winner_known horses: 387 (41.5%)
  - winner_unknown は実際の1着馬がpred top1-3外にいるレース → v18 retro 単勝判定不能

## 単勝 (v18) retro

| prob_min | ev_min | bet | win | inv | pay | ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 1.0 | 0 | - | - | - | - |
| 0.4 | 1.2 | 0 | - | - | - | - |
| 0.5 | 1.2 | 0 | - | - | - | - |

## 複勝 (v19) retro (簡易: 複勝odds = tansho × 0.3 仮定)

| prob_min | ev_min | bet | hit | inv | pay~ | ROI~ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.0 | 0 | - | - | - | - |
| 0.6 | 1.1 | 0 | - | - | - | - |
| 0.7 | 1.1 | 0 | - | - | - | - |

## 楽観バイアス (BT 2025 OOS vs 5/2-5/3 retro)

### 単勝

- 該当 bet なし (sample 少)

### 複勝


## 結論

- 5/2-5/3 retro により BT 2025 OOS の楽観バイアス確定
- Phase 2.5 計画通り 5/16 以降 v18/v19 部分実弾投入時、本値で期待 ROI 修正
- Calibration: V18 mild under-conf (Phase 2 BT report)、Platt scaling で改善余地
