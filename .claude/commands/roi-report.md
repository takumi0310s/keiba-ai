---
description: 最新ROI統計を集計してレポート
---

`data/cumulative_results.csv` から条件別/月別/競馬場別 ROI を集計、Discord通知。

```bash
python tools/roi_analysis.py
```

出力内容:
- 全体サマリー (N, 投資, 払戻, 損益, ROI)
- 条件別 (A-X) — N+ROI+的中率
- 月別推移
- 競馬場別 (10場)
- 直近60Rの条件別
- 大勝レース TOP10 (払戻5,000円超)
- 連敗 (5R以上)
- 土曜本番への示唆 (おすすめ条件 / 警戒条件)

レポート: `report/roi_analysis_20260423.md`
Discord: #updates チャンネルにサマリー送信

`--no-discord` で送信スキップ可能。
