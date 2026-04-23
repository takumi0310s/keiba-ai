---
description: 本番反映前の状態チェックを一括実行
---

`tools/deploy_check.py` を実行して本番準備状態を確認する。

```bash
python tools/deploy_check.py --no-discord
```

チェック項目:
- pytest 動作確認
- Keiba タスクスケジューラ Ready 確認 (10本)
- cookies 状態
- ディスク空き容量
- v15 モデルファイル存在確認
- JRDB データ鮮度 (5日以上古いと警告)
- app.py / predict_core.py 構文チェック
- 直近予測ファイル状況

出力: `report/deploy_check_20260423.md` + 標準出力
判定: 🟢 OK / 🟡 警告 / 🔴 Critical
