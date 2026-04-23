---
description: 事故ナレッジベース検索
---

`data/incident_knowledge_base.json` から類似事故を検索。

```bash
python tools/knowledge_base_query.py --query "症状やキーワード"
```

または直接 grep:

```bash
python -c "
import json
d = json.load(open('data/incident_knowledge_base.json', 'r', encoding='utf-8'))
q = 'KEYWORD'
for i in d['incidents']:
    if q in (i.get('symptom','') + i.get('root_cause','')):
        print(i['incident_id'], i['symptom'][:80])
"
```

登録済み事故 (4/23時点 17件):
- inc-20260419-001〜009: 4/19 SCRAPER-GUARD 連鎖事故
- inc-20260422-001/002: CatBoost race_id_unique / スクレイピング停止
- inc-20260423-001〜006: SED merge / TARGET 不在 / cookie / fire_check / test 既存バグ

各 incident に detection_rule + auto_recovery を含む。
