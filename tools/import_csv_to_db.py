"""CSVの予測結果・レース結果をSQLite DBにインポートするスクリプト"""
import sqlite3
import pandas as pd
import json
import os
import sys
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "keiba_race_results.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT, race_name TEXT, race_date TEXT,
        course TEXT, distance INTEGER, surface TEXT, condition TEXT,
        horse_name TEXT, horse_num INTEGER, ai_rank INTEGER,
        ai_score REAL, odds REAL, predicted_at TEXT,
        actual_finish INTEGER DEFAULT NULL,
        is_top3_pred INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS race_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT UNIQUE, race_name TEXT,
        predicted_at TEXT, result_updated_at TEXT DEFAULT NULL,
        num_horses INTEGER, top1_name TEXT, top1_score REAL,
        trio_bets TEXT DEFAULT NULL,
        hit_trio INTEGER DEFAULT NULL,
        hit_combo TEXT DEFAULT NULL,
        payout INTEGER DEFAULT 0
    )""")
    for col_sql in [
        "ALTER TABLE race_results ADD COLUMN trio_bets TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN hit_trio INTEGER DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN hit_combo TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN payout INTEGER DEFAULT 0",
        "ALTER TABLE race_results ADD COLUMN is_nar INTEGER DEFAULT 0",
        "ALTER TABLE race_results ADD COLUMN wide_bets TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN hit_wide INTEGER DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN wide_payout INTEGER DEFAULT 0",
        "ALTER TABLE race_results ADD COLUMN buy_recommended INTEGER DEFAULT 1",
        "ALTER TABLE race_results ADD COLUMN bet_condition TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN bet_type TEXT DEFAULT NULL",
        "ALTER TABLE race_results ADD COLUMN umaren_bets TEXT DEFAULT NULL",
    ]:
        try:
            c.execute(col_sql)
        except:
            pass
    conn.commit()
    conn.close()


def parse_trio_bets(bets_str):
    """'2-4-5; 2-5-6; ...' → [[2,4,5], [2,5,6], ...]"""
    if not bets_str or pd.isna(bets_str):
        return []
    bets = []
    for part in bets_str.split("; "):
        nums = [int(n) for n in part.strip().split("-") if n.strip().isdigit()]
        if len(nums) >= 2:
            bets.append(sorted(nums))
    return bets


def generate_umaren_bets_from_tops(top1, top2, top3):
    """馬連2点: TOP1-TOP2, TOP1-TOP3"""
    return [sorted([top1, top2]), sorted([top1, top3])]


def import_date(date_str):
    """指定日のCSVデータをDBにインポート"""
    pred_path = os.path.join(BASE_DIR, "data", "daily_predictions", f"{date_str}.csv")
    result_path = os.path.join(BASE_DIR, "data", "daily_results", f"{date_str}.csv")

    if not os.path.exists(pred_path):
        print(f"  予測CSV未発見: {pred_path}")
        return

    df_pred = pd.read_csv(pred_path, encoding='utf-8-sig')
    print(f"  予測: {len(df_pred)}レース")

    # 結果CSV読み込み
    df_result = None
    if os.path.exists(result_path):
        df_result = pd.read_csv(result_path, encoding='utf-8-sig')
        print(f"  結果: {len(df_result)}レース")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    race_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    for _, row in df_pred.iterrows():
        race_id = str(row['race_id'])
        course = str(row.get('course', ''))
        race_name = str(row.get('race_name', 'レース'))
        race_num = int(row.get('race_num', 0))
        condition = str(row.get('condition', ''))
        num_horses = int(row.get('num_horses', 0))
        distance = int(row.get('distance', 0))
        surface = str(row.get('surface', ''))
        track_condition = str(row.get('track_condition', ''))
        top1_num = int(row.get('top1_num', 0))
        top1_name = str(row.get('top1_name', ''))
        top1_score = float(row.get('top1_score', 0))
        top2_num = int(row.get('top2_num', 0))
        top3_num = int(row.get('top3_num', 0))
        trio_bets_str = str(row.get('trio_bets', ''))
        bet_type = str(row.get('bet_type', 'trio'))

        # 既存データ削除
        c.execute("DELETE FROM predictions WHERE race_id = ?", (race_id,))
        c.execute("DELETE FROM race_results WHERE race_id = ?", (race_id,))

        # predictions テーブル: TOP3馬のみ挿入（簡易版）
        predicted_at = f"{race_date} 08:00:00"
        for rank, (hnum, hname) in enumerate([(top1_num, top1_name),
                                                (top2_num, str(row.get('top2_name', ''))),
                                                (top3_num, str(row.get('top3_name', '')))], 1):
            score = top1_score if rank == 1 else top1_score * 0.9
            c.execute("""INSERT INTO predictions
                (race_id, race_name, race_date, course, distance, surface, condition,
                 horse_name, horse_num, ai_rank, ai_score, odds, predicted_at, is_top3_pred)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (race_id, f"{course}{race_num}R", race_date, course, distance, surface,
                 track_condition, hname, hnum, rank, score, 0.0, predicted_at, 1))

        # 買い目生成
        trio_bets = parse_trio_bets(trio_bets_str)
        umaren_bets = generate_umaren_bets_from_tops(top1_num, top2_num, top3_num)

        # race_results テーブル
        c.execute("""INSERT INTO race_results
            (race_id, race_name, predicted_at, num_horses, top1_name, top1_score,
             trio_bets, wide_bets, umaren_bets, is_nar, buy_recommended, bet_condition, bet_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (race_id, f"{course}{race_num}R", predicted_at, num_horses, top1_name, top1_score,
             json.dumps(trio_bets), json.dumps([]), json.dumps(umaren_bets),
             0, 1, condition, bet_type))

        # 結果がある場合は的中判定
        if df_result is not None:
            res_row = df_result[df_result['race_id'].astype(str) == race_id]
            if len(res_row) > 0:
                res = res_row.iloc[0]
                trio_result_str = str(res.get('trio_result', ''))
                trio_hit = int(res.get('trio_hit', 0))
                actual_payout = int(res.get('actual_payout', 0))
                status = str(res.get('status', ''))

                if status == 'settled' and trio_result_str:
                    # 実際の着順結果から finish_order を再構築
                    top1_finish = res.get('top1_finish', '-')
                    top2_finish = res.get('top2_finish', '-')
                    top3_finish = res.get('top3_finish', '-')

                    # predictionsに着順を記録
                    for hnum, finish in [(top1_num, top1_finish), (top2_num, top2_finish), (top3_num, top3_finish)]:
                        if str(finish).isdigit():
                            c.execute("UPDATE predictions SET actual_finish = ? WHERE race_id = ? AND horse_num = ?",
                                      (int(finish), race_id, hnum))

                    # 的中判定結果を反映
                    hit_combo = None
                    if trio_hit == 1:
                        # 的中した買い目を特定
                        trio_result_nums = set(int(n) for n in trio_result_str.split('-'))
                        for bet in trio_bets:
                            if set(bet) == trio_result_nums:
                                hit_combo = json.dumps(sorted(bet))
                                break
                        # umaren的中チェック
                        if not hit_combo and bet_type == 'umaren':
                            umaren_payout = int(res.get('umaren_payout', 0))
                            if umaren_payout > 0:
                                actual_payout = umaren_payout

                    now = f"{race_date} 20:00:00"
                    c.execute("""UPDATE race_results SET result_updated_at=?, hit_trio=?,
                                 hit_combo=?, payout=? WHERE race_id=?""",
                              (now, trio_hit, hit_combo, actual_payout, race_id))

    conn.commit()
    conn.close()


def main():
    print("=" * 60)
    print("  CSV → SQLite DB インポート")
    print("=" * 60)

    init_db()
    print("\nDBテーブル初期化完了")

    # 日付リスト取得
    pred_dir = os.path.join(BASE_DIR, "data", "daily_predictions")
    if not os.path.exists(pred_dir):
        print("daily_predictionsディレクトリが見つかりません")
        return

    dates = []
    for f in sorted(os.listdir(pred_dir)):
        if f.endswith('.csv') and len(f) == 12:  # 20260314.csv
            dates.append(f.replace('.csv', ''))

    if not dates:
        print("インポート対象なし")
        return

    print(f"\n対象日: {', '.join(dates)}")
    for date_str in dates:
        print(f"\n--- {date_str} ---")
        import_date(date_str)

    # 結果確認
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM race_results")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM race_results WHERE hit_trio = 1")
    hits = c.fetchone()[0]
    c.execute("SELECT SUM(payout) FROM race_results WHERE hit_trio = 1")
    total_payout = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM race_results WHERE hit_trio IS NOT NULL")
    settled = c.fetchone()[0]
    conn.close()

    investment = settled * 700
    roi = total_payout / investment * 100 if investment > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  インポート完了")
    print(f"{'=' * 60}")
    print(f"  レース数: {total}")
    print(f"  確定: {settled}")
    print(f"  的中: {hits}")
    print(f"  投資: {investment:,}円")
    print(f"  払戻: {total_payout:,}円")
    print(f"  ROI: {roi:.1f}%")


if __name__ == '__main__':
    main()
