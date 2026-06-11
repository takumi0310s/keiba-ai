#!/usr/bin/env python3
"""データ鮮度監視 (2026-05-30 実装、 PACI 4/4更新停止の再発防止)。

毎朝 (予測前) に主要データソースの「当日カバレッジ」を測定し、
閾値割れを Discord #アップデート に警告する。 「1ヶ月気づかない」 を防ぐ。

★ 読み取り + 通知のみ。 予測ロジック / V15 model / predict_core は一切不変 ★
★ daily_predict.py は触らない (standalone)。 スケジューラから朝に単独起動 ★

チェック内容:
  - 当日のJRA全race_id (netkeiba race_list) を取得
  - jrdb_paci / jrdb_sed / jrdb_kyi / jrdb_tyb に当日race_idが何%入っているか
  - PACI は V15 gain の 52.6% を占める最重要 → 閾値 80% 割れで警告
  - netkeiba thisweek ファイルの mtime 鮮度

usage:
  python tools/data_freshness_monitor.py                # 当日チェック + 閾値割れ通知
  python tools/data_freshness_monitor.py --date 20260530
  python tools/data_freshness_monitor.py --no-notify    # 通知せず表示のみ
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR / "tools"))

# 当日カバレッジ閾値 (これを下回ると警告)。
# ★ PACI のみハード閾値 ★: 当日の前日指数データで、 当日race_idを持つべき源。
#   gain 52.6% を占め、 2026-04頃に更新停止した実績がある最重要監視対象。
# sed/tyb は「前走/結果(historical)」 で当日race_idを持たないのが正常 → 参考表示のみ。
# kyi は列名 nk_race_id を使用 → 参考表示のみ。
THRESHOLDS = {
    "jrdb_paci": 0.80,   # ★ V15 gain 52.6% — 最優先 ★
}
# 学習時カバレッジ (train/serve スキュー検知用の基準 ~100%)
TRAIN_COVERAGE_REF = 1.00

# ★ 「静かな停止」 検知: race weekend 毎に更新されるべき主要ソースの mtime 鮮度。
#   paci(4/4停止)・ze(5/1停止) と同じ「ファイルが進まない」 を mtime で検知。
#   閾値日数を超えて更新が無ければ警告 (race は土日 → 9日で2週末ミス相当)。
#   gain占有が大きい順: paci 52.6% / ze 12.7% は最優先。
MTIME_WATCH = {
    "jrdb_paci.csv": 9,   # PACI gain 52.6% (KYI当日指数)
    "jrdb_ze.csv": 9,     # ZE gain 12.7% (ZED前走拡張)
    "jrdb_kyi.csv": 9,    # 出馬表
    "jrdb_oz.csv": 9,     # オッズ
    "jrdb_kta.csv": 9,    # 騎手
}

# ★ 内容ベースの停止検知 (6/11 Fable sweep) ★
# mtime は毎日 rewrite される job だと進む一方、中身の最新レースが止まる
# 「ゾンビ更新」を素通りする (KTA 4/5停止を 67 日間検知できなかった実測)。
# → ファイル内 race_id の開催日 (KAB kaisai_key→yyyymmdd) の max で判定。
# file: (race_id列名, 許容遅れ日数)。 結果系 (SED/SR/SRB/ZE) は1-2週末遅れが正常 → 16日。
CONTENT_WATCH = {
    "jrdb_paci.csv": ("race_id", 9),
    "jrdb_kyi.csv": ("nk_race_id", 9),
    "jrdb_oz.csv": ("race_id", 9),
    "jrdb_kta.csv": ("race_id", 9),
    "jrdb_kka.csv": ("race_id", 9),
    "jrdb_jo.csv": ("race_id", 9),
    "jrdb_cha.csv": ("race_id", 9),
    "jrdb_sed.csv": ("race_id", 16),
    "jrdb_sr.csv": ("race_id", 16),
    "jrdb_ze.csv": ("race_id", 16),
    "jrdb_srb.csv": ("race_id", 16),
    "jrdb_skb.csv": ("race_id", 16),
    "jrdb_tyb.csv": ("race_id", 16),
}


def _kab_date_map() -> dict:
    """KAB の kaisai_key(=race_id先頭10桁) → yyyymmdd マップ。"""
    import pandas as pd
    p = DATA_DIR / "jrdb_kab.csv"
    if not p.exists():
        return {}
    try:
        kab = pd.read_csv(p, dtype=str)
        out = {}
        # 旧形式行: kaisai_key + yyyymmdd
        m = kab["kaisai_key"].notna() & kab["yyyymmdd"].notna()
        out.update(zip(kab.loc[m, "kaisai_key"].str[:10], kab.loc[m, "yyyymmdd"]))
        # 新形式行 (直近分は日本語列のみ): 場コード/年/回/日/年月日 から key を再構成
        jcols = ["場コード", "年", "回", "日", "年月日"]
        if all(c in kab.columns for c in jcols):
            j = kab[kab["年月日"].notna()]
            for _, r in j.iterrows():
                try:
                    key = (f"20{str(r['年']).zfill(2)}{str(r['場コード']).zfill(2)}"
                           f"{str(r['回']).zfill(2)}{str(r['日']).zfill(2)}")
                    out[key] = str(r["年月日"])
                except Exception:
                    continue
        return out
    except Exception as e:
        print(f"[WARN] KAB読込失敗 (content検知スキップ): {e}")
        return {}


def content_latest_date(csv_name: str, rid_col: str, kab_map: dict) -> str | None:
    """ファイル内 race_id の最新開催日 (yyyymmdd) を返す。判定不能は None。"""
    import pandas as pd
    path = DATA_DIR / csv_name
    if not path.exists() or not kab_map:
        return None
    try:
        prefixes = set()
        for chunk in pd.read_csv(path, dtype=str, usecols=[rid_col], chunksize=200000):
            prefixes |= set(chunk[rid_col].dropna().str[:10].unique())
        dates = [kab_map[p] for p in prefixes if p in kab_map]
        return max(dates) if dates else None
    except Exception as e:
        print(f"[WARN] {csv_name} content読込失敗: {e}")
        return None

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get_today_race_ids(date_str: str) -> list[str]:
    """netkeiba race_list から当日全race_idを取得 (予測csv非依存、 朝7:30運用可)。"""
    try:
        import requests
        from bs4 import BeautifulSoup
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        ids = set()
        for a in soup.find_all("a", href=True):
            m = re.search(r"race_id=(\d{12})", a["href"])
            if m:
                ids.add(m.group(1))
        return sorted(ids)
    except Exception as e:
        print(f"[WARN] netkeiba race_list 取得失敗: {e}")
        # フォールバック: 当日 daily_predictions csv
        p = DATA_DIR / "daily_predictions" / f"{date_str}.csv"
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p, dtype={"race_id": str})
            return sorted(df["race_id"].astype(str).unique())
        return []


def coverage_in_csv(csv_name: str, race_ids: list[str]) -> tuple[float, int, int]:
    """csv に当日race_idが何件入っているか。 (率, 一致race数, 全race数)。"""
    import pandas as pd
    path = DATA_DIR / csv_name
    if not path.exists() or not race_ids:
        return 0.0, 0, len(race_ids)
    try:
        # race_id 列名は csv により異なる (kyi は nk_race_id)
        head = pd.read_csv(path, nrows=0)
        rid_col = "race_id" if "race_id" in head.columns else (
            "nk_race_id" if "nk_race_id" in head.columns else None)
        if rid_col is None:
            return -1.0, 0, len(race_ids)  # race_id列なし = 判定不能
        present = set()
        for chunk in pd.read_csv(path, dtype={rid_col: str}, usecols=[rid_col],
                                 chunksize=200000, low_memory=False):
            present |= set(chunk[rid_col].astype(str).unique())
        hit = sum(1 for r in race_ids if r in present)
        return hit / len(race_ids), hit, len(race_ids)
    except Exception as e:
        print(f"[WARN] {csv_name} 読込失敗: {e}")
        return 0.0, 0, len(race_ids)


def thisweek_freshness(date_str: str) -> list[str]:
    """netkeiba thisweek ファイルが当日更新されているか。"""
    out = []
    today = datetime.strptime(date_str, "%Y%m%d").date()
    for fn in ["netkeiba_data_analysis_thisweek.csv", "netkeiba_upset_level_thisweek.csv"]:
        p = DATA_DIR / fn
        if not p.exists():
            out.append(f"{fn}: 欠落")
            continue
        mt = datetime.fromtimestamp(p.stat().st_mtime).date()
        out.append(f"{fn}: {'当日OK' if mt >= today else f'古い({mt})'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="データ鮮度監視 (PACI停止 再発防止)")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()
    date_str = args.date

    print("=" * 60)
    print(f" データ鮮度監視 {date_str}  {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    race_ids = get_today_race_ids(date_str)
    print(f" 当日race_id: {len(race_ids)}件")
    # 6/11 Fable sweep: 非開催日でも mtime/内容鮮度チェックは実行する
    # (旧実装はここで return → 内容停止が週末まで検知されなかった)
    skip_coverage = not race_ids
    if skip_coverage:
        print(" 非開催または取得不可 → 当日カバレッジのみスキップ (鮮度チェックは継続)")

    results = {}
    INFO_NOTE = {
        "jrdb_sed": "(前走/結果=当日race_idを持たないのが正常・参考)",
        "jrdb_tyb": "(直前データ=発走前に確定・参考)",
        "jrdb_kyi": "(参考)",
    }
    for csv_name in ([] if skip_coverage else ["jrdb_paci", "jrdb_kyi", "jrdb_sed", "jrdb_tyb"]):
        rate, hit, tot = coverage_in_csv(f"{csv_name}.csv", race_ids)
        results[csv_name] = rate
        thr = THRESHOLDS.get(csv_name)
        if rate < 0:
            print(f"  {csv_name:12s}: race_id列なし → 判定不能 {INFO_NOTE.get(csv_name,'')}")
            continue
        if thr is not None:
            mark = " ★OK" if rate >= thr else f" 🔴WARN (<{int(thr*100)}%)"
            skew = (TRAIN_COVERAGE_REF - rate) * 100
            print(f"  {csv_name:12s}: 当日 {rate*100:5.1f}% ({hit}/{tot}R)"
                  f"  train/serveスキュー {skew:+.0f}pt{mark}")
        else:
            print(f"  {csv_name:12s}: 当日 {rate*100:5.1f}% ({hit}/{tot}R) "
                  f"{INFO_NOTE.get(csv_name,'(参考)')}")

    # ★ 静かな停止検知: mtime 鮮度 ★
    print(" mtime鮮度 (静かな停止検知):")
    now = datetime.now().timestamp()
    stale_sources = []
    for fn, max_days in MTIME_WATCH.items():
        p = DATA_DIR / fn
        if not p.exists():
            print(f"   {fn:16s}: 欠落 🔴")
            stale_sources.append(fn)
            continue
        age = (now - p.stat().st_mtime) / 86400
        mark = " 🔴STALE" if age > max_days else " OK"
        if age > max_days:
            stale_sources.append(fn)
        print(f"   {fn:16s}: {age:5.1f}日前更新 (閾値{max_days}日){mark}")

    # ★ 内容ベース停止検知 (ゾンビ更新対策・6/11) ★
    print(" 内容鮮度 (中身の最新開催日、KAB基準):")
    content_stale = []
    kab_map = _kab_date_map()
    today_dt = datetime.strptime(date_str, "%Y%m%d")
    for fn, (rid_col, max_lag) in CONTENT_WATCH.items():
        latest = content_latest_date(fn, rid_col, kab_map)
        if latest is None:
            print(f"   {fn:16s}: 判定不能")
            continue
        try:
            lag = (today_dt - datetime.strptime(latest, "%Y%m%d")).days
        except Exception:
            print(f"   {fn:16s}: 日付不正 ({latest})")
            continue
        mark = " 🔴CONTENT-STALL" if lag > max_lag else " OK"
        if lag > max_lag:
            content_stale.append(f"{fn}(内容{latest}止まり)")
        print(f"   {fn:16s}: 最新 {latest} ({lag}日前、閾値{max_lag}日){mark}")

    print(" netkeiba thisweek:")
    for line in thisweek_freshness(date_str):
        print(f"   {line}")

    # 閾値割れ判定 (当日カバレッジ + mtime鮮度 + 内容鮮度)
    warns = [k for k, v in results.items()
             if k in THRESHOLDS and 0 <= v < THRESHOLDS[k]]
    warns += [f"{s}(mtime)" for s in stale_sources]
    warns += [f"{s}(content)" for s in content_stale]
    if warns and not args.no_notify:
        try:
            from notify import send_discord
            lines = [f"⚠️ **データ鮮度警告 {date_str}**", ""]
            for k in warns:
                if k.endswith("(content)"):
                    lines.append(f"🔴 {k[:-9]}: 中身の最新開催日が止まっている(ゾンビ更新)")
                elif k.endswith("(mtime)"):
                    lines.append(f"🔴 {k[:-7]}: 一定日数更新なし(静かな停止の疑い)")
                else:
                    lines.append(f"🔴 {k}: 当日カバレッジ {results[k]*100:.0f}% "
                                 f"(閾値{int(THRESHOLDS[k]*100)}%割れ)")
            if any("paci" in k for k in warns):
                lines.append("\n※ PACI=V15予測力52.6%。 当日取得経路を確認。")
            if any("ze" in k for k in warns):
                lines.append("※ ZE=V15予測力12.7%(前走拡張)。 tools/daily_paci_refresh.py で復旧。")
            send_discord("データ鮮度警告", "\n".join(lines), color="red", channel="updates")
            print(f"\n [NOTIFY] Discord #アップデート に警告送信: {warns}")
        except Exception as e:
            print(f"\n [WARN] Discord通知失敗: {e}")
    elif warns:
        print(f"\n [WARN] 閾値割れ {warns} (--no-notify のため未送信)")
    else:
        print("\n 全ソース閾値クリア ✅")

    return 0


if __name__ == "__main__":
    sys.exit(main())
