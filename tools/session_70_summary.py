"""Session #70 D: 5月 11R/12R 重賞除外 統計サマリ.

source: data/v18/session_70_filtered_races.csv のみ。
LEAK 防止: model 不使用、 csv aggregate のみ。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def class_label(race_name: str) -> str:
    s = str(race_name)
    if "1勝" in s: return "1勝"
    if "2勝" in s: return "2勝"
    if "3勝" in s: return "3勝"
    if "未勝利" in s or "新馬" in s: return "未勝利/新馬"
    if "(L)" in s: return "L"
    return "OP/特別"


def main():
    df = pd.read_csv(BASE / "data" / "v18" / "session_70_filtered_races.csv",
                     encoding="utf-8-sig")
    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False)
    df["class"] = df["race_name"].apply(class_label)

    out = []
    out.append("# 5月 11R/12R (重賞除外) 統計サマリ (Session #70 D)")
    out.append("")
    out.append("**source**: production_saved_score (リーク完全防止)")
    out.append("")
    out.append("---")
    out.append("")

    # 全体
    out.append("## 1. 全体")
    out.append("")
    out.append(f"- 対象 R: **{len(df)} 件**")
    nh = pd.to_numeric(df["num_horses"], errors="coerce")
    if nh.notna().any():
        out.append(f"- 平均出走頭数: {nh.mean():.1f} 頭 (data 有 R のみ)")
    out.append(f"- 期間: {df['date'].min()} 〜 {df['date'].max()}")
    out.append(f"- 日別内訳: {dict(df['date'].value_counts().sort_index())}")
    out.append("")

    # V15 score 分布 (5/9 のみ score 値あり)
    out.append("## 2. V15 production score 分布 (top1_score)")
    out.append("")
    s = pd.to_numeric(df["top1_score"], errors="coerce").dropna()
    if len(s) > 0:
        out.append(f"- 対象 R (score 有): **{len(s)}** (= 5/9 daily_predictions のみ)")
        out.append(f"- top1_score 平均: **{s.mean():.4f}**")
        out.append(f"- 中央値: {s.median():.4f}")
        out.append(f"- 最大: {s.max():.4f} / 最小: {s.min():.4f}")
        out.append(f"- 各 R: {[f'{v:.3f}' for v in s.tolist()]}")
    else:
        out.append("- ⚠ score 値 0 件 (5/2, 5/3 の cumulative csv 欠損)")
    out.append("")
    out.append("> 5/2, 5/3 の top1_score は cumulative_results.csv で NaN (95% 欠損 既知)。 score 値での分布は 5/9 4 R のみ。")
    out.append("")

    # V15 hit rate (production)
    out.append("## 3. V15 hit rate (production saved、 5/2-5/3 の cumulative + 5/9 案B改 投票結果)")
    out.append("")
    # top1 が 1着 (top1_finish == 1)
    f1 = pd.to_numeric(df["top1_finish"], errors="coerce")
    f2 = pd.to_numeric(df["top2_finish"], errors="coerce")
    f3 = pd.to_numeric(df["top3_finish"], errors="coerce")
    n_with_finish = f1.notna().sum()
    if n_with_finish > 0:
        top1_win  = (f1 == 1).sum()
        top1_top3 = ((f1 >= 1) & (f1 <= 3)).sum()
        # top1/2/3 全員が 1-3 着内 (= trio top3 perfect)
        all3 = ((f1 <= 3) & (f2 <= 3) & (f3 <= 3) & f1.notna() & f2.notna() & f3.notna()).sum()
        out.append(f"- 対象 R: {n_with_finish} (cumulative 11 + 5/9 案B改 投票 1 = 11 R は finish populated、 5/9 残り 3 R は本 session で finish 未取得)")
        out.append(f"- top1 が 1 着: **{top1_win}/{n_with_finish}** ({top1_win/n_with_finish*100:.1f}%)")
        out.append(f"- top1 が 3 着内: **{top1_top3}/{n_with_finish}** ({top1_top3/n_with_finish*100:.1f}%)")
        out.append(f"- top1/2/3 全員 3 着内 (perfect trio): **{all3}/{n_with_finish}** ({all3/n_with_finish*100:.1f}%)")
    out.append("")

    # 案B改 仮投票 ROI (確定済 + 5/9 実投票 を separate)
    out.append("## 4. 案B改 strict 7 点三連複 ROI (production、 確定済のみ)")
    out.append("")

    # 4-1 settled (5/2-5/3 cumulative)
    settled = df[df["trio_hit"].notna()].copy()
    s_inv = int(pd.to_numeric(settled["investment"], errors="coerce").fillna(0).sum())
    s_pay = int(pd.to_numeric(settled["actual_payout"], errors="coerce").fillna(0).sum())
    s_pro = int(pd.to_numeric(settled["profit"], errors="coerce").fillna(0).sum())
    s_hit = int(pd.to_numeric(settled["trio_hit"], errors="coerce").gt(0).sum())
    s_n = len(settled)
    out.append("### 4-1. 5/2 + 5/3 確定済 (cumulative_results.csv 由来、 V15 案B改 7 点 三連複 全 R 投票実行)")
    out.append("")
    out.append(f"- 投票 R: **{s_n}**")
    out.append(f"- 投資: **¥{s_inv:,}** / 払戻: **¥{s_pay:,}** / 損益: **{'+' if s_pro>=0 else ''}¥{s_pro:,}**")
    out.append(f"- ROI: **{s_pay/s_inv*100 if s_inv else 0:.1f}%**")
    out.append(f"- hit: **{s_hit}/{s_n}** ({s_hit/s_n*100 if s_n else 0:.1f}%)")
    out.append("")

    # 4-2 5/9 案B改 strict 投票 1 R (Session #67 確定)
    out.append("### 4-2. 5/9 案B改 strict 投票 (新潟 12R 1勝 のみ)")
    out.append("")
    out.append("- 投票 R: **1** (新潟 12R 4歳以上1勝、 軸 11 ハイクオリティ)")
    out.append("- 結果: 軸 11 → **3 着**、 1-2-3 着 = `3-8-11` → 三連複 7 点 全 miss")
    out.append("- 投資 ¥700 / 払戻 ¥0 / 損益 **-¥700**")
    out.append("- (Session #67 確定値)")
    out.append("")

    # 4-3 5/9 verdict 未投票 (3 R)
    not_voted_5_9 = df[(df["date"] == "20260509") & (df["trio_hit"].isna())].copy()
    nv_n = len(not_voted_5_9)
    nv = "/".join([f"{r['course']} {int(r['race_num'])}R {r['race_name']}" for _, r in not_voted_5_9.iterrows() if not (int(r['race_num']) == 12 and "1勝" in str(r['race_name']))])
    out.append(f"### 4-3. 5/9 verdict 用 (案B改 strict 除外、 投票なし)")
    out.append("")
    out.append(f"- 対象 R: 3 (京都 12R 2勝 / 東京 12R 2勝 / 新潟 11R OP)")
    out.append(f"- 案B改 strict は 12R 1勝 のみ → 上記 3 R は filter で除外、 投票実行なし")
    out.append(f"- これらは Session #67 で 5 system 比較 / もし投票してたら ROI 算出 (本 session では集計外)")
    out.append("")

    # 4-4 月次合計 (確定済 + 5/9 案B改)
    total_inv = s_inv + 700
    total_pay = s_pay  # 5/9 -> 0
    total_pro = s_pro - 700
    total_n = s_n + 1
    total_hit = s_hit  # 5/9 vote miss
    out.append(f"### 4-4. 5月 累計 (確定済 + 5/9 案B改 strict 1R)")
    out.append("")
    out.append(f"- 投票 R: **{total_n}**")
    out.append(f"- 投資: **¥{total_inv:,}** / 払戻: **¥{total_pay:,}** / 損益: **{'+' if total_pro>=0 else ''}¥{total_pro:,}**")
    out.append(f"- ROI: **{total_pay/total_inv*100 if total_inv else 0:.1f}%**")
    out.append(f"- hit: **{total_hit}/{total_n}** ({total_hit/total_n*100 if total_n else 0:.1f}%)")
    out.append("")
    # save for stats stash for later
    main._stats = dict(s_n=s_n, s_pro=s_pro, s_pay=s_pay, s_inv=s_inv, s_hit=s_hit,
                       total_n=total_n, total_pro=total_pro, total_pay=total_pay,
                       total_inv=total_inv, total_hit=total_hit)
    out.append("")

    # クラス別
    out.append("## 5. クラス別")
    out.append("")
    out.append("| クラス | R 数 | hit | hit 率 | 投資 | 払戻 | 損益 |")
    out.append("|---|---|---|---|---|---|---|")
    for cls, sub in df.groupby("class"):
        n = len(sub)
        h = int(pd.to_numeric(sub["trio_hit"], errors="coerce").fillna(0).gt(0).sum())
        i_ = int(pd.to_numeric(sub["investment"], errors="coerce").fillna(0).sum())
        p_ = int(pd.to_numeric(sub["actual_payout"], errors="coerce").fillna(0).sum())
        pr = int(pd.to_numeric(sub["profit"], errors="coerce").fillna(0).sum())
        hr = h / n * 100 if n else 0
        out.append(f"| {cls} | {n} | {h} | {hr:.1f}% | ¥{i_:,} | ¥{p_:,} | {'+' if pr >= 0 else ''}¥{pr:,} |")
    out.append("")

    # surface 別
    out.append("## 6. surface 別 (data 有 R のみ)")
    out.append("")
    surf = df["surface"].fillna("?")
    out.append("| 馬場 | R 数 | hit | hit 率 | 損益 |")
    out.append("|---|---|---|---|---|")
    for sname, sub in df.groupby(surf):
        n = len(sub)
        h = int(pd.to_numeric(sub["trio_hit"], errors="coerce").fillna(0).gt(0).sum())
        pr = int(pd.to_numeric(sub["profit"], errors="coerce").fillna(0).sum())
        hr = h / n * 100 if n else 0
        out.append(f"| {sname} | {n} | {h} | {hr:.1f}% | {'+' if pr >= 0 else ''}¥{pr:,} |")
    out.append("")

    # 5/16 V18 trial 含意
    out.append("## 7. 5/16 V18 trial 含意 (production data から)")
    out.append("")
    out.append("- V15 案B改 strict は 5月 12 R (重賞除外、 production saved) で **hit 率 / ROI 集計可能**。 上記 #4 を base data として 5/16 V18 trial GO/NO-GO の比較対象に使える")
    out.append("- 5/9 単独 -¥700 (投票 1R/MISS) は 5月全体の hit 率を歪める可能性 → 5/2, 5/3 を含めた集計が代表値")
    out.append("- ★ 5/2-5/3 の score 値欠損 (cumulative_results.csv バグ) は **production save logic の補修候補** → Session #65/68 の Stage 2 system 修復と並行で daily_predict.py の保存ロジック audit が必要")
    out.append("")

    out_path = BASE / "data" / "v18" / "may_filtered_summary.md"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"written: {out_path.relative_to(BASE)} ({out_path.stat().st_size} bytes)")
    print()
    print("--- key stats ---")
    s = main._stats
    print(f"5/2-5/3 settled: {s['s_n']} R, hit {s['s_hit']}, ROI {s['s_pay']/s['s_inv']*100:.1f}%, profit {s['s_pro']:+,}")
    print(f"5/9 case B vote: 1 R, miss, profit -700")
    print(f"5月 total (settled + 5/9 vote): {s['total_n']} R, hit {s['total_hit']}, ROI {s['total_pay']/s['total_inv']*100:.1f}%, profit {s['total_pro']:+,}")


if __name__ == "__main__":
    main()
