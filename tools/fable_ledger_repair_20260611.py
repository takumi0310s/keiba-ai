#!/usr/bin/env python3
"""Fable sweep: cumulative_results.csv 台帳外科手術 (2026-06-11)。
払戻監査 (docs/FABLE_SWEEP_LOG.md) で実証された4系統の修復:
 A. dateキー型不一致による 6/7 全23R 二重計上 → 正規化+dedup
 B. 5/23 丸一日欠落 (33R) → data/daily_results/20260523.csv から復元
 C. 3/14・3/15 的中17件が miss 記録 → daily_results CSV (jra_payouts と金額一致検証済) から hit/payout 修正
 D. 202606030509 の date 誤記 20260405 → 20260411

実行前に .bak_20260611_ledger を作成。全ステップ前後の Σprofit/n を表示。
"""
from __future__ import annotations
import os, sys, shutil
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

CUMUL = 'data/cumulative_results.csv'
BAK = 'data/cumulative_results.csv.bak_20260611_ledger'


def norm(v):
    s = str(v).strip()
    return s[:-2] if s.endswith('.0') else s


def stat(df, label):
    s = df[df['status'] == 'settled']
    p = pd.to_numeric(s['profit'], errors='coerce')
    print(f"  [{label}] n_settled={len(s)} Σprofit={p.sum():+,.0f}円")


def main():
    if not os.path.exists(BAK):
        shutil.copy2(CUMUL, BAK)
        print(f"backup -> {BAK}")
    df = pd.read_csv(CUMUL, encoding='utf-8-sig', low_memory=False)
    df['date'] = df['date'].map(norm)
    df['race_id'] = df['race_id'].map(norm)
    stat(df, '修復前(正規化後)')

    # A. dedup (settled優先・先勝ち)
    before = len(df)
    df['_r'] = (df['status'] == 'settled').astype(int)
    df = (df.sort_values(['date', 'race_id', '_r'], ascending=[True, True, False])
            .drop_duplicates(subset=['date', 'race_id'], keep='first').drop(columns=['_r']))
    print(f"A. 重複除去: {before} -> {len(df)} 行 (-{before - len(df)})")
    stat(df, 'A後')

    # B. 5/23 復元
    assert not (df['date'] == '20260523').any(), '5/23 が既に存在 — 状況が監査時と異なる。中断'
    d523 = pd.read_csv('data/daily_results/20260523.csv', encoding='utf-8-sig')
    d523['date'] = '20260523'
    d523['race_id'] = d523['race_id'].map(norm)
    d523['actual_payout'] = pd.to_numeric(d523.get('actual_payout',
                            pd.to_numeric(d523['trio_payout'], errors='coerce').fillna(0)
                            + pd.to_numeric(d523['umaren_payout'], errors='coerce').fillna(0)), errors='coerce')
    df = pd.concat([df, d523.reindex(columns=df.columns)], ignore_index=True)
    print(f"B. 5/23 復元: +{len(d523)} 行 (Σprofit {pd.to_numeric(d523['profit'], errors='coerce').sum():+,.0f}円)")
    stat(df, 'B後')

    # C. 3/14・3/15 hit修正 (daily_results CSV が正・jra_payouts 突合済)
    fixed = 0
    for date in ['20260314', '20260315']:
        src = pd.read_csv(f'data/daily_results/{date}.csv', encoding='utf-8-sig')
        src['race_id'] = src['race_id'].map(norm)
        src = src.set_index('race_id')
        m = df['date'] == date
        for i in df[m].index:
            rid = df.at[i, 'race_id']
            if rid not in src.index:
                continue
            r = src.loc[rid]
            cols = ['trio_bets_str', 'trio_hit', 'trio_payout', 'umaren_hit', 'umaren_payout',
                    'profit', 'trio_result', 'top1_finish', 'top2_finish', 'top3_finish']
            changed = False
            for c in cols:
                if c in df.columns and c in r.index and pd.notna(r[c]):
                    old = df.at[i, c]
                    if str(old) != str(r[c]):
                        df.at[i, c] = r[c]
                        changed = True
            # actual_payout 再構成
            ap = pd.to_numeric(r.get('trio_payout', 0), errors='coerce') or 0
            ap += pd.to_numeric(r.get('umaren_payout', 0), errors='coerce') or 0
            if 'actual_payout' in df.columns and float(pd.to_numeric(df.at[i, 'actual_payout'], errors='coerce') or 0) != float(ap):
                df.at[i, 'actual_payout'] = ap
                changed = True
            fixed += int(changed)
    print(f"C. 3/14・3/15 修正行: {fixed}")
    stat(df, 'C後')

    # D. date 誤記
    m = (df['race_id'] == '202606030509') & (df['date'] == '20260405')
    if m.any():
        df.loc[m, 'date'] = '20260411'
        print("D. 202606030509 date 20260405 -> 20260411")

    df = df.sort_values(['date', 'race_id']).reset_index(drop=True)
    stat(df, '最終')
    s = df[df['status'] == 'settled']
    inv = pd.to_numeric(s['investment'], errors='coerce').sum()
    pay = pd.to_numeric(s['actual_payout'], errors='coerce').sum()
    print(f"  最終 ROI={pay / inv * 100:.2f}% inv={inv:,.0f} pay={pay:,.0f}")
    df.to_csv(CUMUL, index=False, encoding='utf-8-sig')
    print(f"-> {CUMUL} 書き込み完了 (backup: {BAK})")


if __name__ == '__main__':
    main()
