"""
レース開催日クロスチェック + netkeiba一覧取得の堅牢化（フェイルサイレント解消）。

【背景】
2026-05-31(ダービー当日)、開催日(東京12R+京都12R=24R)なのに netkeiba の
レース一覧取得が朝ずっと空 → daily_predict / race_auto_notify が「非開催日」と
静かに誤判定 → per-race通知・paper記録が丸ごと不発になった。
JRDB 側は当日 KYI(出走表)を完備していた = netkeiba 0件は「取得失敗」だった。
これは paci/ZE の「静かな停止」と同じ構造の障害。

【このモジュールの役割】
1. JRDB の KYI(出走表)ファイルで「今日は開催日か」をクロスチェック
   （JRDB は実際の開催日にのみ KYI を発行するため、存在=開催日と確実に判定できる）
2. netkeiba 一覧が空でも「開催日」なら時間を置いてリトライ（一時的ブリップを救う）
3. 開催日なのに最終的に取得できなければ Discord #アップデート に警告（フェイルラウド）

【不変】 V15/V16 ・ predict_core の予測ロジックには一切触れない（障害修正のみ）。
"""
import os
import time


# JRDB 場コード(参考): 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉


def jrdb_race_day_info(date_str, base_dir):
    """JRDB の当日出走表(KYI)から「開催日かどうか」をクロスチェックする。

    JRDB は実際の開催日にのみ KYI(出走表)を発行する。したがって当日の
    KYI{YYMMDD}.txt が存在すれば「今日は開催日」と確実に判定できる。netkeiba
    一覧が 0 件でも、JRDB に KYI があれば「netkeiba取得失敗」と切り分けられる。

    Args:
        date_str: 'YYYYMMDD'
        base_dir: keiba-ai リポジトリルート

    Returns: dict {
        'is_race_day': bool,   # JRDB に当日出走表があるか
        'n_races': int,        # JRDB 上のレース数(開催場×R を distinct 集計)
        'source': str,         # 参照した KYI ファイルパス('' if なし)
    }
    """
    yymmdd = date_str[2:]  # YYYYMMDD -> YYMMDD（JRDBファイル名形式）
    # DailyJrdbKyi は extracted/Kyi/、PaciDailyRefresh は extracted/Paci/ に展開する。
    # どちらに最新が来るかは運用で揺れるため両方を探す。
    candidates = [
        os.path.join(base_dir, 'data', 'jrdb', 'extracted', 'Kyi', f'KYI{yymmdd}.txt'),
        os.path.join(base_dir, 'data', 'jrdb', 'extracted', 'Paci', f'KYI{yymmdd}.txt'),
    ]
    for path in candidates:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 100:
                # JRDB KYI 1行=1頭。先頭8文字 = 場+年+回+日+R のレースキー。
                # distinct 集計でその日のレース数になる(24 = 2場×12R 等)。
                race_keys = set()
                with open(path, encoding='cp932', errors='replace') as f:
                    for line in f:
                        if len(line) >= 8:
                            race_keys.add(line[:8])
                if race_keys:
                    return {'is_race_day': True, 'n_races': len(race_keys), 'source': path}
        except Exception:
            # 読めない/壊れている場合は次の候補へ(判定不能=Falseに倒す)
            continue
    return {'is_race_day': False, 'n_races': 0, 'source': ''}


def fetch_race_list_robust(fetch_fn, date_str, base_dir,
                           log=print, notify_fn=None,
                           retries=3, delay_seconds=120):
    """netkeiba一覧取得を「開催日クロスチェック + リトライ + 警告」で堅牢化する。

    挙動:
      - fetch_fn() が非空 → そのまま返す(通常の開催日。従来通り)。
      - 空 かつ JRDB に当日 KYI なし → 真の非開催日。静かに返す(平日で誤検知しない)。
      - 空 かつ JRDB に当日 KYI あり → ★開催日なのに0件=取得失敗★ と判定し、
          時間を置いて retries 回リトライ。回復すれば返す。
          全リトライ失敗なら Discord #アップデート に警告(フェイルラウド)して返す。

    Args:
        fetch_fn: () -> list   引数なしで netkeiba 一覧を返す callable(空なら [] 期待)
        date_str: 'YYYYMMDD'
        base_dir: keiba-ai ルート
        log: ログ出力関数(default print)
        notify_fn: (message:str)->None  Discord警告送信関数(None なら送信skip)
        retries: 「開催日なのに空」のときの追加リトライ回数
        delay_seconds: リトライ間隔(秒)。一時的な netkeiba ブリップを跨ぐため長め

    Returns: dict {
        'races': list,          # 最終的に取得できたレース一覧
        'is_race_day': bool,    # JRDB クロスチェック結果(真の開催日か)
        'n_races_jrdb': int,    # JRDB 上のレース数
        'fetch_failed': bool,   # 開催日なのに取得できなかった(=要対応)
    }
    """
    races = fetch_fn() or []
    if races:
        return {'races': races, 'is_race_day': True, 'n_races_jrdb': 0, 'fetch_failed': False}

    # netkeiba 一覧が空。本当に非開催日か、取得失敗かを JRDB でクロスチェック。
    info = jrdb_race_day_info(date_str, base_dir)
    if not info['is_race_day']:
        # JRDB にも当日出走表なし = 真の非開催日(平日等)。従来通り静かに終了。
        log(f"[RACE-DAY] {date_str}: JRDB出走表なし = 非開催日(正常)")
        return {'races': [], 'is_race_day': False, 'n_races_jrdb': 0, 'fetch_failed': False}

    # ★ JRDB は開催日(KYI {n}R 有)と言っているのに netkeiba 0件 = 取得失敗の疑い ★
    src = os.path.basename(info['source'])
    log(f"[RACE-DAY] ★異常検知★ {date_str} は開催日(JRDB {src} {info['n_races']}R)だが "
        f"netkeiba一覧 0件 = 取得失敗の疑い。リトライします。")

    for attempt in range(1, retries + 1):
        log(f"[RACE-DAY] netkeiba一覧 リトライ {attempt}/{retries}（{delay_seconds}s待機後に再取得）...")
        time.sleep(delay_seconds)
        races = fetch_fn() or []
        if races:
            log(f"[RACE-DAY] リトライ {attempt} で {len(races)}件取得成功 → 復旧")
            return {'races': races, 'is_race_day': True,
                    'n_races_jrdb': info['n_races'], 'fetch_failed': False}

    # 全リトライ失敗 = フェイルラウド警告(静かにexitしない)
    msg = (f"{date_str} は開催日(JRDB出走表 {info['n_races']}R 確認)ですが、"
           f"netkeibaレース一覧が {retries}回リトライしても 0件でした。\n"
           f"netkeiba一覧の取得失敗の可能性が高く、予測・通知が未実行のまま終了します。\n"
           f"netkeiba回復後に手動再実行してください: python tools/daily_predict.py --date {date_str}\n"
           f"(午前 netkeiba ダウン → 午後回復のケースあり)")
    log(f"[RACE-DAY] [WARN] {msg}")
    if notify_fn is not None:
        try:
            notify_fn(msg)
            log("[RACE-DAY] Discord #アップデート に警告送信しました")
        except Exception as e:
            log(f"[RACE-DAY] Discord警告送信失敗(処理は継続): {e}")
    return {'races': [], 'is_race_day': True,
            'n_races_jrdb': info['n_races'], 'fetch_failed': True}
