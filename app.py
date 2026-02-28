import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time

st.set_page_config(page_title="競馬AI予想", page_icon="🏇", layout="wide")

CSS = """
<style>
.top-card-gold {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border-left: 5px solid #FFD700; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.top-card-silver {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border-left: 5px solid #C0C0C0; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.top-card-bronze {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border-left: 5px solid #CD7F32; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.rank-gold { font-size: 2.0em; font-weight: bold; color: #FFD700; margin-right: 10px; }
.rank-silver { font-size: 2.0em; font-weight: bold; color: #C0C0C0; margin-right: 10px; }
.rank-bronze { font-size: 2.0em; font-weight: bold; color: #CD7F32; margin-right: 10px; }
.horse-name-card { font-size: 1.4em; font-weight: bold; color: #ffffff; }
.details-card { color: #aaaaaa; font-size: 0.95em; margin-top: 5px; }
.bar-bg { background: #333; border-radius: 10px; height: 12px; margin-top: 8px; overflow: hidden; }
.bar-gold { height: 12px; border-radius: 10px; background: linear-gradient(90deg, #FFD700, #FFA500); }
.bar-silver { height: 12px; border-radius: 10px; background: linear-gradient(90deg, #C0C0C0, #A0A0A0); }
.bar-bronze { height: 12px; border-radius: 10px; background: linear-gradient(90deg, #CD7F32, #A0522D); }
.race-hdr { background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); border-radius: 12px; padding: 15px 20px; margin: 15px 0; text-align: center; }
.race-hdr h2 { color: #e0e0e0; margin: 0; font-size: 1.6em; }
.race-hdr p { color: #aaa; margin: 5px 0 0 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("keiba_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_jockey_wr():
    with open("jockey_wr.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
jockey_wr = load_jockey_wr()

FEATURES = [
    '馬体重', '場体重増減', '斤量', '馬齢', '距離(m)',
    '競馬場コード_enc', '芝ダート_enc', '馬場状態_enc',
    '性別_enc', '騎手勝率', '前走着順'
]

COURSE_MAP = {
    '札幌': 0, '函館': 1, '福島': 2, '新潟': 3, '東京': 4,
    '中山': 5, '中京': 6, '京都': 7, '阪神': 8, '小倉': 9
}
SURFACE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
COND_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
SEX_MAP = {'牡': 0, '牝': 1, 'セ': 2, '騸': 2}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MODERN_JOCKEY_WR = {
    'ルメール': 0.220, 'C.ルメール': 0.220,
    '川田将雅': 0.210, '川田': 0.210, '武豊': 0.171,
    '戸崎圭太': 0.140, '戸崎圭': 0.140,
    '横山武史': 0.130, '横山武': 0.130,
    '松山弘平': 0.120, '松山弘': 0.120, '松山': 0.120,
    '池添謙一': 0.110, '池添謙': 0.110,
    '福永祐一': 0.160, '福永祐': 0.160,
    '岩田望来': 0.100, '岩田望': 0.100,
    '岩田康誠': 0.090, '岩田康': 0.090,
    '吉田隼人': 0.090, '吉田隼': 0.090,
    '三浦皇成': 0.080, '三浦皇': 0.080, '三浦': 0.110,
    '田辺裕信': 0.090, '田辺裕': 0.090, '田辺': 0.090,
    '横山和生': 0.100, '横山和': 0.100,
    '横山典弘': 0.121, '横山典': 0.121,
    '大野拓弥': 0.070, '大野拓': 0.070, '大野': 0.070,
    '菅原明良': 0.080, '菅原明': 0.080,
    '北村宏司': 0.070, '北村宏': 0.070,
    '北村友一': 0.080, '北村友': 0.080,
    '丹内祐次': 0.060, '丹内祐': 0.060,
    '柴田大知': 0.060, '柴田大': 0.060,
    '石橋脩': 0.080, '木幡巧也': 0.060, '木幡巧': 0.060,
    'M.デムーロ': 0.140, 'デムーロ': 0.140,
    'R.ムーア': 0.250, 'ムーア': 0.250,
    '坂井瑠星': 0.100, '坂井瑠': 0.100,
    '鮫島克駿': 0.080, '鮫島克': 0.080, '鮫島駿': 0.080,
    '西村淳也': 0.090, '西村淳': 0.090,
    '角田大和': 0.070, '角田大': 0.070,
    '佐々木大': 0.080, '佐々木': 0.080,
    '津村明秀': 0.070, '津村明': 0.070, '津村': 0.070,
    '団野大成': 0.070, '団野大': 0.070,
    '藤岡佑介': 0.080, '藤岡佑': 0.080,
    '幸英明': 0.060, '幸英': 0.060,
    '和田竜二': 0.070, '和田竜': 0.070, '浜中俊': 0.090,
    '柴田善臣': 0.103, '柴田善': 0.103,
    '内田博幸': 0.060, '内田博': 0.060,
    '江田照男': 0.081, '江田照': 0.081,
    '丸山元気': 0.070, '丸山元': 0.070, '菅原': 0.080,
    '永野猛蔵': 0.060, '永野猛': 0.060,
    '野中悠太': 0.050, '野中': 0.050,
    '原優介': 0.050, '原優': 0.050, '原': 0.050,
    '長浜鷹人': 0.040, '長浜': 0.040,
    '上里涼': 0.030, '上里': 0.030,
    '遠藤裕喜': 0.030, '遠藤': 0.030,
    '佐藤翔馬': 0.030, '佐藤翔': 0.030, '佐藤': 0.030,
    '石神道大': 0.040, '石神道': 0.040,
    '原田和真': 0.050, '原田和': 0.050,
    '太宰啓介': 0.050, '太宰': 0.050,
}

def find_jockey_wr(name):
    if name in MODERN_JOCKEY_WR:
        return MODERN_JOCKEY_WR[name]
    if name in jockey_wr:
        return jockey_wr[name]
    for key, val in jockey_wr.items():
        if key.startswith(name) or name.startswith(key):
            return val
    return 0.05

def get_odds(race_id):
    """単勝オッズを取得。{馬番: オッズ} を返す"""
    try:
        # 方法1: PC版オッズページ
        url = "https://race.netkeiba.com/odds/index.html?type=b1&race_id=" + race_id
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        text = resp.text
        odds_dict = {}
        # JavaScript内のオッズデータをパース
        # パターン: "1":{"umaban":"1","odds":"3.5",...}
        matches = re.findall(r'"umaban"\s*:\s*"(\d+)".*?"odds"\s*:\s*"([\d.]+)"', text)
        if matches:
            for um, od in matches:
                try:
                    odds_dict[int(um)] = float(od)
                except ValueError:
                    pass
            return odds_dict

        # 方法2: HTML テーブルからパース
        soup = BeautifulSoup(text, "html.parser")
        # Odds_Table内のデータ
        table = soup.find("table", id="Odds_Table") or soup.find("table", class_="RaceOdds_HorseList_Table")
        if table:
            for row in table.find_all("tr"):
                tds = row.find_all("td")
                if len(tds) >= 2:
                    for idx_um in range(len(tds)):
                        um_text = tds[idx_um].get_text(strip=True)
                        if um_text.isdigit() and 1 <= int(um_text) <= 18:
                            for idx_od in range(idx_um + 1, len(tds)):
                                od_text = tds[idx_od].get_text(strip=True).replace(',', '')
                                try:
                                    od_val = float(od_text)
                                    if 1.0 <= od_val <= 9999:
                                        odds_dict[int(um_text)] = od_val
                                        break
                                except ValueError:
                                    continue
                            break

        # 方法3: SP版オッズページ
        if not odds_dict:
            sp_url = "https://race.sp.netkeiba.com/?pid=odds&race_id=" + race_id
            resp2 = requests.get(sp_url, headers=HEADERS, timeout=10)
            resp2.encoding = "EUC-JP"
            sp_matches = re.findall(r'(\d+)番.*?(\d+\.\d+)倍', resp2.text)
            for um, od in sp_matches:
                try:
                    odds_dict[int(um)] = float(od)
                except ValueError:
                    pass

        return odds_dict
    except Exception:
        return {}

def get_horse_stats(horse_id, target_distance, target_surface):
    """前走着順 + 距離適性 + 馬場適性をPC版テーブルから取得"""
    last_finish = 5
    dist_results = []
    surf_results = []
    try:
        url = "https://db.netkeiba.com/horse/result/" + horse_id + "/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="db_h_race_results")
        if table:
            tbody = table.find("tbody")
            if tbody:
                rows = tbody.find_all("tr")
                for row_idx, row in enumerate(rows):
                    tds = row.find_all("td")
                    if len(tds) < 15:
                        continue
                    # 着順 (11列目, 0-indexed)
                    finish_text = tds[11].get_text(strip=True)
                    if not finish_text.isdigit():
                        continue
                    finish = int(finish_text)

                    # 前走着順 (最初の行)
                    if row_idx == 0:
                        last_finish = finish

                    # 距離情報 (14列目): "芝1600" or "ダ1200"
                    dist_col = tds[14].get_text(strip=True)
                    dm = re.match(r'([芝ダ障])(\d+)', dist_col)
                    if dm:
                        surf_ch = dm.group(1)
                        dist_val = int(dm.group(2))

                        # 距離適性: 今回の距離±200m以内の成績
                        if target_distance > 0 and abs(dist_val - target_distance) <= 200:
                            dist_results.append(finish)

                        # 馬場適性: 同じ芝/ダートの成績
                        surf_name = '芝' if surf_ch == '芝' else 'ダ'
                        if surf_name == target_surface:
                            surf_results.append(finish)

                    # 最大10走分のみ分析
                    if row_idx >= 9:
                        break

        # SP版フォールバック（前走着順のみ）
        if last_finish == 5:
            all_text = soup.get_text()
            m = re.search(r'(\d{1,2})\(\d+人気\)', all_text)
            if m:
                last_finish = int(m.group(1))

    except Exception:
        pass

    # 適性スコア計算
    dist_apt = 0.5
    surf_apt = 0.5
    if dist_results:
        avg = sum(dist_results) / len(dist_results)
        # 平均着順1→1.0, 9→0.5, 18→0.0
        dist_apt = max(0.0, min(1.0, 1.0 - (avg - 1) / 17.0))
    if surf_results:
        avg = sum(surf_results) / len(surf_results)
        surf_apt = max(0.0, min(1.0, 1.0 - (avg - 1) / 17.0))

    return last_finish, dist_apt, surf_apt

def parse_shutuba(race_id):
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + race_id
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")

    race_name = "レース"
    tag = soup.find("div", class_="RaceName")
    if tag and tag.get_text(strip=True):
        race_name = tag.get_text(strip=True)
    else:
        tag = soup.find("h1")
        if tag and tag.get_text(strip=True):
            race_name = tag.get_text(strip=True)
        else:
            tag = soup.find("title")
            if tag:
                title_text = tag.get_text(strip=True)
                m = re.search(r'(\S+\d+R)', title_text)
                if m:
                    race_name = m.group(1)

    race_data01 = soup.find("div", class_="RaceData01")
    data01_text = race_data01.get_text(strip=True) if race_data01 else ""
    if not data01_text:
        data01_text = soup.get_text()

    dist_match = re.search(r'(\d{4})m', data01_text)
    distance = int(dist_match.group(1)) if dist_match else 0
    surface = '芝' if '芝' in data01_text else ('ダ' if 'ダ' in data01_text else '芝')
    cond_match = re.search(r'馬場:(\S+)', data01_text)
    condition = cond_match.group(1) if cond_match else '良'

    race_data02 = soup.find("div", class_="RaceData02")
    data02_text = race_data02.get_text(strip=True) if race_data02 else data01_text
    course_name = ""
    for cname in COURSE_MAP.keys():
        if cname in data02_text:
            course_name = cname
            break

    race_info = dict(distance=distance, surface=surface, condition=condition, course=course_name)
    rows = soup.select("tr.HorseList")
    horses = []
    horse_ids = []

    for row in rows:
        row_class = row.get("class", [])
        if "Cancel" in row_class:
            continue
        name_tag = row.select_one("span.HorseName a")
        if name_tag is None:
            continue
        horse_name = name_tag.get_text(strip=True)
        href = name_tag.get("href", "")
        hid_match = re.search(r'/horse/(\d+)', href)
        horse_id = hid_match.group(1) if hid_match else None

        info_tag = row.select_one("td.Barei") or row.select_one("span.Barei")
        if info_tag:
            sex_age_text = info_tag.get_text(strip=True)
        else:
            tds_all = row.find_all("td")
            sex_age_text = ""
            for td in tds_all:
                t = td.get_text(strip=True)
                if re.match(r'^[牡牝セ騸]\d+$', t):
                    sex_age_text = t
                    break
        sex = sex_age_text[0] if sex_age_text else '牡'
        age = int(sex_age_text[1:]) if sex_age_text and sex_age_text[1:].isdigit() else 3

        kinryo = 55.0
        tds = row.find_all("td")
        for td in tds:
            t = td.get_text(strip=True)
            try:
                val = float(t)
                if 48.0 <= val <= 62.0:
                    kinryo = val
                    break
            except ValueError:
                continue

        jockey_tag = row.select_one("td.Jockey a") or row.select_one("a[href*='jockey']")
        jockey_name = jockey_tag.get_text(strip=True) if jockey_tag else ""

        umaban = 0
        umaban_tag = row.select_one("td.Umaban")
        if umaban_tag:
            ut = umaban_tag.get_text(strip=True)
            if ut.isdigit():
                umaban = int(ut)

        horse_weight = 480
        weight_diff = 0
        for td in tds:
            bw_text = td.get_text(strip=True)
            bw_match = re.search(r'(\d{3,})\(([\+\-]?\d+)\)', bw_text)
            if bw_match:
                w = int(bw_match.group(1))
                if 350 <= w <= 600:
                    horse_weight = w
                    weight_diff = int(bw_match.group(2))
                    break

        horses.append({
            '馬名': horse_name, '馬番': umaban,
            '馬体重': horse_weight, '場体重増減': weight_diff,
            '斤量': kinryo, '馬齢': age,
            '距離(m)': distance,
            '競馬場コード_enc': COURSE_MAP.get(course_name, 4),
            '芝ダート_enc': SURFACE_MAP.get(surface, 0),
            '馬場状態_enc': COND_MAP.get(condition, 0),
            '性別_enc': SEX_MAP.get(sex, 0),
            '騎手勝率': find_jockey_wr(jockey_name),
            '騎手名': jockey_name,
        })
        horse_ids.append(horse_id)
    return race_name, horses, horse_ids, race_info

def render_top3(rank, name, jockey, finish, score, max_score, odds_str, apt_str):
    medal_map = {1: '🥇', 2: '🥈', 3: '🥉'}
    color_map = {1: 'gold', 2: 'silver', 3: 'bronze'}
    medal = medal_map.get(rank, '')
    color = color_map.get(rank, 'bronze')
    bw = int((score / max_score) * 100) if max_score > 0 else 0
    html = '<div class="top-card-' + color + '">'
    html += '<div style="display:flex;align-items:center;">'
    html += '<span class="rank-' + color + '">' + medal + '</span>'
    html += '<div>'
    html += '<div class="horse-name-card">' + name + '</div>'
    html += '<div class="details-card">騎手: ' + jockey
    html += '｜前走: ' + str(finish) + '着'
    if odds_str:
        html += '｜' + odds_str
    html += '｜適性: ' + apt_str
    html += '｜スコア: ' + str(round(score, 3)) + '</div>'
    html += '</div></div>'
    html += '<div class="bar-bg"><div class="bar-' + color + '" style="width:' + str(bw) + '%"></div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apt_label(d, s):
    avg = (d + s) / 2
    if avg >= 0.7:
        return '◎'
    elif avg >= 0.5:
        return '○'
    elif avg >= 0.3:
        return '△'
    else:
        return '✕'

# ===== メイン =====
st.markdown("# 🏇 競馬AI予想")
url_input = st.text_input("netkeibaの出馬表URLを入力")

if st.button("🔍 予想する") and url_input:
    url_input = url_input.replace("race.sp.netkeiba.com", "race.netkeiba.com")
    rid_match = re.search(r'race_id=(\d+)', url_input)
    if not rid_match:
        st.error("URLからrace_idを取得できませんでした")
        st.stop()
    race_id = rid_match.group(1)

    with st.spinner("出馬表を取得中..."):
        race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
    if not horses:
        st.error("馬データを取得できませんでした。URLを確認してください。")
        st.stop()

    # レース情報ヘッダー
    surf = '🟢 芝' if race_info['surface'] == '芝' else '🟤 ダート'
    hdr = '<div class="race-hdr">'
    hdr += '<h2>' + race_info['course'] + ' ' + race_name + '</h2>'
    hdr += '<p>' + surf + ' ' + str(race_info['distance']) + 'm ｜ 馬場: '
    hdr += race_info['condition'] + ' ｜ ' + str(len(horses)) + '頭立て</p>'
    hdr += '</div>'
    st.markdown(hdr, unsafe_allow_html=True)

    # オッズ取得
    with st.spinner("オッズを取得中..."):
        odds_dict = get_odds(race_id)

    # 前走着順 + 距離適性 + 馬場適性取得 (PC版テーブルパース)
    with st.spinner("各馬の成績を分析中..."):
        progress_bar = st.progress(0)
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                lf, d_apt, s_apt = get_horse_stats(
                    hid, race_info['distance'], race_info['surface']
                )
                horse['前走着順'] = lf
                horse['距離適性'] = d_apt
                horse['馬場適性'] = s_apt
            else:
                horse['前走着順'] = 5
                horse['距離適性'] = 0.5
                horse['馬場適性'] = 0.5
            umaban = horse.get('馬番', 0)
            if umaban in odds_dict:
                horse['単勝オッズ'] = odds_dict[umaban]
            else:
                horse['単勝オッズ'] = 0.0
            progress_bar.progress((i + 1) / len(horses))
            if i < len(horses) - 1:
                time.sleep(0.5)
        progress_bar.empty()

    df = pd.DataFrame(horses)

    # AIモデルスコア
    X = df[FEATURES].values
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        ai_scores = proba[:, 1]
    else:
        ai_scores = proba[:, :3].sum(axis=1) if proba.shape[1] >= 3 else proba[:, 0]

    # オッズスコア
    odds_vals = df['単勝オッズ'].values
    has_odds = any(o > 0 for o in odds_vals)
    if has_odds:
        odds_scores = np.array([1.0 / o if o > 0 else 0 for o in odds_vals])
        if odds_scores.max() > 0:
            odds_scores = odds_scores / odds_scores.max()
    else:
        odds_scores = np.zeros(len(df))

    # 適性スコア
    apt_scores = (df['距離適性'].values + df['馬場適性'].values) / 2.0

    # 最終スコア
    if has_odds:
        final_scores = ai_scores * 0.50 + odds_scores * 0.30 + apt_scores * 0.20
    else:
        final_scores = ai_scores * 0.65 + apt_scores * 0.35

    df['AIスコア'] = ai_scores
    df['スコア'] = final_scores
    df['AI順位'] = df['スコア'].rank(ascending=False).astype(int)
    df = df.sort_values('AI順位')
    df['適性'] = df.apply(lambda r: apt_label(r['距離適性'], r['馬場適性']), axis=1)

    # TOP3
    st.markdown("### 🏆 AI推奨 TOP3")
    max_score = df['スコア'].max()
    for _, row in df.head(3).iterrows():
        odds_str = ''
        if row['単勝オッズ'] > 0:
            odds_str = '単勝 ' + str(row['単勝オッズ']) + '倍'
        render_top3(
            int(row['AI順位']), row['馬名'], row['騎手名'],
            int(row['前走着順']), row['スコア'], max_score,
            odds_str, row['適性']
        )

    # スコアグラフ
    st.markdown("### 📊 全馬スコア")
    chart_df = df[['馬名', 'スコア']].copy()
    chart_df = chart_df.set_index('馬名')
    st.bar_chart(chart_df, color='#FFD700')

    # 全馬テーブル
    st.markdown("### 📋 全馬データ")
    display_cols = ['AI順位', '馬名', '騎手名', '前走着順', '適性', '騎手勝率', 'スコア']
    if has_odds:
        display_cols.insert(4, '単勝オッズ')
    result_df = df[display_cols].copy()
    result_df['スコア'] = result_df['スコア'].map(lambda x: str(round(x, 3)))
    result_df['騎手勝率'] = result_df['騎手勝率'].map(lambda x: str(round(x, 3)))
    result_df = result_df.reset_index(drop=True)
    st.table(result_df)
