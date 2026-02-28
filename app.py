import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, re
from bs4 import BeautifulSoup
import requests

# ––––– 定数 –––––

MODEL_PATH = “keiba_model.pkl”
JOCKEY_PATH = “jockey_wr.json”

FEATURES = [
‘馬体重’,‘場体重増減’,‘斤量’,‘馬齢’,‘距離(m)’,
‘競馬場コード_enc’,‘芝ダート_enc’,‘馬場状態_enc’,
‘性別_enc’,‘騎手勝率’,‘前走着順’
]

COURSE_MAP = {
‘札幌’:0,‘函館’:1,‘福島’:2,‘新潟’:3,‘東京’:4,
‘中山’:5,‘中京’:6,‘京都’:7,‘阪神’:8,‘小倉’:9
}
SURFACE_MAP  = {‘芝’:0, ‘ダート’:1}
CONDITN_MAP  = {‘良’:0, ‘稍重’:1, ‘重’:2, ‘不良’:3}
SEX_MAP      = {‘牡’:0, ‘牝’:1, ‘セ’:2}

# ––––– モデル / 騎手辞書の読み込み –––––

@st.cache_resource
def load_model():
with open(MODEL_PATH, “rb”) as f:
return pickle.load(f)

@st.cache_resource
def load_jockey():
with open(JOCKEY_PATH, “r”, encoding=“utf-8”) as f:
return json.load(f)

model = load_model()
jockey_dict = load_jockey()

# ––––– スクレイピング –––––

def scrape(race_url: str):
“”“netkeiba のレースページから出馬表を取得”””
res = requests.get(race_url, timeout=10)
res.encoding = res.apparent_encoding
soup = BeautifulSoup(res.text, “html.parser”)

```
# レース情報
race_name = soup.select_one("h1.RaceName").text.strip() if soup.select_one("h1.RaceName") else ""

# レース詳細（距離・馬場など）
race_data_el = soup.select_one("div.RaceData01")
race_data_text = race_data_el.text.strip() if race_data_el else ""

# 距離
dist_m = re.search(r'(\d{4})m', race_data_text)
distance = int(dist_m.group(1)) if dist_m else 0

# 芝・ダート
if 'ダート' in race_data_text:
    surface = 'ダート'
else:
    surface = '芝'

# 馬場状態
condition = '良'
for cond in ['不良','重','稍重','良']:
    if cond in race_data_text:
        condition = cond
        break

# 競馬場
course_name = ''
for name in COURSE_MAP:
    if name in race_data_text or name in race_name:
        course_name = name
        break
# URLからも取得を試みる
if not course_name:
    for name in COURSE_MAP:
        if name in race_url:
            course_name = name
            break

# 出馬テーブル
horses = []
rows = soup.select("table.Shutuba_Table tr.HorseList")
for row in rows:
    # 取消馬を除外
    row_class = row.get("class", [])
    if "Cancel" in row_class:
        continue

    tds = row.select("td")
    if len(tds) < 12:
        continue

    try:
        # 馬番
        umaban = tds[1].text.strip()
        # 馬名
        horse_name_el = row.select_one("span.HorseName a")
        horse_name = horse_name_el.text.strip() if horse_name_el else tds[3].text.strip()
        # 性別・馬齢
        sex_age = tds[4].text.strip()
        sex = sex_age[0] if sex_age else '牡'
        age = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else 3
        # 斤量
        kinryo = float(tds[5].text.strip()) if tds[5].text.strip() else 55.0
        # 騎手
        jockey_el = row.select_one("td.Jockey a")
        jockey = jockey_el.text.strip() if jockey_el else tds[6].text.strip()
        # 馬体重
        weight_text = tds[8].text.strip() if len(tds) > 8 else ""
        bw_match = re.search(r'(\d+)\(([\+\-]?\d+)\)', weight_text)
        if bw_match:
            body_weight = int(bw_match.group(1))
            weight_diff = int(bw_match.group(2))
        else:
            # 馬体重がまだ発表されていない場合
            bw_only = re.search(r'(\d+)', weight_text)
            body_weight = int(bw_only.group(1)) if bw_only else 480
            weight_diff = 0

        horses.append({
            "馬番": umaban,
            "馬名": horse_name,
            "性別": sex,
            "馬齢": age,
            "斤量": kinryo,
            "騎手": jockey,
            "馬体重": body_weight,
            "場体重増減": weight_diff,
        })
    except Exception:
        continue

return {
    "race_name": race_name,
    "distance": distance,
    "surface": surface,
    "condition": condition,
    "course": course_name,
    "horses": horses,
}
```

# ––––– 予測 –––––

def predict(race_info: dict):
rows = []
for h in race_info[“horses”]:
jwr = jockey_dict.get(h[“騎手”], 0.05)
row = {
“馬体重”: h[“馬体重”],
“場体重増減”: h[“場体重増減”],
“斤量”: h[“斤量”],
“馬齢”: h[“馬齢”],
“距離(m)”: race_info[“distance”],
“競馬場コード_enc”: COURSE_MAP.get(race_info[“course”], 4),
“芝ダート_enc”: SURFACE_MAP.get(race_info[“surface”], 0),
“馬場状態_enc”: CONDITN_MAP.get(race_info[“condition”], 0),
“性別_enc”: SEX_MAP.get(h[“性別”], 0),
“騎手勝率”: jwr,
“前走着順”: 5,  # 暫定固定値
}
rows.append(row)

```
df = pd.DataFrame(rows)[FEATURES]
proba = model.predict_proba(df)[:, 1]
return proba
```

# ––––– UI –––––

st.set_page_config(page_title=“競馬AI予想”, layout=“wide”)
st.title(“🏇 競馬AI予想アプリ”)

url = st.text_input(“netkeibaのレースURL（出馬表）を入力してください”)

if st.button(“予測する”) and url:
with st.spinner(“データ取得中…”):
try:
race = scrape(url)
except Exception as e:
st.error(f”スクレイピングに失敗しました: {e}”)
st.stop()

```
if not race["horses"]:
    st.warning("出馬データを取得できませんでした。URLを確認してください。")
    st.stop()

st.subheader(race["race_name"])
st.write(f"{race['course']}  {race['surface']}{race['distance']}m  馬場: {race['condition']}")

with st.spinner("AI予測中..."):
    proba = predict(race)

result = pd.DataFrame({
    "馬番": [h["馬番"] for h in race["horses"]],
    "馬名": [h["馬名"] for h in race["horses"]],
    "騎手": [h["騎手"] for h in race["horses"]],
    "AI勝率(%)": np.round(proba * 100, 2),
})
result = result.sort_values("AI勝率(%)", ascending=False).reset_index(drop=True)
result.index = result.index + 1
result.index.name = "AI順位"

st.dataframe(result, use_container_width=True)
```
