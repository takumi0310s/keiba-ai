import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競馬AI予想", page_icon="🏇", layout="centered")
st.title("🏇 競馬AI予想")
st.caption("※ 個人利用のみ。馬券購入は自己責任でお願いします。")

FEATURES = ['人気','馬体重','場体重増減','斤量','馬齢','距離(m)','競馬場コード_enc','芝ダート_enc','馬場状態_enc','性別_enc','騎手勝率','前走着順']

@st.cache_resource
def load_model():
    try:
        with open("keiba_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("jockey_wr.json", "r", encoding="utf-8") as f:
            jockey_wr = json.load(f)
        return model, jockey_wr
    except Exception as e:
        st.error("読み込みエラー: " + str(e))
        return None, {}

model, jockey_wr = load_model()
if model is None:
    st.stop()

st.success("AIモデル読み込み完了！騎手データ " + str(len(jockey_wr)) + "人")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}

def scrape(url):
    url = url.replace("race.sp.netkeiba", "race.netkeiba")
    res = requests.get(url, headers=HEADERS, timeout=15)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.text, "html.parser")

    distance, surface, condition, venue = 2000, "芝", "良", "東京"
    try:
        data = soup.select_one(".RaceData01")
        if data:
            text = data.get_text()
            m = re.search(r'(\d+)m', text)
            if m:
                distance = int(m.group(1))
            surface = "ダート" if "ダ" in text else "芝"
            for c in ["不良","重","稍重","良"]:
                if c in text:
                    condition = c
                    break
        for v in ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]:
            if v in res.text[:5000]:
                venue = v
                break
    except:
        pass

    horses = []
    for i, row in enumerate(soup.select("tr.HorseList")):
        try:
            n = row.select_one("span.HorseName a")
            if not n or not n.text.strip():
                continue
            name = n.text.strip()

            pop_tag = row.select_one("td.Ninki")
            pop = int(pop_tag.text.strip()) if pop_tag and pop_tag.text.strip().isdigit() else i+1

            j = row.select_one("td.Jockey a")
            jockey = re.sub(r'[▲△☆]', '', j.text.strip() if j else "不明").strip()

            b = row.select_one("td.Kinryo")
            burden = float(b.text.strip()) if b else 55.0

            s = row.select_one("td.Barei")
            seire = s.text.strip() if s else "牡4"
            gender = seire[0] if seire else "牡"
            age = int(seire[1]) if len(seire) > 1 and seire[1].isdigit() else 4

            w = row.select_one("td.Weight")
            weight, wdiff = 480, 0
            if w:
                m2 = re.search(r'(\d+)\(([+-]?\d+)\)', w.text)
                if m2:
                    weight = int(m2.group(1))
                    wdiff = int(m2.group(2))

            horses.append({"馬名": name, "人気": pop, "性別": gender, "馬齢": age, "斤量": burden, "騎手": jockey, "馬体重": weight, "体重増減": wdiff})
        except:
            continue
    return horses, distance, surface, condition, venue

def predict(horses, distance, surface, condition, venue):
    vmap = {"東京":0,"中山":1,"阪神":2,"京都":3,"中京":4,"札幌":5,"函館":6,"福島":7,"新潟":8,"小倉":9}
    rows = []
    for h in horses:
        wr = jockey_wr.get(h["騎手"], 0.05)
        rows.append({
            "馬名": h["馬名"], "人気": h["人気"], "馬体重": h["馬体重"],
            "場体重増減": h["体重増減"], "斤量": h["斤量"], "馬齢": h["馬齢"],
            "距離(m)": distance, "競馬場コード_enc": vmap.get(venue, 0),
            "芝ダート_enc": {"芝":0,"ダート":1}.get(surface, 0),
            "馬場状態_enc": {"良":0,"稍重":1,"重":2,"不良":3}.get(condition, 0),
            "性別_enc": {"牡":0,"牝":1,"せん":2}.get(h["性別"], 0),
            "騎手勝率": wr, "前走着順": 5
        })
    df = pd.DataFrame(rows)
    df["AIスコア"] = model.predict_proba(df[FEATURES])[:,1]
    return df.sort_values("AIスコア", ascending=False).reset_index(drop=True)

race_url = st.text_input("netkeibaのレースURL", placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=...")

if st.button("🤖 AI予想を実行", type="primary", use_container_width=True):
    if not race_url:
        st.error("URLを入力してください")
    else:
        with st.spinner("出馬表を取得中..."):
            horses, distance, surface, condition, venue = scrape(race_url)
        if not horses:
            st.error("出馬表が取得できませんでした。")
        else:
            st.info("📍 " + venue + " " + str(distance) + "m " + surface + " 馬場：" + condition + " " + str(len(horses)) + "頭")
            with st.spinner("AI分析中..."):
                result_df = predict(horses, distance, surface, condition, venue)
            st.success("予想完了！")
            medals = ["🥇","🥈","🥉"]
            for i, row in result_df.iterrows():
                medal = medals[i] if i < 3 else str(i+1) + "着"
                pct = int(row["AIスコア"] * 100)
                st.write(medal + " " + row["馬名"] + " " + str(pct) + "pt")
            top3 = result_df.head(3)["馬名"].tolist()
            st.subheader("💡 推奨馬券")
            st.write("単勝： " + top3[0])
            st.write("馬連： " + top3[0] + " - " + top3[1])
            st.write("三連複： " + top3[0] + " - " + top3[1] + " - " + top3[2])

st.caption("馬券購入は自己責任でお願いします。")
