import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import pickle
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
            return pickle.load(f)
    except:
        return None

model = load_model()
if model is None:
    st.error("モデル読み込み失敗。")
    st.stop()

st.success("AIモデル読み込み完了！")
HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)"}

def scrape(url):
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.text, "html.parser")
    horses = []

    # レース情報を自動取得
    distance, surface, condition, venue = 2000, "芝", "良", "東京"
    try:
        race_data = soup.select_one("div.RaceData01")
        if race_data:
            text = race_data.text
            m = re.search(r'(\d+)m', text)
            if m:
                distance = int(m.group(1))
            if "ダート" in text:
                surface = "ダート"
            for c in ["良","稍重","重","不良"]:
                if c in text:
                    condition = c
                    break
        venue_tag = soup.select_one("div.RaceName_OnlyPC") or soup.select_one("h1.RaceName")
        place_tag = soup.select_one("a.RaceKaisaiLink") or soup.select_one("span.RaceKaisaiDate")
        if place_tag:
            for v in ["東京","中山","阪神","京都","中京","札幌","函館","福島","新潟","小倉"]:
                if v in place_tag.text:
                    venue = v
                    break
    except:
        pass

    for row in soup.select("tr.HorseList"):
        try:
            n = row.select_one("dt.Horse a") or row.select_one("td.Horse_Info a")
            if not n or not n.text.strip():
                continue
            j = row.select_one("td.Jockey a") or row.select_one("dd.Jockey a")
            jockey = re.sub(r'[▲△☆]', '', j.text.strip() if j else "不明")
            b = row.select_one("td.Kinryo") or row.select_one("dd.Kinryo")
            burden = float(b.text.strip()) if b else 55.0
            s = row.select_one("td.Barei") or row.select_one("dd.Barei")
            seire = s.text.strip() if s else "牡4"
            gender = seire[0] if seire else "牡"
            age = int(seire[1]) if len(seire) > 1 and seire[1].isdigit() else 4
            w = row.select_one("td.Weight") or row.select_one("dd.Weight")
            weight, wdiff = 480, 0
            if w:
                m = re.search(r'(\d+)\(([+-]?\d+)\)', w.text)
                if m:
                    weight = int(m.group(1))
                    wdiff = int(m.group(2))
            horses.append({"馬名": n.text.strip(), "性別": gender, "馬齢": age, "斤量": burden, "騎手": jockey, "馬体重": weight, "体重増減": wdiff})
        except:
            continue
    return horses, distance, surface, condition, venue

def predict(horses, distance, surface, condition, venue):
    vmap = {"東京":0,"中山":1,"阪神":2,"京都":3,"中京":4,"札幌":5,"函館":6,"福島":7,"新潟":8,"小倉":9}
    rows = []
    for i, h in enumerate(horses):
        rows.append({"馬名": h["馬名"], "人気": i+1, "馬体重": h["馬体重"], "場体重増減": h["体重増減"], "斤量": h["斤量"], "馬齢": h["馬齢"], "距離(m)": distance, "競馬場コード_enc": vmap.get(venue, 0), "芝ダート_enc": {"芝":0,"ダート":1}.get(surface, 0), "馬場状態_enc": {"良":0,"稍重":1,"重":2,"不良":3}.get(condition, 0), "性別_enc": {"牡":0,"牝":1,"せん":2}.get(h["性別"], 0), "騎手勝率": 0.05, "前走着順": 5})
    df = pd.DataFrame(rows)
    df["AIスコア"] = model.predict_proba(df[FEATURES])[:,1]
    return df.sort_values("AIスコア", ascending=False).reset_index(drop=True)

race_url = st.text_input("netkeibaのレースURL", placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=...")

if st.button("🤖 AI予想を実行", type="primary", use_container_width=True):
    if not race_url:
        st.error("URLを入力してください")
    else:
        race_url = race_url.replace("race.sp.netkeiba", "race.netkeiba")
        with st.spinner("出馬表を取得中..."):
            horses, distance, surface, condition, venue = scrape(race_url)
        if not horses:
            st.error("出馬表が取得できませんでした。")
        else:
            st.info("📍 " + venue + " " + str(distance) + "m " + surface + " 馬場：" + condition)
            with st.spinner("AI分析中..."):
                result_df = predict(horses, distance, surface, condition, venue)
            st.success("予想完了！" + str(len(horses)) + "頭を分析しました")
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
