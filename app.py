import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import pickle
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="競馬AI予想",
    page_icon="🏇",
    layout="centered"
)

# スタイル
st.markdown("""
<style>
body { background-color: #0f1a0f; }
.stApp { background-color: #0f1a0f; }
h1 { color: #4caf6e !important; }
.result-card {
    background: #1a2a1a;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    border-left: 4px solid #4caf6e;
}
.rank-1 { border-left-color: #e8c84a; }
.rank-2 { border-left-color: #c0c0c0; }
.rank-3 { border-left-color: #d4924a; }
</style>
""", unsafe_allow_html=True)

st.title("🏇 競馬AI予想")
st.caption("※ 個人利用のみ。馬券購入は自己責任でお願いします。")

# ============================================================
# モデル読み込み
# ============================================================
FEATURES = ['人気','馬体重','場体重増減','斤量','馬齢','距離(m)',
            '競馬場コード_enc','芝ダート_enc','馬場状態_enc','性別_enc','騎手勝率','前走着順']

@st.cache_resource
def load_model_and_data():
    if not os.path.exists("keiba_model.pkl"):
        return None, None, None
    with open("keiba_model.pkl", "rb") as f:
        model = pickle.load(f)
    if not os.path.exists("archive.zip"):
        return model, {}, {}
    with zipfile.ZipFile("archive.zip", "r") as z:
        with z.open("19860105-20210731_race_result.csv") as f:
            df = pd.read_csv(f, nrows=300000, encoding="utf-8-sig")
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["着順"])
    df["着順"] = df["着順"].astype(int)
    jockey_wr = df.groupby("騎手").apply(
        lambda x: (x["着順"]==1).sum()/len(x)
    ).to_dict()
    venue_map = dict(zip(df["競馬場名"].unique(),
                        pd.factorize(df["競馬場名"])[0][:len(df["競馬場名"].unique())]))
    return model, jockey_wr, venue_map

model, jockey_wr, venue_map = load_model_and_data()

if model is None:
    st.error("keiba_model.pkl が見つかりません。ColabでStep5を実行してモデルを学習させてください。")
    st.stop()

# ============================================================
# netkeibaからデータ取得
# ============================================================
HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"}

def get_race_list(venue_code):
    """開催レース一覧を取得"""
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_id={venue_code}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.encoding = "EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")
        races = []
        for a in soup.select("a[href*='shutuba']"):
            href = a.get("href","")
            m = re.search(r"race_id=(\d+)", href)
            if m:
                race_id = m.group(1)
                name = a.text.strip() or f"レース{race_id[-2:]}"
                races.append({"name": name, "race_id": race_id})
        return races
    except:
        return []

def scrape_shutuba(race_url):
    """出馬表をスクレイピング"""
    res = requests.get(race_url, headers=HEADERS, timeout=10)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.text, "html.parser")
    horses = []
    for row in soup.select("tr.HorseList"):
        try:
            name_tag = row.select_one("dt.Horse a") or row.select_one("td.Horse_Info a")
            if not name_tag:
                continue
            name = name_tag.text.strip()
            if not name:
                continue
            jockey_tag = row.select_one("td.Jockey a") or row.select_one("dd.Jockey a")
            jockey = jockey_tag.text.strip() if jockey_tag else "不明"
            jockey = re.sub(r'[▲△☆]', '', jockey).strip()
            burden_tag = row.select_one("td.Kinryo") or row.select_one("dd.Kinryo")
            burden = float(burden_tag.text.strip()) if burden_tag else 55.0
            seire_tag = row.select_one("td.Barei") or row.select_one("dd.Barei")
            seire = seire_tag.text.strip() if seire_tag else "牡4"
            gender = seire[0] if seire else "牡"
            age = int(seire[1]) if len(seire)>1 and seire[1].isdigit() else 4
            wt_tag = row.select_one("td.Weight") or row.select_one("dd.Weight")
            weight, wdiff = 480, 0
            if wt_tag:
                m = re.search(r'(\d+)\(([+-]?\d+)\)', wt_tag.text)
                if m:
                    weight = int(m.group(1))
                    wdiff = int(m.group(2))
            horses.append({
                "馬名": name, "性別": gender, "馬齢": age,
                "斤量": burden, "騎手": jockey,
                "馬体重": weight, "体重増減": wdiff
            })
        except:
            continue
    return horses

def predict(horses, distance, surface, condition, venue):
    rows = []
    for i, h in enumerate(horses):
        past = 5
        if jockey_wr:
            past_data = []
        rows.append({
            "馬名": h["馬名"],
            "人気": i+1,
            "馬体重": h["馬体重"],
            "場体重増減": h["体重増減"],
            "斤量": h["斤量"],
            "馬齢": h["馬齢"],
            "距離(m)": distance,
            "競馬場コード_enc": venue_map.get(venue, 0) if venue_map else 0,
            "芝ダート_enc": {"芝":0,"ダート":1}.get(surface, 0),
            "馬場状態_enc": {"良":0,"稍重":1,"重":2,"不良":3}.get(condition, 0),
            "性別_enc": {"牡":0,"牝":1,"せん":2}.get(h["性別"], 0),
            "騎手勝率": jockey_wr.get(h["騎手"], 0.05) if jockey_wr else 0.05,
            "前走着順": past
        })
    result_df = pd.DataFrame(rows)
    result_df["AIスコア"] = model.predict_proba(result_df[FEATURES])[:,1]
    return result_df.sort_values("AIスコア", ascending=False).reset_index(drop=True)

# ============================================================
# UI
# ============================================================

# 入力方法選択
mode = st.radio("入力方法", ["URLを貼り付ける", "条件を手動入力"], horizontal=True)

if mode == "URLを貼り付ける":
    st.subheader("netkeibaのレースURL")
    race_url = st.text_input("", placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=...")

    col1, col2, col3 = st.columns(3)
    with col1:
        distance = st.number_input("距離(m)", value=2000, step=100)
    with col2:
        surface = st.selectbox("コース", ["芝","ダート"])
    with col3:
        condition = st.selectbox("馬場状態", ["良","稍重","重","不良"])

    venue = st.selectbox("競馬場", ["東京","中山","阪神","京都","中京","札幌","函館","福島","新潟","小倉"])

    if st.button("🤖 AI予想を実行", type="primary", use_container_width=True):
        if not race_url:
            st.error("URLを入力してください")
        else:
            race_url = race_url.replace("race.sp.netkeiba", "race.netkeiba")
            with st.spinner("出馬表を取得中..."):
                horses = scrape_shutuba(race_url)
            if not horses:
                st.error("出馬表が取得できませんでした。URLを確認してください。")
            else:
                with st.spinner(f"{len(horses)}頭のデータを分析中..."):
                    result_df = predict(horses, distance, surface, condition, venue)
                st.success(f"予想完了！{len(horses)}頭を分析しました")
                st.subheader("🏆 AI予想結果")
                medals = ["🥇","🥈","🥉"]
                for i, row in result_df.iterrows():
                    medal = medals[i] if i < 3 else f"{i+1}着"
                    rank_class = f"rank-{i+1}" if i < 3 else ""
                    pct = int(row["AIスコア"] * 100)
                    st.markdown(f"""
                    <div class="result-card {rank_class}">
                        {medal} <b>{row['馬名']}</b>
                        <span style="float:right;color:#4caf6e;">{pct}pt</span>
                        <div style="background:#0f1a0f;height:4px;border-radius:2px;margin-top:6px;">
                            <div style="background:#4caf6e;width:{pct}%;height:4px;border-radius:2px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                top3 = result_df.head(3)["馬名"].tolist()
                st.subheader("💡 推奨馬券")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("単勝", top3[0])
                with col2:
                    st.metric("馬連", f"{top3[0]}-{top3[1]}")
                with col3:
                    st.metric("三連複", f"{top3[0]}-{top3[1]}-{top3[2]}")

else:
    st.subheader("レース条件")
    col1, col2 = st.columns(2)
    with col1:
        venue = st.selectbox("競馬場", ["東京","中山","阪神","京都","中京","札幌","函館","福島","新潟","小倉"])
        distance = st.number_input("距離(m)", value=2000, step=100)
    with col2:
        surface = st.selectbox("コース", ["芝","ダート"])
        condition = st.selectbox("馬場状態", ["良","稍重","重","不良"])

    st.subheader("出走馬")
    num_horses = st.number_input("頭数", min_value=2, max_value=18, value=5)

    horses = []
    for i in range(int(num_horses)):
        with st.expander(f"{i+1}番馬", expanded=(i<3)):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("馬名", key=f"name_{i}", placeholder=f"馬{i+1}")
                gender = st.selectbox("性別", ["牡","牝","せん"], key=f"gender_{i}")
            with c2:
                weight = st.number_input("馬体重", value=480, key=f"w_{i}")
                age = st.number_input("馬齢", value=4, min_value=2, max_value=10, key=f"age_{i}")
            with c3:
                jockey = st.text_input("騎手", key=f"jockey_{i}", placeholder="騎手名")
                past = st.number_input("前走着順", value=5, min_value=1, max_value=18, key=f"past_{i}")
            horses.append({"馬名": name or f"馬{i+1}", "性別": gender, "馬齢": age,
                          "斤量": 55.0, "騎手": jockey, "馬体重": weight, "体重増減": 0})

    if st.button("🤖 AI予想を実行", type="primary", use_container_width=True):
        with st.spinner("AI分析中..."):
            result_df = predict(horses, distance, surface, condition, venue)
        st.success("予想完了！")
        st.subheader("🏆 AI予想結果")
        medals = ["🥇","🥈","🥉"]
        for i, row in result_df.iterrows():
            medal = medals[i] if i < 3 else f"{i+1}着"
            pct = int(row["AIスコア"] * 100)
            rank_class = f"rank-{i+1}" if i < 3 else ""
            st.markdown(f"""
            <div class="result-card {rank_class}">
                {medal} <b>{row['馬名']}</b>
                <span style="float:right;color:#4caf6e;">{pct}pt</span>
            </div>
            """, unsafe_allow_html=True)

        top3 = result_df.head(3)["馬名"].tolist()
        st.subheader("💡 推奨馬券")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("単勝", top3[0])
        with col2:
            st.metric("馬連", f"{top3[0]}-{top3[1]}")
        with col3:
            st.metric("三連複", f"{top3[0]}-{top3[1]}-{top3[2]}")

st.markdown("---")
st.caption("⚠️ 個人利用のみ。馬券購入は自己責任でお願いします。")
