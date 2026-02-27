import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競馬AI予想", page_icon="🏇", layout="centered")

st.markdown("""
<style>
body { background-color: #0f1a0f; }
.stApp { background-color: #0f1a0f; }
h1 { color: #4caf6e !important; }
.result-card { background: #1a2a1a; border-radius: 12px; padding: 12px 16px; margin: 6px 0; border-left: 4px solid #4caf6e; }
.rank-1 { border-left-color: #e8c84a; }
.rank-2 { border-left-color: #c0c0c0; }
.rank-3 { border-left-color: #d4924a; }
</style>
""", unsafe_allow_html=True)

st.title("🏇 競馬AI予想")
st.caption("※ 個人利用のみ。馬券購入は自己責任でお願いします。")

FEATURES = ['人気','馬体重','場体重増減','斤量','馬齢','距離(m)','競馬場コード_enc','芝ダート_enc','馬場状態_enc','性別_enc','騎手勝率','前走着順']

@st.cache_resource
def load_model():
    try:
        with open("keiba_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("モデルの読み込みに失敗しました。しばらく待ってからページを更新してください。")
    st.stop()

st.success("AIモデル読み込み完了！")

HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"}

def scrape_shutuba(race_url):
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
            jockey_tag = row.select_one("td.Jockey a​​​​​​​​​​​​​​​​
