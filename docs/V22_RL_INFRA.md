# V22 RL training infra

**作成**: Session #83
**用途**: V22 RL 学習環境 (GPU + library + 学習時間) の選定

---

## 1. GPU 要件

### 1-1. 推奨 GPU

| GPU | VRAM | 推定 学習時間 (30 年 data) | 備考 |
|-----|------|--------------------------|------|
| NVIDIA RTX 3090 | 24 GB | **2-3 日** | 推奨、 既存 PC 増設候補 |
| NVIDIA RTX 4090 | 24 GB | **1-2 日** | 最速、 高価 (40 万円+) |
| NVIDIA RTX 4080 | 16 GB | 2-3 日 | コスパ良、 batch size 制限あり |
| Google Colab Pro+ (A100) | 40 GB | **1 日** | 月額 ¥5,500、 cloud 推奨 |
| AWS p3.2xlarge (V100) | 16 GB | 2 日 | 時間課金 ($3.06/h)、 短期向き |

### 1-2. 既存 PC の評価

- 現状 GPU 確認 必要 (Session #56 で V20 ensemble 学習済 → CUDA 環境あり)
- VRAM 16 GB 以上なら **既存 PC OK**、 batch size 削減で対応
- VRAM 不足なら **Colab Pro+ 一択** (月額 5,500 円、 1-2 ヶ月で完結)

### 1-3. 推奨構成

```
phase 1 (10/1-10/15): PoC = 既存 PC
phase 2 (10/15-11/30): 本学習 = Colab Pro+ または 既存 PC で 2-3 日連続
```

---

## 2. Library 選定

| library | 特性 | V22 適合 |
|---------|------|---------|
| **stable-baselines3** | PyTorch base、 PPO/SAC/DQN 標準実装、 docs 充実 | ★ **推奨 (PoC)** ★ |
| **Ray RLlib** | 大規模 parallel 学習、 distributed | phase 2 で検討 |
| **CleanRL** | 単一ファイル実装、 customize 容易 | 学習用 |
| **Tianshou** | PyTorch base、 高速 | 代替候補 |

### 2-1. stable-baselines3 採用理由

1. PPO 実装が production-ready (RL の de facto standard)
2. gym env interface が clean
3. tensorboard / wandb 連携 容易
4. 既存 PyTorch 環境 (V20/V21) と互換

### 2-2. 必要パッケージ

```
torch>=2.0
stable-baselines3>=2.3
gymnasium>=0.29  # OpenAI gym 後継
sb3-contrib  # 拡張 algorithm
tensorboard
wandb  # 学習 monitoring (optional)
```

---

## 3. 学習時間 試算

### 3-1. 30 年 data 規模

| 項目 | 値 |
|------|----|
| 開催数 | 約 5,500 日 (30 年 × 平均 180 日/年) |
| race 数 | 約 200,000 R |
| horse-runs | 約 2,000,000 件 |
| episode 数 (B案、 開催単位) | 5,500 |
| step 数 (1 day 平均 24 R = 24 step) | 約 132,000 step / epoch |

### 3-2. PPO 学習 step 推定

| GPU | step/sec | 1M step 時間 | 10M step 時間 |
|-----|---------|-------------|--------------|
| RTX 3090 | ~500 | 約 33 min | **5.5 h** |
| RTX 4090 | ~800 | 約 21 min | 3.5 h |
| Colab A100 | ~700 | 約 24 min | 4 h |
| RTX 4080 | ~400 | 約 42 min | 7 h |

### 3-3. 必要 step 推定

PPO で stable converge には **10-30 M step** 必要 (Atari 系 benchmark 基準)。

→ 30 M step × RTX 3090 = **約 15-17 時間**
→ 複数 hyperparameter trial (5 回) で **2-3 日**

### 3-4. 結論

| 規模 | GPU | 期間 |
|------|-----|------|
| PoC (1 trial) | 既存 PC | 1 日 |
| 本学習 (5 trial) | RTX 3090 / Colab A100 | **2-3 日** |
| 拡張 (10 trial、 hyperparameter sweep) | Colab Pro+ | 5-7 日 |

---

## 4. インフラ コスト

| 項目 | 月額 | 備考 |
|------|------|------|
| Colab Pro+ (A100) | ¥5,500 | 10-12 月の 3 ヶ月のみ = ¥16,500 |
| AWS p3.2xlarge (option) | $3.06/h × 50h = ¥23,000 | spot で半額 ¥11,500 |
| RTX 3090 中古購入 (option) | 一時 10-15 万円 | 長期使用なら安い |

★ 推奨: **Colab Pro+ 3 ヶ月 ¥16,500** (V20/V21 月額 1 万円に上乗せ可) ★

---

## 5. 既存 環境 との 互換

| 項目 | 現状 | V22 対応 |
|------|------|---------|
| Python | 3.11 | OK |
| PyTorch | 2.11.0+cu126 | OK (sb3 は 2.0+ 必要) |
| CUDA | enabled | OK |
| RAM | 32 GB | OK (RL 学習も 32 GB で十分) |
| disk | (確認要) | 30 年 data 約 50 GB 想定 |

### 5-1. 追加導入

```bash
pip install stable-baselines3 gymnasium sb3-contrib wandb
```

既存 V20/V21 環境を **汚染しない** ため venv 推奨:

```bash
python -m venv ~/v22-rl-venv
~/v22-rl-venv/Scripts/activate
pip install -r requirements_v22.txt
```

---

## 6. monitoring + reproducibility

| tool | 用途 |
|------|------|
| tensorboard | reward / loss curve 可視化 |
| wandb | experiment tracking、 hyperparameter sweep |
| seed 固定 | numpy / torch / gym 全 seed 統一 |
| checkpoint | 1M step ごとに save (10M step で 10 file = 約 10 GB) |

---

## 7. 関連

- [V22_RL_DESIGN.md](V22_RL_DESIGN.md) — MDP + algorithm 選定
- [RL_VS_STRATEGY_COMPARISON.md](RL_VS_STRATEGY_COMPARISON.md) — paper 比較
- [V22_RISK_ANALYSIS.md](V22_RISK_ANALYSIS.md) — リスク + 撤退 logic
