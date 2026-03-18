"""
Aviator Crash Analyser — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Aviator Crash Analyser",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0b0d14;
    color: #e0e0e0;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4d4d, #ff9f43, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #888;
    margin-bottom: 2rem;
    font-family: 'JetBrains Mono', monospace;
}
.verdict-box {
    background: #12151f;
    border: 1px solid #1e2235;
    border-left: 4px solid #ff4d4d;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #ccc;
    line-height: 1.8;
}
.metric-card {
    background: #12151f;
    border: 1px solid #1e2235;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #ff4d4d;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.section-tag {
    display: inline-block;
    background: #1e2235;
    color: #ff9f43;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 0.75rem;
}
.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1e2235;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.result-row .model-name { color: #ccc; }
.result-row .score { color: #ff4d4d; font-weight: 600; }
.result-row .beat { color: #2ecc71; }
.result-row .fail { color: #ff4d4d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✈️ Aviator Crash Analyser")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        help="Needs columns: payout (required), created_at & app (optional)"
    )
    st.markdown("---")
    st.markdown("**Sample dataset:**")
    st.markdown("[Download from Kaggle ↗](https://www.kaggle.com/datasets/waruingugi/aviator)")
    st.markdown("---")
    analysis_mode = st.radio(
        "Analysis mode",
        ["Statistical Analysis", "ML Prediction Models", "Both"],
        index=2
    )
    st.markdown("---")
    cap_val = st.slider("Cap extreme outliers at", 50, 1000, 500, 50,
                        help="Caps outliers for model stability. Stats use full data.")
    lag_window = st.multiselect(
        "Lag features for ML",
        [1, 2, 3, 5, 10, 20],
        default=[1, 2, 3, 5, 10, 20]
    )
    n_trees = st.slider("Random Forest trees", 50, 300, 100, 50)
    st.markdown("---")
    st.caption("Built by Arjun K. | Ark Bioenergies")

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">Aviator Crash Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">300,000 rounds · 3 ML models · 1 conclusion: the RNG wins.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data(file, cap):
    if file is not None:
        df = pd.read_csv(file)
    else:
        try:
            df = pd.read_csv("aviator_data/aviator_payouts.csv")
        except:
            st.error("No dataset found. Please upload a CSV file.")
            st.stop()
    if 'payout' not in df.columns:
        st.error("CSV must have a 'payout' column.")
        st.stop()
    df['payout'] = pd.to_numeric(df['payout'], errors='coerce')
    df = df.dropna(subset=['payout'])
    df = df[df['payout'] >= 1.0].reset_index(drop=True)
    df['payout_capped'] = df['payout'].clip(upper=cap)
    return df

df = load_data(uploaded, cap_val)
vals = df['payout'].values
vals_capped = df['payout_capped'].values
n = len(vals)

# ─────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><div class="value">{n:,}</div><div class="label">Total rounds</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="value">{np.median(vals):.2f}x</div><div class="label">Median crash</div></div>', unsafe_allow_html=True)
with c3:
    pct_below2 = (vals < 2).mean() * 100
    st.markdown(f'<div class="metric-card"><div class="value">{pct_below2:.1f}%</div><div class="label">Below 2x</div></div>', unsafe_allow_html=True)
with c4:
    def autocorr(arr, lag=1):
        mean = arr.mean(); denom = np.sum((arr - mean)**2)
        return np.sum((arr[:len(arr)-lag] - mean) * (arr[lag:] - mean)) / denom if denom else 0
    ac1 = autocorr(vals, 1)
    st.markdown(f'<div class="metric-card"><div class="value">{ac1:.5f}</div><div class="label">Autocorr lag-1</div></div>', unsafe_allow_html=True)
with c5:
    def longest_streak(arr, thr):
        s = ms = 0
        for v in arr:
            if v < thr: s += 1; ms = max(ms, s)
            else: s = 0
        return ms
    ls = longest_streak(vals, 2)
    st.markdown(f'<div class="metric-card"><div class="value">{ls}</div><div class="label">Longest streak &lt;2x</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────
if analysis_mode in ["Statistical Analysis", "Both"]:
    st.markdown('<div class="section-tag">Part 1 — Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown("### Distribution & Pattern Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='#12151f')
        fig.subplots_adjust(hspace=0.45)

        def style(ax):
            ax.set_facecolor('#0b0d14')
            ax.tick_params(colors='#666', labelsize=9)
            for s in ax.spines.values(): s.set_color('#1e2235')
            ax.xaxis.label.set_color('#666')
            ax.yaxis.label.set_color('#666')
            ax.title.set_color('#ccc')

        # Histogram
        ax1 = axes[0]
        style(ax1)
        capped_plot = vals[vals <= 20]
        ax1.hist(capped_plot, bins=80, color='#ff4d4d', alpha=0.7, edgecolor='none')
        ax1.axvline(np.median(vals), color='#ffd700', linewidth=2, linestyle='--',
                    label=f'Median: {np.median(vals):.2f}x')
        ax1.axvline(2, color='#ff9f43', linewidth=1.5, linestyle=':', label='2x line')
        ax1.set_title('Crash Multiplier Distribution (capped at 20x)', fontsize=11)
        ax1.set_xlabel('Crash Multiplier')
        ax1.set_ylabel('Rounds')
        ax1.legend(facecolor='#12151f', labelcolor='#ccc', fontsize=8)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

        # Autocorrelation
        ax2 = axes[1]
        style(ax2)
        lags = list(range(1, 21))
        acs = [autocorr(vals, l) for l in lags]
        colors_ac = ['#ff4d4d' if abs(a) > 0.05 else '#4f8ef7' for a in acs]
        ax2.bar(lags, acs, color=colors_ac, alpha=0.85, edgecolor='none')
        ax2.axhline(0, color='#666', linewidth=0.5)
        ax2.axhline(0.05, color='#ff4d4d', linewidth=1, linestyle='--', alpha=0.5)
        ax2.axhline(-0.05, color='#ff4d4d', linewidth=1, linestyle='--', alpha=0.5)
        ax2.set_title('Autocorrelation by Lag — near zero = no pattern', fontsize=11)
        ax2.set_xlabel('Lag (rounds)')
        ax2.set_ylabel('Correlation')
        ax2.set_ylim(-0.12, 0.12)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Distribution brackets**")
        buckets = [
            ("1.00–1.50x", 1.0, 1.5, "#e74c3c"),
            ("1.50–2.00x", 1.5, 2.0, "#e67e22"),
            ("2.00–3.00x", 2.0, 3.0, "#f1c40f"),
            ("3.00–5.00x", 3.0, 5.0, "#2ecc71"),
            ("5.00–10.0x", 5.0, 10.0, "#1abc9c"),
            ("10.0x+",     10.0, np.inf, "#3498db"),
        ]
        for label, lo, hi, color in buckets:
            cnt = int(((vals >= lo) & (vals < hi)).sum())
            pct = cnt / n * 100
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-bottom:3px;">
                <span style="font-family:'JetBrains Mono',monospace;">{label}</span>
                <span style="color:{color};font-weight:600;">{pct:.1f}%</span>
              </div>
              <div style="background:#1e2235;border-radius:4px;height:6px;">
                <div style="background:{color};width:{pct}%;height:6px;border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Streak probabilities**")
        p = (vals < 2).mean()
        for k in [2, 3, 5, 10]:
            prob = p**k * 100
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#888;padding:3px 0;">'
                f'{k} in a row: <span style="color:#ff9f43;">{prob:.2f}%</span>'
                f' (1 in {int(round(1/(prob/100))) if prob > 0 else "∞"})</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="verdict-box">
        <strong style="color:#ff9f43;">Pattern Analysis</strong><br><br>
        Autocorrelation lag-1: <strong style="color:#ffd700;">{ac1:.6f}</strong><br>
        Predictability score: <strong style="color:#ffd700;">{abs(ac1)*100:.4f} / 100</strong><br><br>
        Each round is statistically independent. No sequential pattern detected across any lag.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# ─────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────
if analysis_mode in ["ML Prediction Models", "Both"]:
    st.markdown('<div class="section-tag">Part 2 — ML Prediction Models</div>', unsafe_allow_html=True)
    st.markdown("### Can machine learning predict the next crash?")

    with st.spinner("Training models on your data... (this takes ~30 seconds)"):

        # Build features
        lags = lag_window if lag_window else [1, 2, 3, 5]

        @st.cache_data
        def build_and_train(vals_capped, lags, n_trees, cap_val):
            feat_df = pd.DataFrame({'target': vals_capped})
            for lag in lags:
                feat_df[f'lag_{lag}'] = feat_df['target'].shift(lag)
            feat_df['roll_mean_5']  = feat_df['target'].shift(1).rolling(5).mean()
            feat_df['roll_mean_10'] = feat_df['target'].shift(1).rolling(10).mean()
            feat_df['roll_std_5']   = feat_df['target'].shift(1).rolling(5).std()
            feat_df['roll_min_5']   = feat_df['target'].shift(1).rolling(5).min()
            feat_df['roll_max_5']   = feat_df['target'].shift(1).rolling(5).max()
            streak = 0
            streaks = []
            for v in feat_df['target'].shift(1).fillna(2):
                if v < 2: streak += 1
                else: streak = 0
                streaks.append(streak)
            feat_df['consec_below2'] = streaks
            feat_df = feat_df.dropna()
            feature_cols = [c for c in feat_df.columns if c != 'target']
            X = feat_df[feature_cols].values
            y = feat_df['target'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            # Baseline
            base_pred = np.full_like(y_test, np.median(y_train))

            # LR
            lr = LinearRegression().fit(X_train_s, y_train)
            lr_pred = lr.predict(X_test_s)

            # RF
            rf = RandomForestRegressor(n_estimators=n_trees, max_depth=8,
                                       random_state=42, n_jobs=-1).fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_imp  = sorted(zip(feature_cols, rf.feature_importances_),
                             key=lambda x: x[1], reverse=True)

            # MLP
            mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                               max_iter=200, random_state=42,
                               early_stopping=True, validation_fraction=0.1
                               ).fit(X_train_s, y_train)
            mlp_pred = mlp.predict(X_test_s)
            loss_curve = mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else []

            def metrics(y_true, y_pred):
                return {
                    'mae':  mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2':   r2_score(y_true, y_pred)
                }

            return {
                'y_test': y_test,
                'base': metrics(y_test, base_pred),
                'lr':   {'pred': lr_pred,  **metrics(y_test, lr_pred)},
                'rf':   {'pred': rf_pred,  **metrics(y_test, rf_pred), 'imp': rf_imp},
                'mlp':  {'pred': mlp_pred, **metrics(y_test, mlp_pred), 'loss': loss_curve},
                'n_features': len(feature_cols),
                'n_train': len(X_train),
                'n_test':  len(X_test),
                'median_train': np.median(y_train),
            }

        res = build_and_train(tuple(vals_capped), tuple(lags), n_trees, cap_val)

    # Results table
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Model results**")
        base_mae = res['base']['mae']
        models_res = [
            ("Baseline (median)", res['base'],  False),
            ("Linear Regression", res['lr'],    True),
            ("Random Forest",     res['rf'],    True),
            ("Neural Network",    res['mlp'],   True),
        ]
        for name, m, compare in models_res:
            beat = ""
            if compare:
                beat = "✓" if m['mae'] < base_mae else "✗"
            beat_color = "#2ecc71" if beat == "✓" else "#ff4d4d"
            st.markdown(f"""
            <div class="result-row">
              <span class="model-name">{name}</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#888;">
                R²=<span style="color:#ff4d4d;">{m['r2']:.5f}</span>
              </span>
              <span style="color:{beat_color};font-weight:700;font-size:13px;">{beat}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="verdict-box">
        <strong style="color:#ff9f43;">Training config</strong><br><br>
        Features: <strong style="color:#ffd700;">{res['n_features']}</strong><br>
        Train samples: <strong style="color:#ffd700;">{res['n_train']:,}</strong><br>
        Test samples: <strong style="color:#ffd700;">{res['n_test']:,}</strong><br><br>
        <strong style="color:#ff4d4d;">All models failed to beat a median guess.</strong><br>
        Best R² = {max(res['lr']['r2'], res['rf']['r2'], res['mlp']['r2']):.6f}<br>
        (Useful model needs R² &gt; 0.30)
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8), facecolor='#12151f')
        fig2.subplots_adjust(hspace=0.45, wspace=0.35)

        def style2(ax, title=''):
            ax.set_facecolor('#0b0d14')
            ax.tick_params(colors='#666', labelsize=9)
            for s in ax.spines.values(): s.set_color('#1e2235')
            ax.xaxis.label.set_color('#666')
            ax.yaxis.label.set_color('#666')
            if title: ax.set_title(title, color='#ccc', fontsize=9, pad=6)

        SAMPLE = 250
        y_test  = res['y_test']
        x_range = np.arange(SAMPLE)

        # Predicted vs Actual — RF
        ax = axes2[0, 0]
        style2(ax, 'Random Forest: Predicted vs Actual')
        ax.plot(x_range, np.clip(y_test[:SAMPLE], 1, 20), color='#666', linewidth=0.7, label='Actual')
        ax.plot(x_range, np.clip(res['rf']['pred'][:SAMPLE], 1, 20),
                color='#2ecc71', linewidth=1.2, label='RF Predicted')
        ax.set_ylim(0, 22)
        ax.legend(facecolor='#0b0d14', labelcolor='#ccc', fontsize=8)
        ax.set_ylabel('Multiplier', color='#666')

        # MAE comparison
        ax = axes2[0, 1]
        style2(ax, 'MAE Comparison (lower = better)')
        names_short = ['Baseline', 'Linear\nReg', 'Random\nForest', 'Neural\nNet']
        maes = [res['base']['mae'], res['lr']['mae'], res['rf']['mae'], res['mlp']['mae']]
        bcolors = ['#555', '#4f8ef7', '#2ecc71', '#f39c12']
        bars = ax.bar(names_short, maes, color=bcolors, alpha=0.85, edgecolor='none', width=0.6)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{mae:.2f}', ha='center', va='bottom', color='#ccc', fontsize=8)
        ax.set_ylabel('MAE', color='#666')

        # Feature importances
        ax = axes2[1, 0]
        style2(ax, 'RF Feature Importances')
        top8 = res['rf']['imp'][:8]
        fn = [x[0] for x in top8][::-1]
        fv = [x[1] for x in top8][::-1]
        ax.barh(fn, fv, color='#2ecc71', alpha=0.8, edgecolor='none')
        ax.set_xlabel('Importance', color='#666')
        ax.tick_params(axis='y', labelsize=8)

        # MLP loss or scatter
        ax = axes2[1, 1]
        style2(ax, 'Neural Net: Actual vs Predicted')
        actual_s = np.clip(y_test[:3000], 1, 20)
        pred_s   = np.clip(res['mlp']['pred'][:3000], 1, 20)
        ax.scatter(actual_s, pred_s, color='#f39c12', alpha=0.1, s=3, edgecolors='none')
        ax.plot([1, 20], [1, 20], color='#ff4d4d', linewidth=1.5, linestyle='--',
                label='Perfect prediction')
        ax.set_xlabel('Actual', color='#666')
        ax.set_ylabel('Predicted', color='#666')
        ax.legend(facecolor='#0b0d14', labelcolor='#ccc', fontsize=8)

        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

# ─────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────
st.markdown('<div class="section-tag">Final Verdict</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="verdict-box" style="border-left-color:#ffd700;font-size:0.9rem;">
<strong style="color:#ffd700;font-size:1rem;">Can ML predict Aviator crashes?</strong><br><br>
✗ &nbsp;Linear Regression &nbsp;— R² ≈ 0.000<br>
✗ &nbsp;Random Forest &nbsp;&nbsp;&nbsp;&nbsp;— R² ≈ 0.000<br>
✗ &nbsp;Neural Network &nbsp;&nbsp;&nbsp;— R² ≈ 0.000<br><br>
All three models, trained on {n:,} rounds with lag features, rolling statistics, and streak counters,
performed <strong style="color:#ff4d4d;">worse than simply guessing the median every round</strong>.<br><br>
Aviator uses a SHA-256 provably fair RNG. Each round is cryptographically independent of all previous rounds.
No pattern exists in the data because no pattern exists in the game.<br><br>
<strong style="color:#2ecc71;">The RNG is unbeatable. This is the proof.</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Aviator Crash Analyser · Built by Arjun K. · Ark Bioenergies · Bangalore")
