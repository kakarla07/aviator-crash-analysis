# Aviator Crash Game — Statistical & ML Analysis

> **Can machine learning predict the next crash multiplier in Aviator?**  
> I trained 3 different models on 300,000 rounds of real game data to find out.  
> **Spoiler: They all fail. Here's the proof.**

---

## Project Overview

Aviator is a crash game by Spribe used across dozens of betting platforms worldwide. Players bet on a multiplier that grows from 1x upward — and can crash at any moment. The question this project answers:

**Is there any statistical pattern in crash outcomes that a machine learning model could exploit?**

This project is not about gambling strategy. It is a rigorous data science investigation into randomness, RNG cryptography, and the limits of ML prediction on truly independent data.

---

## Dataset

- **Source:** [Kaggle — Aviator Payouts](https://www.kaggle.com/datasets/waruingugi/aviator)
- **Size:** 300,000 rounds
- **Platforms:** WINPESA, ODIBETS, BETIKA, BETGR8
- **Date range:** January 2024 – February 2024
- **Engine:** Spribe Aviator (provably fair, SHA-256 RNG)

---

## Files

```
aviator-crash-analysis/
│
├── aviator_analysis.py       # Statistical analysis (Part 1)
├── aviator_ml.py             # ML prediction models (Part 2)
├── aviator_analysis.png      # Charts — distribution, streaks, house edge
├── aviator_ml_results.png    # Charts — model comparison, residuals
└── README.md
```

---

## Part 1 — Statistical Analysis

### Key Findings

| Metric | Value |
|---|---|
| Total rounds | 300,000 |
| Median crash | 1.95x |
| Rounds below 2x | 51.2% |
| Rounds below 1.5x | 35.1% |
| Longest streak below 2x | 23 rounds |
| Autocorrelation (lag-1) | -0.000015 |
| Predictability score | 0.0015 / 100 |

### Autocorrelation — The Key Test

Autocorrelation measures whether the current round's result is influenced by previous rounds. A value near **0** means each round is independent — no pattern exists.

```
Lag-1:  -0.000015  ✓ No pattern
Lag-2:  -0.000008  ✓ No pattern
Lag-3:  -0.000015  ✓ No pattern
Lag-5:  -0.000013  ✓ No pattern
```

Every lag tested returned a value indistinguishable from zero.

### Distribution

```
1.00 – 1.50x    35.1%   ← Majority of rounds wipe out bets
1.50 – 2.00x    16.1%
2.00 – 3.00x    16.0%
3.00 – 5.00x    12.9%
5.00 – 10.0x     9.8%
10.0x+          10.1%
```

---

## Part 2 — ML Prediction Models

### Features Engineered

12 features per round, giving models maximum opportunity to find any signal:

- Lagged values: `lag_1`, `lag_2`, `lag_3`, `lag_5`, `lag_10`, `lag_20`
- Rolling statistics: `roll_mean_5`, `roll_mean_10`, `roll_std_5`, `roll_min_5`, `roll_max_5`
- Streak counter: `consec_below2`

### Models Trained

| Model | Architecture |
|---|---|
| Linear Regression | Standard OLS |
| Random Forest | 100 trees, max depth 8 |
| Neural Network | MLP — 128 → 64 → 32 → ReLU |

### Results

| Model | MAE | RMSE | R² | Beat baseline? |
|---|---|---|---|---|
| Baseline (always predict median) | 5.69 | 30.92 | -0.028 | — |
| Linear Regression | 8.33 | 30.51 | -0.000081 | ✗ No |
| Random Forest | 8.32 | 30.54 | -0.002162 | ✗ No |
| Neural Network (MLP) | 8.33 | 30.53 | -0.001933 | ✗ No |

**All three models perform worse than simply guessing the median every round.**

R² scores are negative and near zero — the models explain none of the variance in outcomes. This is not a failure of model design or hyperparameter tuning. It is the expected result when trying to fit a cryptographically random sequence.

### What a Useful Model Would Need

```
R² > 0.30   → Explains 30%+ of variance
MAE < 1.0   → Predictions within 1x of actual

Best achieved:
R²  = -0.000081   (explains nothing)
MAE =  8.32       (worse than guessing)
```

---

## Why ML Cannot Work Here

Spribe's Aviator uses a **provably fair** system:

1. Before each round, the server generates a hash: `SHA256(server_seed + client_seed + nonce)`
2. The crash multiplier is derived from this hash
3. The result is cryptographically determined **before** the round starts
4. Players can verify each result after the fact

This means:
- Each round is **mathematically independent** of all previous rounds
- Historical data carries **zero predictive signal** about future outcomes
- No model — regardless of complexity — can extract a pattern that does not exist

---

## Conclusion

> A Neural Network with 3 hidden layers, trained on 240,000 rounds of real crash data with 12 engineered features, could not beat a model that always guesses 1.95x.
>
> The game is not beatable through prediction. The math is working exactly as designed.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/kakarla07/aviator-crash-analysis
cd aviator-crash-analysis

# Install dependencies
pip install pandas numpy matplotlib scikit-learn

# Download dataset from Kaggle and place as:
# aviator_data/aviator_payouts.csv

# Run statistical analysis
python aviator_analysis.py

# Run ML models
python aviator_ml.py
```

---

## Tech Stack

`Python` `pandas` `NumPy` `scikit-learn` `Matplotlib`

---

## Author

**Arjun K.**  
Entrepreneur & Python Developer | Bangalore  
[GitHub](https://github.com/kakarla07) · [LinkedIn](https://linkedin.com/in/yourusername)
