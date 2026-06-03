# Ethereum Gas Price Prediction — COM4532

LSTM-based next-day Ethereum **gas price (Gwei)** forecasting, with XGBoost / ARIMA / SMA-7 / Naive
baselines and SHAP explainability. Implements the COM4532 project proposal.

- `gas_prediction.ipynb` — the complete pipeline (runs on Kaggle, Colab, or locally)
- `requirements.txt` — for local runs only (Kaggle/Colab already have these)
- `COM4532 - Proposal-*.pdf` — the project proposal
- `project_implementation_guide.docx` — the original implementation guide

The notebook **auto-detects** Kaggle / Colab / local and **auto-locates** the CSVs by filename
keyword, so you don't need to hardcode paths or exact filenames.

---

## 1. Get the data

✅ The 5 CSV files are **already downloaded** into `./data/`. (They cover 2015→present; the notebook
auto-filters to the proposal window.) The instructions below are only if you need to re-download them.

Download these **5 CSV files** from Etherscan's free public charts (top-right *Download: CSV Export* button).

| Feature | Etherscan chart |
|---|---|
| Average Gas Price | https://etherscan.io/chart/gasprice |
| Daily Gas Used | https://etherscan.io/chart/gasused |
| Average Gas Limit | https://etherscan.io/chart/gaslimit |
| Daily Transactions | https://etherscan.io/chart/tx |
| Ether Price (USD) | https://etherscan.io/chart/etherprice |

Filenames don't have to match exactly — the notebook finds each file by a keyword in its name
(`gasprice`, `gasused`, `gaslimit`, `tx`/`txgrowth`, `etherprice`).

> **Unit note:** Etherscan's gas-price export is occasionally in **Wei** instead of Gwei. Cell 2 prints
> the median value and warns if it looks like Wei — if so, divide `Gas_Price_Gwei` by `1e9`.

---

## 2. Run it

### Option A — Kaggle (recommended)
1. Go to https://kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook** and upload `gas_prediction.ipynb`.
3. Right panel **+ Add Input → Upload → Dataset**, and upload the 5 CSV files.
4. (Optional) Settings → Accelerator → **GPU** for faster LSTM training.
5. **Run All**. Outputs (PNGs, CSVs, models) are written to `/kaggle/working`.

### Option B — Google Colab
1. Open https://colab.research.google.com → **Upload** `gas_prediction.ipynb`.
2. Upload the 5 CSVs to the session (file panel) or mount Google Drive into `/content/drive/MyDrive`.
3. **Runtime → Run all**. Outputs go to the `outputs/` folder.

### Option C — Local
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the 5 CSVs in ./data/, then:
jupyter notebook gas_prediction.ipynb
```
> TensorFlow needs **Python 3.10–3.12**. Python 3.13 is not yet well supported — use Kaggle/Colab if you hit install errors.

---

## 3. Notebook cells (in order)

| Cell | Does |
|---|---|
| 0 | Detect env, locate the 5 CSVs |
| 1 | Imports (installs missing libs locally) |
| 2 | Load + merge CSVs (forward-fill gaps), derive utilization, **delta** & next-day target |
| 3 | EDA plots → `eda_overview.png` |
| 4 | **Winsorize @99th pct** (train-only), min-max scale (train-only fit), 30-day windows, 80/10/10 chronological split |
| 5 | **Lightweight grid search (lr × batch)** + train 2-layer LSTM → `lstm_loss.png` |
| 5b | **Walk-forward validation** (expanding window, periodic refit) |
| 6 | Baselines: Naive, SMA-7, ARIMA, XGBoost |
| 7 | Metrics table: MAE / RMSE / MAPE / R² |
| 8 | Prediction vs actual → `predictions.png` |
| 9 | SHAP: LSTM (DeepExplainer→Gradient→Kernel fallback) + XGBoost (TreeExplainer) → `shap_summary.png` |
| 10 | Save models + `predictions_all_models.csv`, `metrics.csv` |

---

## 4. After *Run All*, you should see
- Cell 2: a dataframe of ~878 rows × 10 columns (5 merged + utilization + delta + 2 day-of-week + target) and the correct date range.
- Cell 4: `Train / Val / Test` shapes printed — `(672, 30, 9)`, `(88, 30, 9)`, `(88, 30, 9)`.
- Cell 5: a grid-search log over `lr × batch`, the chosen hyperparameters, and a training/validation loss curve.
- Cell 5b: a walk-forward MAE/RMSE printed for comparison with the single-split result.
- Cell 7: a metrics table (MAE / RMSE / MAPE / R²) for LSTM + four baselines.
- Output files: `eda_overview.png`, `lstm_loss.png`, `predictions.png`, `shap_summary.png`,
  `shap_lstm_summary.png`, `predictions_all_models.csv`, `metrics.csv`, `lstm_gas_model.keras`, `xgb_gas_model.pkl`.

### Test-set metrics (last 10% — 4 Oct → 30 Dec 2023, 88 days)
Latest run with all 9 features (incl. the day-of-week encoding), ordered best→worst by R²:

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Naïve | 5.590 | 7.668 | 17.08% | 0.751 |
| SMA-7 | 7.868 | 10.376 | 25.84% | 0.549 |
| LSTM | 9.134 | 11.370 | 30.16% | 0.458 |
| ARIMA | 13.394 | 16.015 | 71.49% | −0.075 |
| XGBoost | 14.394 | 16.421 | 63.62% | −0.130 |

Adding the day-of-week feature improved the LSTM on every metric (MAE 9.550→9.134, RMSE
11.578→11.370, MAPE 35.12%→30.16%, R² 0.438→0.458), and SHAP ranks it as the strongest
non-price feature. The simple Naïve / SMA-7 baselines still lead — see the note below.

> **Note on results:** daily gas price is close to a random walk, so the Naïve baseline is a
> very strong reference. As the proposal itself states (Sec. 4.3), the comparison against
> Naïve / SMA-7 / ARIMA is the scientific contribution — report the metrics honestly rather
> than tuning for a particular ranking.

---

## 5. Alignment with the proposal
The notebook implements the proposal end to end:
- **9 input features:** the 6 in Sec. 2.3 (gas price, gas used, gas limit, utilization, tx count,
  ETH price) **+ the engineered day-over-day gas-price delta** and **the day-of-week temporal
  feature** (cyclically encoded as `DOW_Sin`/`DOW_Cos`), both from Sec. 3.1.
- **Winsorization @99th percentile** (Sec. 3.4), with caps computed on the training rows only
  (applied to the continuous features only; the bounded day-of-week encodings are left as-is).
- **No data leakage:** feature and target scalers are fit on the training rows only.
- **Forward-fill** of missing days (Sec. 3.4).
- **Lightweight grid search** over learning rate & batch size (Sec. 3.4), validated chronologically.
- **Walk-forward validation** (Sec. 4.2) via an expanding window with periodic refit (a tractable
  approximation of step-wise retraining; the refit interval is printed, not hidden).
- **SHAP DeepExplainer** for the LSTM (Sec. 3.3), with a `GradientExplainer` → `KernelExplainer`
  fallback for GPU/TF builds where the CuDNN LSTM kernel has no SHAP gradient, plus XGBoost TreeExplainer.

Implementation conveniences (not deviations): Wei→Gwei auto-conversion, keyword-based CSV
loading, MAPE guarded against near-zero prices, and the LSTM saved in the modern `.keras` format.
