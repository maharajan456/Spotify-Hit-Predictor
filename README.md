# Project  — Spotify Hit Predictor

## What it does
A tabular ML pipeline that reads a CSV, handles missing values, encodes categorical variables, scales numeric variables, optionally applies SMOTE within the training pipeline, trains a Random Forest classifier, and saves evaluation metrics plus a model artifact. The Streamlit app uses the saved model to estimate whether a track crosses the project's popularity threshold.

The target label is a project-defined proxy (`track_popularity >= 70`), not an official Spotify hit definition. Model predictions are not guarantees of commercial success.

## Included dataset
`data/synthetic_spotify_tracks.csv` contains 600 randomly generated Spotify-like records. It is synthetic and does not represent real tracks, artists, or Spotify data. Dataset details are in `data/README.md`.

## Run
From this directory:

```bash
pip install -r requirements.txt
python train.py --data data/synthetic_spotify_tracks.csv --target track_popularity --threshold 70
streamlit run app.py
```

The training command writes `models/hit_predictor.joblib` and `models/metrics.json`. A copy of the tested metrics is included at `results/synthetic_training_metrics.json`.

## Test results for the included synthetic dataset
- Rows: **600**
- Features used: **10**
- Positive class (`track_popularity >= 70`): **69 rows**
- Test split: **20%** (120 rows), random state **42**
- Accuracy: **0.8917**
- ROC-AUC: **0.8652**
- Positive-class precision: **0.5455**
- Positive-class recall: **0.4286**
- Positive-class F1: **0.4800**
- Confusion matrix (actual rows, predicted columns): `[[101, 5], [8, 6]]`
- SMOTE: **used on training data only**

These are reproducible smoke-test metrics on fabricated data, not evidence of real-world performance. Class imbalance means accuracy alone is misleading; inspect positive-class recall and precision too. Do not claim these values as the original resume result or as a result on real Spotify data.

## Optional LLM explanation
Set `OPENAI_API_KEY` in your environment to enable the optional explanation feature in the app. Predictions work without an API key. Treat generated text as a cautious explanation, not causal proof.
