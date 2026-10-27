import json,os
from pathlib import Path
import joblib,pandas as pd,streamlit as st,requests
st.set_page_config(page_title="Spotify Hit Predictor",page_icon="🎵",layout="wide")
st.title("Spotify Hit Predictor")
st.caption("Estimate the project's popularity-threshold label; this is not a guarantee of commercial success.")
mp=Path("models/hit_predictor.joblib")
if not mp.exists(): st.warning("Train first: python train.py --data data/spotify_tracks.csv"); st.stop()
b=joblib.load(mp); pipe=b["pipeline"]; features=b["features"]
metrics_path=Path("models/metrics.json")
if metrics_path.exists():
 with st.expander("Evaluation metrics"): st.json(json.loads(metrics_path.read_text()))
st.subheader("Track features"); vals={}; cols=st.columns(2)
for i,f in enumerate(features):
 with cols[i%2]: vals[f]=st.text_input(f,value="")
if st.button("Predict hit potential",type="primary"):
 row={}
 for f,v in vals.items():
  if not v.strip(): row[f]=None
  else:
   try: row[f]=float(v)
   except ValueError: row[f]=v
 try:
  x=pd.DataFrame([row],columns=features); pred=int(pipe.predict(x)[0]); prob=float(pipe.predict_proba(x)[0,1]); st.metric("Estimated hit probability",f"{prob:.1%}"); st.write("Prediction:","**Hit**" if pred else "**Not classified as a hit**"); st.caption(f"Hit definition: {b['target']} >= {b['threshold']}")
  key=os.getenv("OPENAI_API_KEY")
  if key and st.checkbox("Generate LLM explanation for this prediction"):
   prompt=f"Explain this classifier result cautiously. Do not claim causation or guarantee popularity; mention limitations. class={pred}, probability={prob:.4f}, features={json.dumps(row,default=str)}"
   r=requests.post("https://api.openai.com/v1/responses",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"model":os.getenv("OPENAI_MODEL","gpt-4.1-mini"),"input":prompt},timeout=45); r.raise_for_status(); st.info(r.json().get("output_text","No readable explanation returned."))
  elif not key: st.caption("Optional LLM explanation disabled; set OPENAI_API_KEY to enable it.")
 except Exception as e: st.error(f"Prediction failed ({type(e).__name__}). Check input values and retrain if needed.")
