from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
DROP_HINTS=("id","uri","url","track_name","artist_name","album_name","track_id")
def load_dataset(path):
    df=pd.read_csv(path)
    if df.empty: raise ValueError("Dataset is empty")
    return df
def train_model(df,target,threshold,out_dir):
    if target not in df.columns: raise ValueError(f"Target {target!r} not found; columns={list(df.columns)}")
    target_values=pd.to_numeric(df[target],errors="coerce")
    valid=target_values.notna()
    df=df.loc[valid].copy(); target_values=target_values.loc[valid]
    y=(target_values>=threshold).astype(int)
    features=[c for c in df.columns if c!=target and not any(h in c.lower() for h in DROP_HINTS)]
    X=df[features].copy()
    if y.nunique()<2: raise ValueError("Threshold produced fewer than two classes; adjust --threshold")
    nums=X.select_dtypes(include=["number","bool"]).columns.tolist(); cats=[c for c in X.columns if c not in nums]; tx=[]
    if nums: tx.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats: tx.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats))
    prep=ColumnTransformer(tx)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    steps=[("prep",prep)]
    if int(ytr.value_counts().min())>5: steps.append(("smote",SMOTE(random_state=42)))
    steps.append(("model",RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1)))
    pipe=ImbPipeline(steps); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    metrics={"target_column":target,"hit_threshold":threshold,"n_rows":len(df),"n_features":len(features),"features":features,"class_distribution":{str(k):int(v) for k,v in y.value_counts().items()},"accuracy":float(accuracy_score(yte,pred)),"roc_auc":float(roc_auc_score(yte,prob)),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"classification_report":classification_report(yte,pred,output_dict=True,zero_division=0),"test_size":.2,"random_state":42,"smote_used":len(steps)>2,"note":"Metrics depend on dataset and threshold; reproduce before reporting."}
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); joblib.dump({"pipeline":pipe,"features":features,"target":target,"threshold":threshold},out/"hit_predictor.joblib"); (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); return metrics
