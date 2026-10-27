import argparse,json
from spotify_pipeline import load_dataset,train_model
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--target",default="track_popularity"); p.add_argument("--threshold",type=float,default=70); p.add_argument("--output-dir",default="models"); a=p.parse_args(); print(json.dumps(train_model(load_dataset(a.data),a.target,a.threshold,a.output_dir),indent=2))
if __name__=="__main__": main()
