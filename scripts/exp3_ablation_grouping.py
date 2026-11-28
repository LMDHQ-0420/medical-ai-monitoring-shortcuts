"""来源分解收集入口；不训练也不推理。"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.exp3.grouping import collect
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
print(collect(a.output_root,json.loads((a.output_root/'analysis_protocol.json').read_text())))
