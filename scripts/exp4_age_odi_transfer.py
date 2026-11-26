"""exp4：复用同一批正式权重，统一生成年龄与外部验证结果。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.exp5.run_external import main

if __name__=='__main__':main()
