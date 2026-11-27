#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import BUILDERS
from style import export
export(BUILDERS["fig2d"](), Path(__file__).resolve().parent, "fig2d")
