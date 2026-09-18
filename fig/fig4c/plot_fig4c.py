#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import BUILDERS, LEGEND_SPECS
from style import export, legend_figure
export(BUILDERS["fig4c"](), Path(__file__).resolve().parent, "fig4c")
ls=LEGEND_SPECS["fig4c"]
export(legend_figure(ls["handles"],ls["width"],ls["height"],ls["ncol"]), Path(__file__).resolve().parent, "fig4c-legend")
