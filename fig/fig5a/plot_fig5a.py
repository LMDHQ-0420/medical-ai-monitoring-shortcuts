#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import BUILDERS, LEGEND_SPECS
from style import export, legend_figure
export(BUILDERS["fig5a"](), Path(__file__).resolve().parent, "fig5a")
ls=LEGEND_SPECS["fig5a"]
export(legend_figure(ls["handles"],ls["width"],ls["height"],ls["ncol"]), Path(__file__).resolve().parent, "fig5a-legend")
