"""Render the available tracked panels and their separate legends."""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from figures import BUILDERS, LEGEND_SPECS
from style import export, legend_figure

ROOT = Path(__file__).resolve().parent


def main():
    for panel, builder in BUILDERS.items():
        outdir = ROOT / panel
        if not outdir.is_dir():
            continue
        spec = yaml.safe_load((outdir / "figure.yaml").read_text())
        print(f"render {panel}", flush=True)
        export(builder(), outdir, panel, dpi=spec["png_dpi"])
        if panel in LEGEND_SPECS:
            legend = LEGEND_SPECS[panel]
            export(legend_figure(legend["handles"], legend["width"],
                                 legend["height"], legend["ncol"]),
                   outdir, f"{panel}-legend", dpi=spec["png_dpi"])
        plt.close("all")


if __name__ == "__main__":
    main()
