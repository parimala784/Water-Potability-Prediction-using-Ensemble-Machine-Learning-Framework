"""
Run all notebook code cells in order. Generates model folder and .pkl files.
Uses nbformat to execute notebooks headless (no Jupyter kernel).
"""
import os
import sys
from pathlib import Path

# Run from backend/ so notebook paths like ../data and ../models resolve
BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nbformat
except ImportError:
    print("Installing nbformat...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat", "-q"])
    import nbformat

NOTEBOOKS_DIR = BACKEND_DIR / "notebooks"
OUTPUTS_DIR = BACKEND_DIR / "notebook_outputs"
NOTEBOOK_ORDER = ["01_eda.ipynb", "02_preprocessing.ipynb", "03_visualization.ipynb", "04_model_building.ipynb"]


def run_notebook(notebook_path: Path, globals_dict: dict) -> dict:
    """Execute all code cells of a notebook; return updated globals."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if not source.strip():
            continue
        try:
            exec(compile(source, f"<{notebook_path.name}>", "exec"), globals_dict)
        except Exception as e:
            print(f"  [ERROR in {notebook_path.name}]: {e}")
            raise
    return globals_dict


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "models").mkdir(parents=True, exist_ok=True)

    # Redirect plt.show to save to output dir (headless)
    _fig_counter = [0]
    def save_fig_instead():
        if plt.get_fignums():
            _fig_counter[0] += 1
            plt.gcf().savefig(OUTPUTS_DIR / f"fig_{_fig_counter[0]:03d}.png", dpi=100)
        plt.close("all")

    os.chdir(NOTEBOOKS_DIR)
    g: dict = {
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "plt": plt,
        "sns": __import__("seaborn"),
        "Path": Path,
        "OUTPUTS_DIR": OUTPUTS_DIR,
    }
    g["plt"].show = save_fig_instead

    # Patch Plotly so fig.show() saves HTML instead of opening browser (avoids hang)
    try:
        import plotly.graph_objects as go
        _out = OUTPUTS_DIR / "plotly_3d.html"
        def _show_write_html(self):
            self.write_html(str(_out))
        go.Figure.show = _show_write_html
    except ImportError:
        pass

    for name in NOTEBOOK_ORDER:
        path = NOTEBOOKS_DIR / name
        if not path.exists():
            print(f"Skipping (not found): {name}")
            continue
        print(f"Running {name}...")
        run_notebook(path, g)
        print(f"  Done: {name}")

    # Ensure model .pkl files exist (notebook 04 writes them; if we skipped, run train)
    models_dir = BACKEND_DIR / "models"
    if not (models_dir / "water_quality_model.pkl").exists():
        print("Model not found; running train.py...")
        os.chdir(BACKEND_DIR)
        import train
        train.main()

    print("\nAll notebooks run. Model folder:", models_dir)
    print("Output figures:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
