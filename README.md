# ctd-fusionnet-deepfake-detection
Deepfake Detection using CTD Fusion Net

## Environment setup (uv)

1. Install [uv](https://docs.astral.sh/uv/) if it is not already available:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. From the repo root, let uv create the `.venv` folder and install everything declared in `pyproject.toml` (Python 3.10):
   ```bash
   uv sync --python 3.10
   ```
3. Activate the environment whenever you want to run `main.ipynb` locally:
   ```bash
   source .venv/bin/activate
   ```
4. (Optional, for VS Code / Jupyter) register the kernel:
   ```bash
   .venv/bin/python -m ipykernel install --user --name ctd-fusionnet --display-name "ctd-fusionnet (uv)"
   ```

You can now start JupyterLab or another notebook runner, select the `ctd-fusionnet (uv)` kernel, and execute the cells in `main.ipynb`.
