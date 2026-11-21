"""Execute both v2 evidence notebooks and collect their displayed JSON results."""

import json
import os
import platform
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("encoding-quality.ipynb", "decoding.ipynb")
COMMAND = "uv run --no-sync python notebooks/run.py"


def main():
    report = {"schema_version": 1, "reproduce": COMMAND, "runtime": {
        "python": platform.python_version(), "machine": platform.machine(),
        "os": platform.system(),
        **{p: version(p) for p in ("esdm", "torch", "numpy", "scipy", "nbformat", "nbclient",
                                    "ipykernel", "matplotlib")},
    }}
    with TemporaryDirectory() as scratch:
        # Fresh kernel home/config paths and a minimal child process environment.
        kernel_env = {
            "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.defpath)),
            "HOME": scratch, "TMPDIR": scratch, "PYTHONNOUSERSITE": "1",
            "IPYTHONDIR": f"{scratch}/ipython", "MPLCONFIGDIR": f"{scratch}/matplotlib",
            "JUPYTER_RUNTIME_DIR": f"{scratch}/jupyter", "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            # IPython's inline backend embeds figures in the notebook.
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        }
        for filename in NOTEBOOKS:
            path = ROOT / "notebooks" / filename
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            client = NotebookClient(notebook, kernel_name="python3", timeout=120,
                                    record_timing=False)
            manager = client.create_kernel_manager()
            # Match the runtime receipt even if a user-level python3 spec points elsewhere.
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher",
                                        "-f", "{connection_file}"]
            if os.name == "posix":
                manager.transport = "ipc"
                manager.ip = f"{scratch}/kernel"
            client.execute(cwd=str(ROOT), env=kernel_env)
            cells = [c for c in notebook.cells if "result-json" in c.metadata.get("tags", [])]
            payload, = [out.data["application/json"] for c in cells for out in c.outputs
                        if "application/json" in out.get("data", {})]
            report.update(payload)
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f"Executed notebooks/{filename}")
    sources = [f"notebooks/{f}" for f in NOTEBOOKS]
    sources += ["notebooks/run.py", "esdm/experiments/v2_readout.py",
                "esdm/models/bounded_readout.py", "esdm/models/esdm.py",
                "esdm/models/clock.py", "esdm/models/vibrational.py",
                "esdm/models/diffusion_decoder.py", "esdm/models/elbo.py",
                "esdm/_configs/default.yaml", "uv.lock"]
    report["sha256"] = {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    path = ROOT / "docs/paper/v2-results/experiment.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Collected {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
