"""Generates one Colab notebook per chapter exercise (S3-S12) into
examples/chapter/notebooks/. Notebooks are derived from each section's
exercise.py; never edit them by hand -- re-run this script.

    python scripts/build_colab_notebooks.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "chapter"
OUT = CHAPTER / "notebooks"
# Placeholder until the code-release URL is fixed (see manuscript PHASEF-PENDING).
REPO_URL = "https://github.com/rimraul/slr-automation.git"

SETUP = f'''import os, subprocess, sys

REPO_URL = "{REPO_URL}"
if "google.colab" in sys.modules and not os.path.isdir("slr-automation"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
if os.path.isdir("slr-automation"):
    os.chdir("slr-automation")
sys.path.insert(0, os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "examples/chapter/requirements-colab.txt"], check=True)
print("Paruosta:", os.getcwd())'''

LIVE = '''# NEPRIVALOMA: tikri kreipiniai i DKM (sumoka uz naudojimą). Raktą laikykite
# Colab "Secrets" (rakto piktograma kairėje), niekada ne užrašykite kode.
# from google.colab import userdata
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")'''


def cell(kind: str, text: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def build(sec: Path) -> dict:
    src = (sec / "exercise.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or sec.name
    body = re.sub(r'^if __name__ == "__main__":\n\s+main\(\)\s*$', "main()", src, flags=re.M)
    body = re.sub(r'^""".*?"""\n', "", body, count=1, flags=re.S)
    # notebooks have no __file__; the setup cell chdirs to the repo root
    body = body.replace('Path(__file__).resolve().parents[1] / "data"', 'Path("examples/chapter") / "data"')
    sid = sec.name.split("_")[0].upper()
    ref = f"examples/chapter/{sec.name}"
    cells = [
        cell("markdown", f"# {sid} pratimas\n\n```\n{doc}\n```\n\n"
             "1. Paleiskite paruošimo langelį. 2. Užpildykite `TODO` funkcijas "
             "pratimo langelyje ir paleiskite jį. 3. Palyginkite su pavyzdiniu sprendimu."),
        cell("code", SETUP),
        cell("code", body),
        cell("markdown", "## Pavyzdinis sprendimas\n\nPažiūrėkite tik pabandę patys. "
             "Žemiau esantys testai tikrina **pavyzdinį sprendimą**, ne jūsų kodą."),
        cell("code", f"!cat {ref}/solution.py"),
        cell("code", f"!python -m pytest {ref} -q"),
        cell("code", LIVE),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for sec in sorted(CHAPTER.glob("s[0-9][0-9]_*")):
        if not (sec / "exercise.py").is_file():
            continue
        path = OUT / f"{sec.name}.ipynb"
        path.write_text(json.dumps(build(sec), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
