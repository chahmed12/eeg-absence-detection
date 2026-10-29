"""Orchestrateur : exécute la chaîne de recalcul complète, étape par étape,
en journalisant dans ``cache/pipeline.log``. Lancé en arrière-plan.

Usage : python run_pipeline.py            (toutes les étapes)
        python run_pipeline.py lopo       (à partir d'une étape donnée)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")

STEPS = [
    ("lopo",           ["scripts/scripts_step3_lopo.py"]),
    ("models_explain", ["scripts/scripts_step5_models_explain.py"]),
    ("detection",      ["scripts/scripts_step4_detection.py", "--force"]),
    ("improvements",   ["scripts/scripts_step5_improvements.py"]),
    ("window",         ["scripts/scripts_step5_window_sensitivity.py"]),
    ("band",           ["scripts/scripts_step5_band_analysis.py"]),
]


def main(start: str | None = None) -> None:
    steps = STEPS
    if start:
        names = [s[0] for s in STEPS]
        steps = STEPS[names.index(start):]
    for name, args in steps:
        print(f"\n===== [{time.strftime('%H:%M:%S')}] ÉTAPE {name} : {' '.join(args)} =====",
              flush=True)
        t0 = time.time()
        script, *extra = args
        r = subprocess.run([PY, script, *extra], cwd=str(PROJ))
        print(f"===== [{time.strftime('%H:%M:%S')}] {name} terminé en "
              f"{time.time()-t0:.0f}s (code {r.returncode}) =====", flush=True)
        if r.returncode != 0:
            print(f"!! Échec de l'étape {name} (code {r.returncode})", flush=True)
    print("\nPIPELINE COMPLET.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
