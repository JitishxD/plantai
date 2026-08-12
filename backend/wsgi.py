"""WSGI entrypoint for gunicorn / local imports.

    gunicorn --bind 0.0.0.0:5001 --timeout 180 --workers 2 wsgi:app
    python wsgi.py
"""

from __future__ import annotations
from plant_disease import config
from plant_disease.app import app

import sys
from pathlib import Path

# Repo root on path so `plant_disease` imports work
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=False)
