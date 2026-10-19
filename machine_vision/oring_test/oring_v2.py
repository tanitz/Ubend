"""Shim: the detector lives in app/oring_v2.py (single source for the app and for these offline tools)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import app.oring_v2 as _impl  # noqa: E402

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})   # incl. the private helpers qa_zoom.py uses
