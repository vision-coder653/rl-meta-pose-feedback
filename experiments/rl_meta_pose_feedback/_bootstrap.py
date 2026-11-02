"""Enable direct CLI execution without changing another experiment's imports."""
from pathlib import Path
import sys

def activate():
    app = Path(__file__).resolve().parents[2]
    for path in (app, app / 'core', app / 'experiments'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
