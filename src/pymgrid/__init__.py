from pathlib import Path
try:
    __version__ = (Path(__file__).parent.parent.parent / "version.txt").read_text()
except FileNotFoundError:
    raise FileNotFoundError(f'Failed with __init__ path: {Path(__file__)}')

PROJECT_PATH = Path(__file__).parent

from ._deprecated.non_modular_microgrid import NonModularMicrogrid
from .microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator

