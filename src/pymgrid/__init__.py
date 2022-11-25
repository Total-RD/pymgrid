from pathlib import Path

PROJECT_PATH = Path(__file__).parent
try:
    __version__ = (PROJECT_PATH / "version.txt").read_text()
except FileNotFoundError:
    raise FileNotFoundError(PROJECT_PATH / "version.txt")

from ._deprecated.non_modular_microgrid import NonModularMicrogrid
from .microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator

