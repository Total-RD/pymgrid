from pathlib import Path

PROJECT_PATH = Path(__file__).parent

__version__ = (PROJECT_PATH / "version.txt").read_text()

from ._deprecated.non_modular_microgrid import NonModularMicrogrid
from .microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator

