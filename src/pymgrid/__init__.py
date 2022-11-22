from pathlib import Path

__version__ = (Path(__file__).parent.parent.parent / "version.txt").read_text()

PROJECT_PATH = Path(__file__).parent

from ._deprecated.non_modular_microgrid import NonModularMicrogrid
from .modular_microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator

