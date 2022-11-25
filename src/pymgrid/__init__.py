from pathlib import Path
from .version import __version__

PROJECT_PATH = Path(__file__).parent

from ._deprecated.non_modular_microgrid import NonModularMicrogrid
from .microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator
