from pathlib import Path

PROJECT_PATH = Path(__file__).parent

from .non_modular_microgrid import NonModularMicrogrid
from .microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator
