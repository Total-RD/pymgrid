from pathlib import Path

PROJECT_PATH = Path(__file__).parent

from .non_modular_microgrid import NonModularMicrogrid
from .microgrid import ModularMicrogrid
from .MicrogridGenerator import MicrogridGenerator
