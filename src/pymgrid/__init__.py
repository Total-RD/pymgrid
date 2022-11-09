from pathlib import Path

PROJECT_PATH = Path(__file__).parent

from .NonModularMicrogrid import NonModularMicrogrid
from .microgrid import ModularMicrogrid
from .MicrogridGenerator import MicrogridGenerator
