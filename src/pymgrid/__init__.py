from pathlib import Path

PROJECT_PATH = Path(__file__).parent
try:
    __version__ = (PROJECT_PATH / "version.txt").read_text()
except FileNotFoundError:
    file = PROJECT_PATH / "version.txt"
    nl ='\n'
    err_msg = f'version.txt file not found at {file}.\n' \
              f'Contents of parent dir:\n{nl.join([str(x) for x in sorted(file.parent.iterdir())])}'
    raise FileNotFoundError(err_msg)

from ._deprecated.non_modular_microgrid import NonModularMicrogrid
from .microgrid import Microgrid
from .MicrogridGenerator import MicrogridGenerator

