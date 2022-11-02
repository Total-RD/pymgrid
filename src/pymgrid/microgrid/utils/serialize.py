import numpy as np
import yaml


def _numpy_representer_seq(dumper, data):
    return dumper.represent_sequence('!ndarray:', data.tolist())


def _numpy_represent_floating(dumper, data):
    return dumper.represent_float(data.item())


def _numpy_represent_int(dumper, data):
    return dumper.represent_int(data.item())


def add_numpy_representers():
    yaml.SafeDumper.add_representer(np.ndarray, _numpy_representer_seq)
    yaml.SafeDumper.add_multi_representer(np.floating, _numpy_represent_floating)
    yaml.SafeDumper.add_multi_representer(np.integer, _numpy_represent_int)

