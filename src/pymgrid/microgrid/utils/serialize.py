import numpy as np
import yaml


def _numpy_representer_seq(dumper, data):
    return dumper.represent_sequence('!ndarray:', data.tolist())


def _numpy_represent_floating(dumper, data):
    return dumper.represent_float(data.item())


def _numpy_represent_int(dumper, data):
    return dumper.represent_int(data.item())


def _numpy_constructor_seq(loader, node):
    return np.array(loader.construct_sequence(node))


def add_numpy_representers():
    yaml.SafeDumper.add_representer(np.ndarray, _numpy_representer_seq)
    yaml.SafeDumper.add_multi_representer(np.floating, _numpy_represent_floating)
    yaml.SafeDumper.add_multi_representer(np.integer, _numpy_represent_int)


def add_numpy_constructors():
    yaml.SafeLoader.add_constructor('!ndarray:', _numpy_constructor_seq)
