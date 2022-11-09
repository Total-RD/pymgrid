import numpy as np
import pandas as pd
import yaml

from pathlib import Path

TO_CSV_TYPES = np.ndarray, pd.core.generic.NDFrame


def dump_data(data_dict, stream, yaml_tag):
    if not hasattr(stream, "name"):
        return data_dict

    path = Path(stream.name).parent / "data"
    return add_path_to_arr_like(data_dict, path, yaml_tag)


def add_path_to_arr_like(data_dict, path, yaml_tag):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = add_path_to_arr_like(value, path / key, yaml_tag)
        elif isinstance(value, TO_CSV_TYPES):
            if isinstance(value, np.ndarray):
                value = NDArraySubclass(value)
            value.path = path / f'{yaml_tag.lstrip("!")}/{key}.csv'
            data_dict[key] = value

    return data_dict


def _numpy_representer_seq(dumper, data):
    return dumper.represent_sequence('!ndarray', data.tolist())


def _numpy_represent_floating(dumper, data):
    return dumper.represent_float(data.item())


def _numpy_represent_int(dumper, data):
    return dumper.represent_int(data.item())


def _pandas_representer_df(dumper, data):
    return dumper.represent_mapping('!DataFrame', data.to_dict())


def _numpy_constructor_seq(loader, node):
    return np.array(loader.construct_sequence(node))


def _pandas_constructor_df(loader, node):
    return pd.DataFrame(loader.construct_mapping(node))


def add_numpy_pandas_representers():
    yaml.SafeDumper.add_representer(np.ndarray, _numpy_representer_seq)
    yaml.SafeDumper.add_representer(pd.DataFrame, _pandas_representer_df)
    yaml.SafeDumper.add_multi_representer(np.floating, _numpy_represent_floating)
    yaml.SafeDumper.add_multi_representer(np.integer, _numpy_represent_int)


def add_numpy_pandas_constructors():
    yaml.SafeLoader.add_constructor('!ndarray', _numpy_constructor_seq)
    yaml.SafeLoader.add_constructor('!DataFrame', _pandas_constructor_df)
