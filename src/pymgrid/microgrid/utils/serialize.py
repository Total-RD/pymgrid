import numpy as np
import pandas as pd
import yaml

from pathlib import Path

TO_CSV_TYPES = np.ndarray, pd.core.generic.NDFrame


def _dump_data(data, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path)
    return str(path)


def dump_csvs(data_dict, stream, yaml_tag):
    if not hasattr(stream, "name"):
        return data_dict

    path = Path(stream.name).parent / "data"

    for data_type, data in data_dict.items():
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, TO_CSV_TYPES):
                    data[key] = _dump_data(val, path / f"{data_type}/{yaml_tag.lstrip('!')}_{key}.csv")

        elif isinstance(data, TO_CSV_TYPES):
            data_dict[data_type] = _dump_data(data, path / f"{yaml_tag.strip('!')}_{data_type}.csv")

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
