import numpy as np
import pandas as pd
import yaml

from pathlib import Path

TO_CSV_TYPES = np.ndarray, pd.core.generic.NDFrame


def add_pymgrid_yaml_representers():
    add_numpy_pandas_representers()
    from pymgrid.microgrid.trajectory import (
        DeterministicTrajectory,
        StochasticTrajectory,
        FixedLengthStochasticTrajectory
    )

    from pymgrid.microgrid.reward_shaping import (
        PVCurtailmentShaper,
        BatteryDischargeShaper
    )

    from pymgrid.modules.battery.transition_models import (
        BatteryTransitionModel,
        BiasedTransitionModel,
        DecayTransitionModel
    )


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
            value.path = path / f'{yaml_tag.lstrip("!")}/{key}.csv.gz'
            data_dict[key] = value

    return data_dict


def add_numpy_pandas_representers():
    yaml.SafeDumper.add_representer(pd.DataFrame, _pandas_df_representer)
    yaml.SafeDumper.add_multi_representer(np.ndarray, _numpy_arr_representer)
    yaml.SafeDumper.add_multi_representer(np.floating, _numpy_represent_floating)
    yaml.SafeDumper.add_multi_representer(np.integer, _numpy_represent_int)


def add_numpy_pandas_constructors():
    yaml.SafeLoader.add_constructor('!NDArray', _numpy_arr_constructor)
    yaml.SafeLoader.add_constructor('!DataFrame', _pandas_df_constructor)


def _numpy_represent_floating(dumper, data):
    return dumper.represent_float(data.item())


def _numpy_represent_int(dumper, data):
    return dumper.represent_int(data.item())


def _numpy_arr_representer(dumper, data):
    return _arr_representer(dumper, data, 'NDArray')


def _pandas_df_representer(dumper, data):
    return _arr_representer(dumper, data, 'DataFrame')


def _arr_representer(dumper, data, r_type):
    if hasattr(data, "path"):
        rel_path = _dump_representation(data, data.path, Path(dumper.stream.name).parent)
        return dumper.represent_scalar(f'!{r_type}', rel_path)

    try:
        return dumper.represent_mapping(f'!{r_type}', data.to_dict())
    except AttributeError:
        return dumper.represent_sequence(f'!{r_type}', data.tolist())


def _dump_representation(data, path, stream_loc):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path)
    return str(path.relative_to(stream_loc))


def _pandas_df_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return pd.DataFrame(loader.construct_mapping(node))

    data_path = Path(loader.construct_scalar(node))

    if not data_path.is_absolute():
        try:
            stream_name = loader.stream.name
        except AttributeError:
            raise ValueError(f"Path {data_path} must be absolute if yaml stream has no 'name'.")

        data_path = Path(stream_name).parent / data_path

    return pd.read_csv(data_path, index_col=0)


def _numpy_arr_constructor(loader, node):
    if isinstance(node, yaml.SequenceNode):
        return np.array(loader.construct_sequence(node))

    return _pandas_df_constructor(loader, node).values


class NDArraySubclass(np.ndarray):
    """
    A simple python class that allows a 'path' attribute for serialization.
    `path` may be lost if object is manipulated.
    """
    def __new__(cls, input_array, path=None):
        obj = np.asarray(input_array).view(cls)
        obj.path = path
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.path = getattr(obj, 'path', None)
