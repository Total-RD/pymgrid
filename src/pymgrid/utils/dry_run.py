import yaml

from contextlib import contextmanager
from typing import Union

import pymgrid


@contextmanager
def dry_run(pymgrid_object: Union['pymgrid.Microgrid', 'pymgrid.modules.base.BaseMicrogridModule']):
    serialized = yaml.safe_dump(pymgrid_object)

    try:
        yield pymgrid_object
    finally:
        deserialized = yaml.safe_load(serialized)
        try:
            # Module
            data = deserialized._serialize_state_attributes()
        except AttributeError:
            # Microgrid
            data = deserialized._serialization_data()
            pymgrid_object._modules = deserialized.modules

        pymgrid_object.deserialize(data)