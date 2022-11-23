.. _api.microgrid:


Microgrid
=================

.. currentmodule:: pymgrid

Constructor
-----------
.. autosummary::
    :toctree: api/microgrid/

    Microgrid

Methods
-------
.. autosummary::

    :toctree: api/microgrid/

    Microgrid.run
    Microgrid.reset
    Microgrid.sample_action
    Microgrid.get_log
    Microgrid.get_forecast_horizon
    Microgrid.get_empty_action

Serialization/IO/Conversion
---------------------------
.. autosummary::

    :toctree: api/microgrid/

    Microgrid.load
    Microgrid.dump
    Microgrid.from_nonmodular
    Microgrid.from_scenario
    Microgrid.serialize
    Microgrid.to_nonmodular