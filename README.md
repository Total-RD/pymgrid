# pymgrid

![Build](https://github.com/Total-RD/pymgrid/workflows/build/badge.svg?dummy=unused)

pymgrid (PYthon MicroGRID) is a python library to generate and simulate a large number of microgrids.

For more context, please see the [presentation](https://www.climatechange.ai/papers/neurips2020/3) done at Climate Change AI
and the [documentation](https://pymgrid.readthedocs.io).

## Installation

The easiest way to install pymgrid is with pip:

`pip install -U pymgrid`

Alternatively, you can install from source. First clone the repo:
 
```bash
git clone https://github.com/Total-RD/pymgrid.git
``` 
Then navigate to the root directory of pymgrid and call

```bash
pip install .
```
## Getting Started

Microgrids are straightforward to generate from scratch. Simply define some modules and pass them
to a microgrid:
```python
import numpy as np
from pymgrid import Microgrid
from pymgrid.modules import GensetModule, BatteryModule, LoadModule, RenewableModule


genset = GensetModule(running_min_production=10,
                      running_max_production=50,
                      genset_cost=0.5)

battery = BatteryModule(min_capacity=0,
                        max_capacity=100,
                        max_charge=50,
                        max_discharge=50,
                        efficiency=1.0,
                        init_soc=0.5)

# Using random data
renewable = RenewableModule(time_series=50*np.random.rand(100))

load = LoadModule(time_series=60*np.random.rand(100),
                  loss_load_cost=10)

microgrid = Microgrid([genset, battery, ("pv", renewable), load])
```

This creates a microgrid with the modules defined above, as well as an unbalanced energy module -- 
which reconciles situations when energy demand cannot be matched to supply.

Printing the microgrid gives us its architecture:

```python
>> microgrid

Microgrid([genset x 1, load x 1, battery x 1, pv x 1, balancing x 1])
```

A microgrid is contained of fixed modules and flex modules. Some modules can be both -- `GridModule`, for example
-- but not at the same time.


A *fixed* module has requires a request of a certain amount of energy ahead of time, and then attempts to 
produce or consume said amount. `LoadModule` is an example of this; you must tell it to consume a certain amount of energy
and it will then do so.

 A *flex* module, on the other hand, is able to adapt to meet demand. `RenewableModule` is an example of this as
 it allows for curtailment of any excess renewable produced.
 
 A microgrid will tell you which modules are which:
 
 ```python
>> microgrid.fixed_modules

{
  "genset": "[GensetModule(running_min_production=10, running_max_production=50, genset_cost=0.5, co2_per_unit=0, cost_per_unit_co2=0, start_up_time=0, wind_down_time=0, allow_abortion=True, init_start_up=True, raise_errors=False, provided_energy_name=genset_production)]",
  "load": "[LoadModule(time_series=<class 'numpy.ndarray'>, loss_load_cost=10, forecaster=NoForecaster, forecast_horizon=0, forecaster_increase_uncertainty=False, raise_errors=False)]",
  "battery": "[BatteryModule(min_capacity=0, max_capacity=100, max_charge=50, max_discharge=50, efficiency=1.0, battery_cost_cycle=0.0, battery_transition_model=None, init_charge=None, init_soc=0.5, raise_errors=False)]"
}

>>microgrid.flex_modules

{
  "pv": "[RenewableModule(time_series=<class 'numpy.ndarray'>, raise_errors=False, forecaster=NoForecaster, forecast_horizon=0, forecaster_increase_uncertainty=False, provided_energy_name=renewable_used)]",
  "balancing": "[UnbalancedEnergyModule(raise_errors=False, loss_load_cost=10, overgeneration_cost=2)]"
}

```


Running the microgrid is straightforward. Simply pass an action for each fixed module to `microgrid.run`. The microgrid
can also provide you a random action by calling `microgrid.sample_action.` Once the microgrid has been run for a
certain number of steps, results can be viewed by calling microgrid.get_log.

```python
>> for j in range(10):
>>    action = microgrid.sample_action(strict_bound=True)
>>    microgrid.run(action)

>> microgrid.get_log(drop_singleton_key=True)

      genset  ...                     balance
      reward  ... fixed_absorbed_by_microgrid
0  -5.000000  ...                   10.672095
1 -14.344353  ...                   50.626726
2  -5.000000  ...                   17.538018
3  -0.000000  ...                   15.492778
4  -0.000000  ...                   35.748724
5  -0.000000  ...                   30.302300
6  -5.000000  ...                   36.451662
7  -0.000000  ...                   66.533872
8  -0.000000  ...                   20.645077
9  -0.000000  ...                   10.632957
```

## Benchmarking

`pymgrid` also comes pre-packaged with a set of 25 microgrids for benchmarking.
The config files for these microgrids are available in `data/scenario/pymgrid25`.
Simply deserialize one of the yaml files to load one of the saved microgrids; for example,
to load the zeroth microgrid:

```python
import yaml
from pymgrid import PROJECT_PATH

yaml_file = PROJECT_PATH / 'data/scenario/pymgrid25/microgrid_0/microgrid_0.yaml'
microgrid = yaml.safe_load(yaml_file.open('r'))
```

Alternatively, `Microgrid.load(yaml_file.open('r'))` will perform the same deserialization.


## Citation

If you use this package for your research, please cite the following paper:

@misc{henri2020pymgrid,
      title={pymgrid: An Open-Source Python Microgrid Simulator for Applied Artificial Intelligence Research}, 
      author={Gonzague Henri, Tanguy Levent, Avishai Halev, Reda Alami and Philippe Cordier},
      year={2020},
      eprint={2011.08004},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}

You can find it on Arxiv here: https://arxiv.org/abs/2011.08004

## Data

Data in pymgrid are based on TMY3 (data based on representative weather). The PV data comes from DOE/NREL/ALLIANCE (https://nsrdb.nrel.gov/about/tmy.html) and the load data comes from OpenEI (https://openei.org/doe-opendata/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations-in-the-united-states)

The CO2 data is from Jacque de Chalendar and his gridemissions API.

## Contributing
Pull requests are welcome for bug fixes. For new features, please open an issue first to discuss what you would like to add.

Please make sure to update tests as appropriate.

## License

This repo is under a GNU LGPL 3.0 (https://github.com/total-sa/pymgrid/edit/master/LICENSE)

## Contact

For any question you can contact me at avishai.halev [@] external.totalenergies.com
