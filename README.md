# pymgrid

pymgrid (PYthon MicroGRID) is a python library to generate and simulate a large number of microgrids.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pymgrid. (available soon)

```bash
pip install pymgrid
```

## Benchmarks datasets

We pre-computed two microgrids datasets for researchers to compare their algorithms on:
1. pymgrid10: ten microgrids with the same architecture (PV + battery + genset), the main goal of this dataset if for user to beging running simulations on pymgrid
2. pymgrid25: 25 microgrids with diverse architecture, we propose this dataset as the main way to compare algorithms.

If you have ideas for new benchmark dataset, feel free to contact us!


## Usage

You can easily import the library from pip, and then import MicrogridGenerator from pymgrid.

```python
from pymgrid import MicrogridGenerator as mg

m_gen=mg.MicrogridGenerator()
m_gen.generate_microgrid()
```

By default, this command will let you generate 10 microgrids. The object m_gen will have a list of microgrids that you can use.

First, you can get the control information with this command:
```python
m_gen.microgrids[0].get_control_info()
```
The control_dict dictionnary it the main way you will interact with the microgrid class. It will allow you to pass control commands to the microgrids. Using get_control_info() will let you know what fields you need to fill based on the microgrids architecture.

Now you know what fields in control_dict, you can fill it up and pass it to your microgrid:
```python
ctrl = # your own control actions
m_gen.microgrids[0].run(ctrl)
```
All the management of the timesteps, and verifiying that the control actions respect the microgrid constraints.

If you are interested in using pymgrid for machine learning or reinforcement learning, you will find this command useful.
You can split your dataset in two with:
```python
m_gen.microgrids[0].train_test_split() # you will automatically be using the training set with this command
```
If you want to run your training algorithm through multiple epochs you can reset the microgrid once the simulation reaches the last timestep:
```python
if m_gen.microgrids[0].done == True: #the done argument becomes true once you reache the last timestep of your simulation
  m_gen.microgrids[0].reset() 
```

You can swith to the testing set with:
```python
m_gen.microgrids[0].reset(testing=True)
```


## Data

Data in pymgrid are based on TMY3 (data based on representative weather). The PV data comes from NREL (https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/) and the load data comes from OpenEI (https://openei.org/doe-opendata/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations-in-the-united-states)

## Contributing
Pull requests are welcome for bug fixes. For new features, please open an issue first to discuss what you would like to add.

Please make sure to update tests as appropriate.

## License

This repo is under a GNU GPL 2.1 (https://github.com/total-sa/pymgrid/edit/master/LICENSE)
