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

```python
from pymgrid import MicrogridGenerator as mg

m_gen=mg.MicrogridGenerator()
m_gen.generate_microgrid()
```

## Data

Data in pymgrid are based on TMY3 (data based on representative weather). The PV data comes from NREL (https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/) and the load data comes from OpenEI (https://openei.org/doe-opendata/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations-in-the-united-states)

## Contributing
Pull requests are welcome for bug fixes. For new features, please open an issue first to discuss what you would like to add.

Please make sure to update tests as appropriate.

## License
