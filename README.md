# pymgrid

![Build](https://github.com/Total-RD/pymgrid/workflows/build/badge.svg?dummy=unused)

pymgrid (PYthon MicroGRID) is a python library to generate and simulate a large number of microgrids.

For more context, please see the [presentation](https://www.climatechange.ai/papers/neurips2020/3) done at Climate Change AI.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pymgrid. You can clone and cd in the repo and then do: 

```bash
pip install .
```

You can also run this pip command:
```bash
pip install git+https://github.com/Total-RD/pymgrid/
```


And in Google Colab:
```bash
!pip install git+https://github.com/Total-RD/pymgrid/
```

## Getting Started

You can easily import the library from pip, and then import MicrogridGenerator from pymgrid.

```python
from pymgrid import MicrogridGenerator as mg

m_gen=mg.MicrogridGenerator()
m_gen.generate_microgrid()
```

By default, this command will let you generate 10 microgrids. The object m_gen will have a list of microgrids that you can use.

First, you can get the control information with this command:
```python
m_gen.microgrids[0].print_control_info()

```

pymgrid contains OpenAI Gym environments, you can use the following command to generate one:
```python
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv
from pymgrid import MicrogridGenerator as m_gen

#these line will create a list of microgrid
env = m_gen.MicrogridGenerator(nb_microgrid=25)
pymgrid25 = env.load('pymgrid25')
mg = pymgrid25.microgrids

#you can pass any of the microgrid to environment class:
env = MicroGridEnv({'microgrid':mg[0]})

#example of codes to to interact with the environment:
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = #your algorithm to select the next action
    obs, reward, done, info = env.step(action)
    episode_reward += reward
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

## Benchmarks datasets

We pre-computed two microgrids datasets for researchers to compare their algorithms on:
1. pymgrid10 (deprecated at the moment): ten microgrids with the same architecture (PV + battery + genset), the main goal of this dataset if for user to beging running simulations on pymgrid
2. pymgrid25: 25 microgrids with diverse architecture, we propose this dataset as the main way to compare algorithms.

If you have ideas for new benchmark dataset, feel free to contact us!

You can load these datasets with:
```python
from pymgrid import MicrogridGenerator as mg

m_gen=mg.MicrogridGenerator()
m_gen.load('pymgrid25') 
```
## Citation

If you use this package for your research, please cite the following paper:

@misc{henri2020pymgrid,
      title={pymgrid: An Open-Source Python Microgrid Simulator for Applied Artificial Intelligence Research}, 
      author={Gonzague Henri and Tanguy Levent and Avishai Halev and Reda Alami and Philippe Cordier},
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

For any question you can contact me at tanguy.levent [@] external.total.com
