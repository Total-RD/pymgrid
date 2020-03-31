'''
Inputs:
- nb microgrid
- size of load = rand
- shape of load = rand
- pv = 1
- genset = 1
- wind = 0
- gas = 0
- hydrogen = 0
- battery = 1
- (smart load)
- random_seed
- timestep

Function:
- get solar profile (maybe 5 timeseries in the ploutos)
- get wind profile (maybe 5 timeseries)
- generate load profile (triangle, square, AR process? others)
- get sizing based on annual load size? +- noise

Next step:
- implement benchmark algorithms
- control for RL in notebook

A (V0) microgrid is:
- PV time series, + P_max
- Load time series, + E_yearly
- Genset, a polynome, + P_rated, Power + and -
- Battery, Capa, P + and -, SOC + and -
- optional: Grid, interconnexion capacity + and -


'''

'''
todo
- Add files in the data folders (one or 2 pv, one or 2 load)

Test the generation fo microgrid scenarios




'''
import numpy as np
import pandas as pd
from ploutos import Microgrid
from os import listdir
from os.path import isfile, join

class MicrogridGenerator:

    def __init__(self, nb_microgrid=1, random_seed=42, timestep=1, path='yourpath'):
        np.random.seed(random_seed)
        #todo manage simulation duration and different timesteps
        self.microgrids= [] # generate a list of microgrid object
        #self.annual_load
        self.nb_microgrids=nb_microgrid
        self.timestep=1
        self.path=path

    ###########################################
    #utility functions
    ###########################################

    def get_random_file(self, path):

        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        #todo check for files name in a cleanedr way
        if '.DS_Store'  in onlyfiles:
            onlyfiles.remove('.DS_Store')

        file = pd.read_csv(path + onlyfiles[np.random.randint(low=0, high=len(onlyfiles))])

        # get number of files in our database
        # generate a random integer to select the files
        # Resample the file if needed
        return file

    def scale_ts(self, ts, size, scaling_method='sum'):

        actual_ratio=1
        if scaling_method =='sum':
            actual_ratio = size/ts.sum()#.values[0]

        if scaling_method == 'max':
            actual_ratio=ts.max().values[0]/size

        ts = ts * actual_ratio

        return ts

    def resize_timeseries(timeserie, current_time_step, new_time_step):

        index = pd.date_range('1/1/2015 00:00:00', freq=str(int(current_time_step * 60)) + 'Min',
                              periods=(len(timeserie)))  # , freq='0.9S')

        unsampled = pd.Series(timeserie, index=index)
        resampled = unsampled.resample(rule=str(int(new_time_step * 60)) + 'Min').mean().interpolate(method='linear')

        return resampled.values

    ###########################################
    # methods to generate timeseries
    ###########################################
    # def load_generator(self, shape, annual_consumption=1): #consumption inMWh
    #     ts_load = np.random(shape)
    #     annual_consumption = annual_consumption

    def get_pv_ts(self):
        #open pv folder
        # get list of file
        # select randomly rank if file to select in the list

        path = self.path+'data/pv/'
        return self.get_random_file(path)

    def get_load_ts(self):
        #open load folder
        # get list of file
        # select randomly rank if file to select in the list

        path = self.path+'data/load/'
        return self.get_random_file(path)

    def get_wind_ts(self):
        #open load folder
        # get list of file
        # select randomly rank if file to select in the list

        path = self.path+'data/wind/'
        return self.get_random_file(path)

    def generate_weak_grid_profile(self, outage_per_day, duration_of_outage,nb_time_step_per_year):

        #weak_grid_timeseries = np.random.random_integers(0,1, int(nb_time_step_per_year+1) ) #for a number of time steps, value between 0 and 1
        #generate a timeseries of 8760/timestep points based on np.random seed
        #profile of ones and zeros
        weak_grid_timeseries = np.random.random(int(nb_time_step_per_year+1) ) #for a number of time steps, value between 0 and 1


        weak_grid_timeseries = [0 if weak_grid_timeseries[i] < outage_per_day/24 else 1 for i in range(len(weak_grid_timeseries))]

        timestep=8760/nb_time_step_per_year
        for i in range(len(weak_grid_timeseries)):
            if weak_grid_timeseries[i] == 0:
                for j in range(1, int(duration_of_outage/timestep)):
                    if i-j > 0:
                        weak_grid_timeseries[i-j] = 0



        #print weak_grid_timeseries

        return pd.DataFrame(weak_grid_timeseries, columns=['grid_status']) #[0 if weak_grid_timeseries[i] < h_outage_per_day/24 else 1 for i in range(len(weak_grid_timeseries))]


    ###########################################
    # sizing functions
    ###########################################
    def size_mg(self, load, size_load=1):
        # generate a list of size based on the number of architecture  generated
        # 2 size the other generators based on the load
        pv=size_load*(np.random.randint(low=30, high=120)/100)

        #battery_size = self.size_battery(load)
        # return a dataframe with the power of each generator, and if applicable the number of generator

        size={
            'pv': pv,
            'load': size_load,
            'battery': self.size_battery(load),
            'genset': self.size_genset(load),
            'grid': max(load.values)*2,
        }

        return size

    def size_genset(self, load, max_operating_loading = 0.9):
        #random number > 3 < 20
        # polynomial for fuel consumption

        size_genset = np.max(load.values)/max_operating_loading

        return size_genset


    def size_battery(self, load):
        #energy duration
        battery = np.random.randint(low=3,high=5)*np.mean(load.values)
        #todo duration & power
        return battery

    ###########################################
    # parameters of components
    ###########################################

    def get_genset(self, rated_power=1000, pmax=0.9, pmin=0.2):
        polynom=[np.random.rand()*10, np.random.rand(), np.random.rand()/10] #fuel consumption

        genset={
            'polynom':polynom,
            'rated_power':rated_power,
            'pmax':pmax,
            'pmin':pmin,
            'fuel_cost':0.4
        }

        return genset

    def get_battery(self, capa=1000, duration=4, pcharge=100, pdischarge=100, soc_max=1, soc_min=0.2, efficiency=0.9):
        battery={
            'capa':capa,
            'pcharge':capa/duration,
            'pdischarge':capa/duration,
            'soc_max':soc_max,
            'soc_min':soc_min,
            'efficiency':efficiency,
            'soc_0':min(max(np.random.randn(), soc_min),soc_max),
            'cost_cycle':0.3

        }
        return battery


    def get_grid(self, rated_power=1000, weak_grid=0, pmin=0.2, price_export = 0, price_import =0.3):

        grid_ts=[]
        if weak_grid == 1:
            rand_outage_per_day = np.random.randn()*3/4 +0.25
            rand_duration = np.random.randint(low=1, high =8)
            grid_ts = self.generate_weak_grid_profile( rand_outage_per_day, rand_duration,8760/self.timestep)
        else:
            grid_ts=pd.DataFrame([1+i*0 for i in range(int(np.floor(8760/self.timestep)))], columns=['grid_status'])

        grid={
            'grid_power_import':rated_power,
            'grid_power_export':rated_power,
            'grid_ts':grid_ts,
            'grid_price_export':price_export,
            'grid_price_import': price_import,
        }

        return grid

    ###########################################
    #generate the microgrid
    ###########################################


    def create_microgrid(self, ):
        # get the sizing data
        # create microgrid object and append
        # return the list

        rand = np.random.randn()
        bin_genset = 0
        bin_grid = 0

        if rand <0.33:
            bin_genset =1
        elif rand>= 0.33 and rand <0.66:
            bin_grid =1
        else:
            bin_genset=1
            bin_grid=1

        architecture = {'PV':1, 'battery':1, 'genset':bin_genset, 'grid':bin_grid}
        #size_df=8760
        size_load = np.random.randint(low=876000,high=10000000)
        load = pd.DataFrame(self.scale_ts(self.get_load_ts(), size_load))

        size = self.size_mg(load, size_load)

        column_actions=[]

        column_actual_production=[]
        grid_ts=[]
        df_status = pd.DataFrame([0], columns=['net_load'])
        df_parameters =pd.DataFrame([size_load],columns=['load'])
        df_parameters['cost_loss_load'] = 10000
        df_cost = pd.DataFrame([0.0], columns=['cost'])

        if architecture['PV']==1:

            df_parameters['PV_rated_power'] = size['pv']
            column_actual_production.append('pv_consummed')
            column_actual_production.append('pv_curtailed')


        if architecture['battery']==1:
            battery = self.get_battery(capa=size['battery'])
            #print(battery['soc_0'])
            df_parameters['battery_soc_0'] = battery['soc_0']
            df_parameters['battery_power_charge'] = battery['pcharge']
            df_parameters['battery_power_discharge'] = battery['pdischarge']
            df_parameters['battery_capacity'] = battery['capa']
            df_parameters['battery_efficiency'] = battery['efficiency']
            df_parameters['battery_soc_min'] = battery['soc_min']
            df_parameters['battery_soc_max'] = battery['soc_max']
            df_parameters['battery_cost_cycle'] = battery['cost_cycle']
            column_actual_production.append('battery_charge')
            column_actual_production.append('battery_discharge')
            df_status['battery_soc'] = battery['soc_0']
            df_status['battery_soc'] = battery['soc_0']

        if architecture['genset']==1:
            genset = self.get_genset(rated_power=size['genset'])
            df_parameters['genset_polynom_order'] = len(genset['polynom'])

            for i in range(len(genset['polynom'])):
                df_parameters['genset_polynom_'+str(i)]=genset['polynom'][i]
            df_parameters['genset_rated_power'] = genset['rated_power']
            df_parameters['genset_pmin'] = genset['pmin']
            df_parameters['genset_pmax'] = genset['pmax']
            df_parameters['fuel_cost'] = genset['fuel_cost']
            column_actual_production.append('genset')

        grid_spec=0
        if architecture['grid']==1:
            rand_weak_grid = np.random.randint(low=0, high=1)

            grid = self.get_grid(rated_power=size['grid'], weak_grid=rand_weak_grid)
            df_parameters['grid_power_import'] = grid['grid_power_import']
            df_parameters['grid_power_export'] = grid['grid_power_export']
            grid_ts = grid['grid_ts']
            df_parameters['grid_price_import'] = grid['grid_price_import']
            df_parameters['grid_price_export'] = grid['grid_price_export']
            column_actual_production.append('grid_import')
            column_actual_production.append('grid_export')

        column_actions = column_actual_production

        df_actions= pd.DataFrame(columns = column_actions, )
        df_actual_production = pd.DataFrame(columns=column_actual_production)

        #todo change microgrid spec to a more general set of attribure

        microgrid_spec={
            'parameters':df_parameters,
            'df_actions':df_actions,
            'architecture':architecture,
            'df_status':df_status,
            'df_actual_generation':df_actual_production,
            'grid_spec':grid_spec,
            'df_cost':df_cost,

            'pv':pd.DataFrame(self.scale_ts(self.get_pv_ts(), size['pv'])),
            'load': load,
            'grid_ts':grid_ts
        }

        microgrid = Microgrid.Microgrid(microgrid_spec)

        return microgrid

        #1. Generagte a DF with the parameters of each generator

        #2. Get the timeseries for load and generation

        #3. Dataframe with a timeseries of conrol order and updated battery parameters
        #- updated battery SOC, fuel consumption, load, power produced

    def generate_microgrid(self):

        for i in range(self.nb_microgrids):
            #size=self.size_mg()
            self.microgrids.append(self.create_microgrid())


        return self.microgrids
