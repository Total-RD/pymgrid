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


    #function to plot the parameters of all the microgrid generated
    def mg_parameters(self, id = 'all'):


        if id == 'all':

            if self.microgrids != []:
                parameters = pd.DataFrame()
                for i in range(self.nb_microgrids):

                    parameters = parameters.append(self.microgrids[i].parameters, ignore_index=True)

                pd.options.display.max_columns = None
                display(parameters)

        elif isinstance(id, int) and id < self.nb_microgrids:
        	
            display(self.microgrids[id].parameters)


    def _get_random_file(self, path):

        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        #todo check for files name in a cleanedr way
        if '.DS_Store'  in onlyfiles:
            onlyfiles.remove('.DS_Store')

        file = pd.read_csv(path + onlyfiles[np.random.randint(low=0, high=len(onlyfiles))])

        # get number of files in our database
        # generate a random integer to select the files
        # Resample the file if needed
        return file

    def _scale_ts(self, df_ts, size, scaling_method='sum'):

        actual_ratio=1
        if scaling_method =='sum':
            actual_ratio = size/df_ts.sum()#.values[0]

        if scaling_method == 'max':
            actual_ratio=df_ts.max().values[0]/size

        df_ts = df_ts * actual_ratio

        return df_ts

    def _resize_timeseries(timeserie, current_time_step, new_time_step):

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

    def _get_pv_ts(self):
        #open pv folder
        # get list of file
        # select randomly rank if file to select in the list

        path = self.path+'data/pv/'
        return self._get_random_file(path)

    def _get_load_ts(self):
        #open load folder
        # get list of file
        # select randomly rank if file to select in the list

        path = self.path+'data/load/'
        return self._get_random_file(path)

    def _get_wind_ts(self):
        #open load folder
        # get list of file
        # select randomly rank if file to select in the list

        path = self.path+'data/wind/'
        return self._get_random_file(path)

    def _get_genset(self, rated_power=1000, pmax=0.9, pmin=0.2):

        polynom=[np.random.rand()*10, np.random.rand(), np.random.rand()/10] #fuel consumption

        genset={
            'polynom':polynom,
            'rated_power':rated_power,
            'pmax':pmax,
            'pmin':pmin,
            'fuel_cost':0.4
        }

        return genset

    def _get_battery(self, capa=1000, duration=4, pcharge=100, pdischarge=100, soc_max=1, soc_min=0.2, efficiency=0.9):
        battery={
            'capa':capa,
            'pcharge':int(np.ceil(capa/duration)),
            'pdischarge':int(np.ceil(capa/duration)),
            'soc_max':soc_max,
            'soc_min':soc_min,
            'efficiency':efficiency,
            'soc_0':min(max(np.random.randn(), soc_min),soc_max),
            'cost_cycle':0.3

        }
        return battery


    def _get_grid(self, rated_power=1000, weak_grid=0, pmin=0.2, price_export = 0, price_import =0.3):

        if weak_grid == 1:
            rand_outage_per_day = np.random.randn()*3/4 +0.25
            rand_duration = np.random.randint(low=1, high =8)
            grid_ts = self._generate_weak_grid_profile( rand_outage_per_day, rand_duration,8760/self.timestep)

        else:
            #grid_ts=pd.DataFrame([1+i*0 for i in range(int(np.floor(8760/self.timestep)))], columns=['grid_status'])
            grid_ts = pd.DataFrame(np.ones(int(np.floor(8760 / self.timestep))),
                                   columns=['grid_status'])

        grid={
            'grid_power_import':rated_power,
            'grid_power_export':rated_power,
            'grid_ts':grid_ts,
            'grid_price_export':price_export,
            'grid_price_import': price_import,
        }

        return grid

    def _generate_weak_grid_profile(self, outage_per_day, duration_of_outage,nb_time_step_per_year):

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
    def _size_mg(self, load, size_load=1):
        # generate a list of size based on the number of architecture  generated
        # 2 size the other generators based on the load
        pv=size_load*(np.random.randint(low=30, high=121)/100)

        #battery_size = self._size_battery(load)
        # return a dataframe with the power of each generator, and if applicable the number of generator

        size={
            'pv': pv,
            'load': size_load,
            'battery': self._size_battery(load),
            'genset': self._size_genset(load),
            'grid': int(max(load.values)*2),
        }

        return size

    def _size_genset(self, load, max_operating_loading = 0.9):
        #random number > 3 < 20
        # polynomial for fuel consumption

        _size_genset = int(np.ceil(np.max(load.values)/max_operating_loading))

        return _size_genset


    def _size_battery(self, load):
        #energy duration
        battery = int(np.ceil(np.random.randint(low=3,high=6)*np.mean(load.values)))
        #todo duration & power
        return battery


    ###########################################
    #generate the microgrid
    ###########################################

    def generate_microgrid(self, verbose=True):

        for i in range(self.nb_microgrids):
            #size=self._size_mg()
            self.microgrids.append(self._create_microgrid())
        
        if verbose == True:
            self.mg_parameters()

    def _create_microgrid(self):
        # get the sizing data
        # create microgrid object and append
        # return the list
        rand = np.random.rand()
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
        size_load = np.random.randint(low=876000,high=10000001)
        load = self._scale_ts(self._get_load_ts(), size_load) #obtain dataframe of loads
        size = self._size_mg(load, size_load) #obtain a dictionary of mg sizing components

        column_actions=[]
        column_actual_production=[]
        grid_ts=[]
        df_parameters = pd.DataFrame()
        df_cost = pd.DataFrame()
        df_status = pd.DataFrame()


        df_parameters['load'] = [size_load]
        df_parameters['cost_loss_load'] = 10000
        df_cost['cost'] = [0.0]
        #df_status['net_load'] = [load.iloc[0].values] --> il y a doublon pour l'instant avec l'architecture PV

        if architecture['PV'] == 1:

            df_parameters['PV_rated_power'] = np.around(size['pv'],1)
            column_actual_production.append('pv_consummed')
            column_actual_production.append('pv_curtailed') 
            pv = pd.DataFrame(self._scale_ts(self._get_pv_ts(), size['pv']))           
            df_status['net_load'] = np.around(load.iloc[0].values - pv.iloc[0].values,1)

        if architecture['battery']==1:

            battery = self._get_battery(capa=size['battery']) #return a dictionary of battery characteristic
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

            capa_to_charge = max(
                (df_parameters['battery_soc_max'].values[0] * df_parameters['battery_capacity'].values[0] -
                 df_parameters['battery_soc_0'].iloc[-1] *
                 df_parameters['battery_capacity'].values[0]
                 ) / df_parameters['battery_efficiency'].values[0], 0)

            capa_to_discharge = max((df_parameters['battery_soc_0'].iloc[-1] *
                                     df_parameters['battery_capacity'].values[0]
                                     - df_parameters['battery_soc_min'].values[0] *
                                     df_parameters['battery_capacity'].values[0])
                                     * df_parameters['battery_efficiency'].values[0], 0)

            df_status['capa_to_charge'] = np.around(capa_to_charge,1)
            df_status['capa_to_discharge'] = np.around(capa_to_discharge,1)

        if architecture['genset']==1:
            genset = self._get_genset(rated_power=size['genset'])
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

            rand_weak_grid = np.random.randint(low=0, high=2)

            grid = self._get_grid(rated_power=size['grid'], weak_grid=rand_weak_grid)
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
            'parameters':df_parameters, #Dictionnary
            'df_actions':df_actions, #Dataframe
            'architecture':architecture, #Dictionnary
            'df_status':df_status, #Dictionnary
            'df_actual_generation':df_actual_production,#Dataframe
            'grid_spec':grid_spec, #value = 0
            'df_cost':df_cost, #Dataframe of 1 value = 0.0
            'pv':pv, #Dataframe
            'load': load, #Dataframe
            'grid_ts':grid_ts #Dataframe
        }

        microgrid = Microgrid.Microgrid(microgrid_spec)

        return microgrid

        #1. Generagte a DF with the parameters of each generator

        #2. Get the timeseries for load and generation

        #3. Dataframe with a timeseries of conrol order and updated battery parameters
        #- updated battery SOC, fuel consumption, load, power produced

