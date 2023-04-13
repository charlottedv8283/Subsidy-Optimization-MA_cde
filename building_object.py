# for testing purposes; performance tracking
import tracemalloc
import time
# packages for logging, saving, ...
import pickle as pkl
import logging
import os
# data packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
# Gurobi packages
import gurobipy as gp
from gurobipy import GurobiError
# own packages imports
import python.clustering.clustering_medoid as clustering
import python.input_import.parse_inputs as pik
import python.input_import.reference_building as ref_bui
import python.auxiliary_modules.aux_log as aux_log
import python.auxiliary_modules.aux_save as aux_save
# opti model packages
from python.grbi.set_vars import set_vars_eco, set_vars_tec, \
    set_vars_restruc_meas, set_vars_sub
from python.grbi.set_constr import set_constr_building, \
    set_constr_costs, set_constr_tech, set_constr_scenario, set_constr_subs
from python.grbi.set_params import set_params


# List with possible inputs for building object
l_input_typ = ("SFH", "TH", "MFH", "AB")
l_input_age = ("0_1859", "1860_1918", "1919_1948", "1949_1957",
               "1958_1968", "1969_1978", "1979_1983", "1984_1994",
               "1995_2001", "2002_2009", "2010_2015", "2016_2100")


@dataclass
class Options():

    # Set Project Name
    project_name: str = "unnamed_project"

    # Files for reading inputs
    dev_file: str = "raw_inputs/devices.xlsx"
    weather_file: str = "raw_inputs/weather_files/"
    eco_file: str = "raw_inputs/economics.xlsx"
    subs_file: str = "raw_inputs/subsidies.xlsx"
    bld_file: str = "raw_inputs/buildings.xlsx"
    # Parameter to obtain more information about non feasible model
    infeas_unbound_process: bool = False

    # Optimization of costs (True) or emissions (False)
    opt_costs: bool = True

    # Subsidy programs
    eeg: bool = False
    kwkg: bool = False
    # new subsidy programs (September 2021)
    # Bundesförderung für effiziente Gebäude - Einzelmaßnahmen
    beg_em: bool = False
    # Bundesförderung für effiziente Gebäude - Wohngebäude
    beg_wg: bool = False

    # Modeling parameters
    nw_bld: bool = False
    dhw_electric: bool = False
    design_heat_load: bool = True
    scen: str = "free"
    useable_roofarea: float = 0.3
    exist_boiler: bool = False
    p_grid_max: float = 0
    el_EE_share: float = 0
    el_emi_factor: float = 0
    max_emi: float = 0
    co2_price: bool | str = "off"  # "min", "med", "max" or "off"
    el_dem: str = "medium"
    dhw_dem: str = "medium"

    # Clustering options
    clust_method: str = "k_MILP"  # "k_medoids", "k_MILP"
    num_cluster: int = 8

    # Gurobi parameters
    grb_LogToConsole: int = 0
    grb_OutputFlag: int = 1
    grb_LogFile: str = 'gurobi.log'
    grb_TimeLimit: int = 500
    grb_MIPGap: float = 0.01
    grb_NumericFocus: int = 0
    grb_MIPFocus: int = 2
    grb_Aggregate: int = 1
    grb_write: str = "output.lp"
    grb_PoolSearchMode: int = 2
    grb_PoolSolutions: int = 10
    grb_PoolGap: float = 0.05

    if grb_write.split('.')[1].upper() not in ['MPS', 'REW', 'LP', 'RLP', 'ILP', 'OPB']:  # NOQA: E501
        fail_end = grb_write.split('.')[1]
        raise ValueError(
            f"""
            Gurobi parameter \'grb_write\' should have ending out
            of ['MPS', 'REW', 'LP', 'RLP', 'ILP', 'OPB'] \n
            but ends with {fail_end}. Please adapt value for grb_write.
            """)


class Building():

    def __init__(self, typ: str, age: str, loc: str, apart_quant: int,
                 apart_size: int, apart_habitant: int, opt: object):

        # create dummy attributes to be filled later
        self.input_filename = None
        self.pkl_dir = None
        self.xl_dir = None
        self.model = None
        self.mod_vars = {}
        self.mod_constr = {}

        self.clustered = {}
        self.raw_inputs = {}
        self.params = {}
        self.buildings = {}
        self.scenarios = {}
        self.len_day = None

        # set input parameters as bld attributes
        if typ not in l_input_typ:
            raise ValueError(
                f"""
                    Input for attribute typ is {typ}
                    but should be any out of {l_input_typ}
                    """)
        else:
            self.typ = typ
        if age not in l_input_age:
            raise ValueError(
                f"""
                Input for attribute age is {age}
                but should be any out of {l_input_age}
                """)
        else:
            self.age = age
        self.norm_loc = str.title(loc.lower().replace("ä", "ae").replace(
            "ö", "oe").replace("ü", "ue").replace(" ", "_"))
        loc_raw = os.listdir(os.path.join(os.getcwd(),
                                          "raw_inputs",
                                          "weather_files"))
        l_input_loc = []
        for l_raw in loc_raw:
            l_raw = l_raw.split(".")[0]
            raw_loc = ""
            for i, lo in enumerate(l_raw.split("_")):
                if lo not in ("solar", "roof", "temperature", "north", "east", "south", "west"):  # NOQA: E501
                    if i == 0:
                        raw_loc += lo
                    else:
                        raw_loc += " " + lo
            l_input_loc.append(raw_loc)
        l_input_loc = set(l_input_loc)
        if self.norm_loc not in l_input_loc:
            raise ValueError(
                f"""
                Input for attribute loc is {loc}
                but no matching weather files are
                found in \"raw_inputs/weather_files\"
                """)
        else:
            self.loc = loc

        apart_quant_raw = os.listdir(os.path.join(os.getcwd(),
                                                  "raw_inputs",
                                                  "mfh"))
        l_input_apart_quant = [1, 2]
        for quant_raw in apart_quant_raw:
            quant_raw = quant_raw.split(".")[0]
            for q in quant_raw.split("_"):
                if q.isdigit():
                    l_input_apart_quant.append(int(q))
        l_input_apart_quant = set(l_input_apart_quant)
        if apart_quant not in l_input_apart_quant:
            raise ValueError(
                f"""
                Input for attribute apart_quant is {apart_quant}
                but no matchinng consumer profiles are
                found in \"raw_inputs/mfh\"
                """)
        else:
            self.apart_quant = apart_quant
        apart_habitant_raw = os.listdir(os.path.join(os.getcwd(),
                                                     "raw_inputs",
                                                     "sfh"))
        l_input_apart_habitant = []
        for hab_raw in apart_habitant_raw:
            hab_raw = hab_raw.split(".")[0]
            for h_raw in hab_raw.split("_"):
                if h_raw.isdigit():
                    l_input_apart_habitant.append(int(h_raw))
        l_input_apart_habitant = set(l_input_apart_habitant)
        if apart_habitant not in l_input_apart_habitant:
            raise ValueError(
                f"""
                Input for attribute apart_habitant is {apart_habitant}
                but no matchinng consumer profiles are
                found in \"raw_inputs/sfh\"
                """)
        else:
            self.apart_habitant = apart_habitant
        if opt.el_dem not in ("low", "medium", "high"):
            raise ValueError(
                f"""
                Input for attribute el_dem is {opt.el_dem}
                but should be \"low\", \"medium\" or \"high\"
                """)
        else:
            self.el_dem = opt.el_dem
        if opt.dhw_dem not in ("low", "medium", "high"):
            raise ValueError(
                f"""
                Input for attribute dhw_dem is {opt.dhw_dem}
                but should be \"low\", \"medium\" or \"high\"
                """)
        else:
            self.dhw_dem = opt.dhw_dem
        self.apart_size = apart_size
        self.proj_name = opt.project_name

        self.opt_subs = {
            "EEG": opt.eeg,
            "KWKG": opt.kwkg,
            "BEG_EM": opt.beg_em,
            "BEG_WG": opt.beg_wg}
        self.opt_model = {
            "opt_costs": opt.opt_costs,
            "nw_bld": opt.nw_bld,
            "dhw_electric": opt.dhw_electric,
            "design_heat_load": opt.design_heat_load,
            "scenario": opt.scen,
            "useable_roofarea": opt.useable_roofarea,
            "exist_boiler": opt.exist_boiler,
            "p_grid_max": opt.p_grid_max,
            "el_EE_share": opt.el_EE_share,
            "el_emi_factor": opt.el_emi_factor,
            "max_emi": opt.max_emi,
            "co2_price": opt.co2_price}
        self.opt_cluster = {
            "clust_method": opt.clust_method,
            "num_cluster": opt.num_cluster}
        self.opt_fl_nms = {
            "dev_file": opt.dev_file,
            "weather_file": opt.weather_file,
            "eco_file": opt.eco_file,
            "subs_file": opt.subs_file,
            "bld_file": opt.bld_file}
        self.infeas_unbound_process = opt.infeas_unbound_process
        self.grb_params = {
            "LogToConsole": opt.grb_LogToConsole,
            "OutputFlag": opt.grb_OutputFlag,
            "LogFile": opt.grb_LogFile,
            "TimeLimit": opt.grb_TimeLimit,
            "MIPGap": opt.grb_MIPGap,
            "NumericFocus": opt.grb_NumericFocus,
            "MIPFocus": opt.grb_MIPFocus,
            "Aggregate": opt.grb_Aggregate,
            "write": opt.grb_write,
            "PoolSearchMode": opt.grb_PoolSearchMode,
            "PoolSolutions": opt.grb_PoolSolutions,
            "PoolGap": opt.grb_PoolGap,
        }

        # set project path and create dictionary
        self.proj_path = os.path.join(os.getcwd(), "results", self.proj_name)
        os.makedirs(self.proj_path, exist_ok=True)
        # create unique bld name for project
        c = 0
        bld_name = f"{self.typ}_{self.age}_{self.norm_loc}"
        dir_bld = os.path.join(self.proj_path, bld_name)
        if os.path.exists(dir_bld):
            new_folder = True
        else:
            new_folder = False
        while os.path.exists(dir_bld):
            c += 1
            bld_name = f"{self.typ}_{self.age}_{self.norm_loc}_{c}"
            dir_bld = os.path.join(self.proj_path, bld_name)
        self.bld_name = bld_name
        os.makedirs(dir_bld)
        self.dir_bld = dir_bld

        # Initialize file and stream logging
        self.logger = logging.getLogger(bld_name)
        self.logger.setLevel(logging.DEBUG)
        log_format_file = logging.Formatter(
            '%(asctime)-.19s:%(levelname)s:%(name)s:%(message)s')
        log_format_stream = logging.Formatter(
            '%(asctime)-.19s:%(levelname)s:%(name)s:%(message)s')

        # create logger
        self.log_dir = os.path.join(self.dir_bld, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        log_name = os.path.join(self.log_dir, 'output.log')
        log_input_name = os.path.join(self.log_dir, "input.log")
        self.set_up_logging(log_input_name, log_format_file, log_format_stream)
        aux_log.log_input(self)
        self.set_up_logging(log_name, log_format_file, log_format_stream)

        # directories for storing results as excel and pickle files
        self.xl_dir = os.path.join(self.dir_bld, "excel")
        os.makedirs(self.xl_dir, exist_ok=True)
        self.pkl_dir = os.path.join(self.dir_bld, "pkl")
        os.makedirs(self.pkl_dir, exist_ok=True)

        # Give user warning if project folder already existed
        if new_folder:
            self.logger.info(
                """
                Building folder already existed in project folder.
                Added enumerated building folder into project folder.
                """)

        # get input directory.
        # If it already exists use stored data else cluster raw data
        folder_cluster = "Clusters_" + str(self.opt_cluster["num_cluster"]) + \
            "_" + self.opt_cluster["clust_method"]
        os.makedirs(os.path.join(os.getcwd(), "inputs", folder_cluster,
                                 self.norm_loc, self.typ, self.age),
                    exist_ok=True)
        pkl_name = f"Quant{self.apart_quant}_Habitants{self.apart_habitant}_"
        pkl_name += f"Size{self.apart_size}_El_{self.el_dem}_"
        pkl_name += f"DHW_{self.dhw_dem}.pkl"
        self.input_filename = os.path.join(
            os.getcwd(), "inputs", folder_cluster,
            self.norm_loc, self.typ, self.age, pkl_name)

        self.logger.info('CREATED BUILDING OBJECT')
        self.logger.info('Obtaining clustered data')

        if os.path.exists(self.input_filename):
            self.logger.info('Using clustered data from input folder')
            self.read_clustered()
            self.num_cluster = self.clustered[list(
                self.clustered.keys())[0]].shape[0]
        else:
            self.get_raw_inputs()
            self.get_clusters()
            self.num_cluster = self.clustered[list(
                self.clustered.keys())[0]].shape[0]
            self.store_clustered()
        self.get_devs()

        # storing of inputs
        aux_save.store_inputs_xl(self)
        aux_save.store_inputs_pkl(self)

    def __del__(self):
        """
        By deleting the object, logging handlers need to be deleted too.
        Else the logging object keeps existing and by creating a new building
        object instance, a second logging object with identical duplicate
        handlers will be created
        """
        self.logger.handlers = []

    def set_up_logging(self, name, frmt_fle, frmt_stream):
        """
        Facilitates creating an instance of logging object and avoids duplicate
        logging instances

        Args:
            name (os.path): path to logging file
            frmt_fle (logging.Formatter): format of logged content
            frmt_stream (logging.Formatter): format of logged content
        """
        # set up logger if no logger is set yet
        if not self.logger.handlers:
            file_hand = logging.FileHandler(name)
            file_hand.setLevel(logging.DEBUG)
            file_hand.setFormatter(frmt_fle)

            stream_hand = logging.StreamHandler()
            stream_hand.setLevel(logging.INFO)
            stream_hand.setFormatter(frmt_stream)

            self.logger.addHandler(file_hand)
            self.logger.addHandler(stream_hand)
        # delete existing logging handlers and set up new one
        else:
            self.logger.handlers = []

            file_hand = logging.FileHandler(name)
            file_hand.setLevel(logging.DEBUG)
            file_hand.setFormatter(frmt_fle)

            stream_hand = logging.StreamHandler()
            stream_hand.setLevel(logging.INFO)
            stream_hand.setFormatter(frmt_stream)

            self.logger.addHandler(file_hand)
            self.logger.addHandler(stream_hand)

    def reset_mdl_(self):
        """
        Can be used to reset model and free memory
        """
        self.model = None
        self.mod_vars = {}
        self.mod_constr = {}

    def get_raw_inputs(self):
        """
        Reads and parses files from user input

        Raises:
            ValueError: See error message
        """
        self.logger.info('READING INPUTS')
        self.logger.debug('Reading in solar inputs')

        # reads in solar radiation data
        for unit in ["roof", "south", "east", "north", "west"]:
            fil_name = self.opt_fl_nms["weather_file"]
            fil_name += f"{self.norm_loc}_solar_{unit}.csv"
            if not os.path.isfile(fil_name):
                e_message = f'Could not find file {fil_name}\n Please \
                    check if {self.loc} is a valid location'
                self.logger.error(e_message)
                raise ValueError(e_message)
            self.raw_inputs['solar_' + unit] = \
                np.maximum(0, np.loadtxt(fil_name) / 1000)

        # reads in ambient temperature for given location
        self.logger.debug('Reading in temperature inputs')
        fil_name_temp = self.opt_fl_nms["weather_file"]
        fil_name_temp += f"{self.norm_loc}_temperature.csv"
        self.raw_inputs['temperature'] = np.loadtxt(fil_name_temp)

        # reads in internal gains, electricity and dhw profiles
        # based on building type, apartment quantity (if type=="MFH")
        # or aparment habitants (if type=="SFH")
        if self.typ == 'SFH' or self.typ == 'TH':

            self.logger.debug('Reading in electricity, dhw demands \
                and internal gains')
            fil_name = "raw_inputs/sfh/dhw_" + str(self.apart_habitant) + \
                "_" + self.dhw_dem + ".csv"

            self.raw_inputs['dhw'] = np.maximum(
                0, np.loadtxt(fil_name) / 1000) * self.apart_quant

            fil_name = "raw_inputs/sfh/electricity_" + str(
                self.apart_habitant) + "_" + self.el_dem + ".csv"
            self.raw_inputs['electricity'] = np.maximum(
                0, np.loadtxt(fil_name) / 1000) * self.apart_quant

            '''fil_name = "raw_inputs/sfh/int_gains_" + str(
                self.apart_habitant) + "_" + self.el_dem + ".csv"'''
            fil_name = "raw_inputs/sfh/internal_gains_angepasst.csv"
            self.raw_inputs['int_gains'] = np.maximum(
                0, np.loadtxt(fil_name) / 1000) * self.apart_quant

        elif self.typ == 'MFH' or self.typ == 'AB':

            self.logger.debug('Reading in electricity, dhw demands \
                and internal gains')
            fil_name = "raw_inputs/mfh/electricity_" + self.typ.lower() + \
                "_" + str(self.apart_quant) + ".csv"

            el_raw = np.maximum(0, np.loadtxt(fil_name) / 1000)

            fil_name = "raw_inputs/mfh/internal_gains_" + self.typ.lower()\
                + "_" + str(self.apart_quant) + ".csv"


            int_gains_raw = np.maximum(0, np.loadtxt(fil_name) / 1000)

            fil_name = "raw_inputs/mfh/dhw_" + self.typ.lower()\
                + "_" + str(self.apart_quant) + ".csv"

            dhw_raw = np.maximum(0, np.loadtxt(fil_name) / 1000)

            if self.el_dem == 'low':
                self.raw_inputs['electricity'] = el_raw * 0.75
                self.raw_inputs['int_gains'] = int_gains_raw * 0.75
            elif self.el_dem == 'medium':
                self.raw_inputs['electricity'] = el_raw
                self.raw_inputs['int_gains'] = int_gains_raw
            elif self.el_dem == 'high':
                self.raw_inputs['electricity'] = el_raw * 1.25
                self.raw_inputs['int_gains'] = int_gains_raw * 1.25
            if self.dhw_dem == 'low':
                self.raw_inputs['dhw'] = dhw_raw * 0.75
            elif self.dhw_dem == 'medium':
                self.raw_inputs['dhw'] = dhw_raw
            elif self.dhw_dem == 'high':
                self.raw_inputs['dhw'] = dhw_raw * 1.25

    def get_temp_design(self):
        """
        Reads in excel with data from DIN/TS 12831-1 and returns design outdoor
        temperature of given location. If location has multiple postal codes,
        the average over all postal codes is returned.

        Raises ValueError if location is not accessible in csv
        """
        # path to excel with weather data for all postal codes in Germany
        # file is provided by DIN/TS 12831-1
        filename = os.path.join(os.getcwd(), "raw_inputs", "weather_files",
                                "DIN_TS_12831-1_Klimadaten.xlsx")
        # read in data as pd.DataFrame
        df_weather_data = pd.read_excel(filename, skiprows=5, header=None,
                                        names=["ind", "post_code", "location",
                                               "design_outdoor_temp",
                                               "yrly_average_outdoor_temp",
                                               "height", "skip1", "skip2"],
                                        index_col=0)
        # return design temperature for location
        if not pd.isna(round(df_weather_data[
                df_weather_data["location"] == self.loc][
                "design_outdoor_temp"].mean(), 2)):
            return round(df_weather_data[
                df_weather_data["location"] == self.loc][
                    "design_outdoor_temp"].mean(), 2)
        # if the location is not exactly equal to values in DataFrame,
        # match is tried to find location differing by special cases or similar
        elif not pd.isna(round(df_weather_data[
                df_weather_data["location"].str.match(self.loc)][
                "design_outdoor_temp"].mean(), 2)):
            self.logger.warning(
                f"""Could not match given location word-for-word with values of
                location column in {filename}
                Using matching now for accessing design outdoor temperature
                which might include other locations than desired!""")
            return round(df_weather_data[
                df_weather_data["location"].str.match(self.loc)][
                    "design_outdoor_temp"].mean(), 2)
        # if the location cannot be matched to any value in DataFrame
        # ValueError is raised
        else:
            error_text = \
                f"""Could not find nor match given location - {self.loc} -
                to any locations of\n
                {filename} for setting design outdoor temperature"""

            self.logger.error(error_text)
            raise ValueError(error_text)

    def get_clusters(self):
        """
        Based on provided input data, a K-medoid clustering algorithm is called
        and merges datapoints such that only num_cluster days are left and the
        sum of deviations from the clustered day to its originals is minimized.
        Weights display the impact of the different input quantities.
        """

        self.logger.debug('Starting Clustering')

        # array with lists of input data
        inputs_clustering = np.array([self.raw_inputs["electricity"],
                                      self.raw_inputs["dhw"],
                                      self.raw_inputs["solar_roof"],
                                      self.raw_inputs["temperature"],
                                      self.raw_inputs["solar_south"],
                                      self.raw_inputs["solar_west"],
                                      self.raw_inputs["solar_east"],
                                      self.raw_inputs["solar_north"],
                                      self.raw_inputs["int_gains"]
                                      ])
        # calling clustering function with input array and clustering params
        (clu_inp, nc, z) = clustering.cluster(inputs_clustering,
                                              self.opt_cluster["num_cluster"],
                                              norm=2,
                                              mip_gap=0.0,
                                              weights=[8, 8, 8, 3,
                                                       1, 1, 1, 1, 1],
                                              method=self.opt_cluster["clust_method"])  # NOQA: E501
        # set further attributes
        self.len_day = int(inputs_clustering.shape[1] / 365)
        self.clustered = {"electricity": clu_inp[0],
                          "dhw": clu_inp[1],
                          "solar_roof": clu_inp[2],
                          "temp_ambient": clu_inp[3],
                          "solar_s": clu_inp[4],
                          "solar_w": clu_inp[5],
                          "solar_e": clu_inp[6],
                          "solar_n": clu_inp[7],
                          "int_gains": clu_inp[8],
                          "weights": nc,
                          "temp_indoor": 20,
                          "temp_design": self.get_temp_design()
                          }
        self.clustered['temp_delta'] = \
            np.maximum(0, (self.clustered["temp_indoor"] -
                           self.clustered["temp_ambient"]))

    def get_devs(self):
        """
        reads in and parses information provided in csv files in raw_inputs
        """
        self.logger.debug('Reading in Devices')

        # read in csv file devices
        self.params["devs"] = pik.read_devices(
            temp_ambient=self.clustered["temp_ambient"],
            sol_irrad=self.clustered["solar_roof"],
            filename=self.opt_fl_nms["dev_file"])

        # read in and parse economic parameters
        self.params = pik.read_economics(self.params,
                                         filename=self.opt_fl_nms["eco_file"])
        if self.opt_model["nw_bld"]:
            self.params['alpha'] = 0
        else:
            self.params['alpha'] = 1
        if self.typ == 'MFH' or self.typ == 'AB':
            self.params['MFH'] = 1
        else:
            self.params['MFH'] = 0

        # read in and parse subsidy related informations
        self.params["subs_params"] = pik.read_subsidies(
            self.params["eco"], filename=self.opt_fl_nms["subs_file"])

        # read in building parameters and create reference building
        self.buildings = pik.read_building_dimension(
            self.opt_fl_nms["bld_file"])
        self.scenarios = pik.retrofit_scenarios(self.buildings)

        self.params["building"] = {}
        self.params["building"]["U-values"] = \
            self.scenarios[self.typ][self.age]
        self.params["building"]["dimensions"] = \
            self.buildings[self.typ][self.age]
        self.params["building"]["usable_roof"] = \
            self.opt_model["useable_roofarea"]
        self.params["building"]["dimensions"]["Area"] = \
            self.apart_quant * self.apart_size
        self.params["G_t"] = \
            self.get_gradtagszahl(self.clustered["temp_delta"],
                                  self.clustered["weights"])
        '''self.params["ref_building"] = ref_bui.reference_building(
            self.params["building"]["dimensions"],
            self.params["G_t"])'''
        self.params["ref_building"] = ref_bui.reference_building_angepasst(
            self.params["building"]["dimensions"],
            self.params["G_t"])

        self.logger.info('Adding model parameters')
        self.params['clustered'] = self.clustered
        if self.opt_model["exist_boiler"]:
            # In order to implement existing boiler but not oversize it
            # gas boiler can be bought without fixe invest costs and only
            # half of variable invest cost (arbitrary value)
            self.params["devs"]["gas_hy"]["c_inv_fix"] = 0
            self.params["devs"]["gas_hy"]["c_inv_var"] = \
                self.params["devs"]["gas_hy"]["c_inv_var"] * 0.5
        self.params = set_params(
            self.params, self.apart_quant, self.opt_model)
        self.clear_properties()

    def clear_properties(self):
        """
        deletes temporary data which has been passed to other attributes
        in order to free memory
        """
        del self.clustered
        del self.raw_inputs
        del self.buildings
        del self.scenarios

    def opti_model(self, log: bool = True):
        """
        Creates optimization model with given user input

        Args:
            log (bool, optional): Activates logging. Defaults to True.
        """

        if log:
            self.logger.info('START BUILDING OPTIMIZATION MODEL')

        self.model = gp.Model('Single_Building_Optimization')
        if log:
            self.logger.info('Adding economic variables')
        self.model, self.mod_vars = set_vars_eco(
            self.model, self.mod_vars, self.params)
        if log:
            self.logger.info('Adding technical variables')
        self.model, self.mod_vars = set_vars_tec(
            self.model, self.mod_vars, self.params)
        if log:
            self.logger.info('Adding restructure measurements variables')
        self.model, self.mod_vars = set_vars_restruc_meas(
            self.model, self.mod_vars, self.params)
        if log:
            self.logger.info('Adding subsidy variables')
        self.model, self.mod_vars = set_vars_sub(
            self.model, self.mod_vars, self.params)
        if log:
            self.logger.info('Adding objective function')
        if self.opt_model["opt_costs"]:
            self.model.setObjective(self.mod_vars['c_total'],
                                    gp.GRB.MINIMIZE)
        else:
            self.model.setObjective(self.mod_vars['emission'],
                                    gp.GRB.MINIMIZE)

        # %% Capacity bounds and building related constraints
        if log:
            self.logger.info('Adding building constraints')
        self.model, self.mod_constr = set_constr_building(
            self.model, self.mod_vars, self.mod_constr, self.params)
        # %% Investments, operation and maintenance, demand related costs,
        # fixed administration costs
        if log:
            self.logger.info('Adding cost constraints')
        self.model, self.mod_constr = set_constr_costs(
            self.model, self.mod_constr, self.mod_vars, self.params)
        # %% Space Heating, Vent/Heat Loss, Solar Gains, Heating/Storage
        # Systems, Thermal/Electricity Balance, DSH, CO2
        if log:
            self.logger.info('Adding technical constraints')
        self.model, self.mod_constr = set_constr_tech(
            self.model, self.mod_constr, self.mod_vars, self.opt_model,
            self.params)
        # %% Subsidies
        if log:
            self.logger.info('Adding subsidy constraints')
        self.model, self.mod_constr = set_constr_subs(
            self.model, self.mod_constr, self.mod_vars, self.opt_subs,
            self.params)
        # %% Scenarios
        if log:
            self.logger.info('Adding scenario constraints')
        self.model, self.mod_constr = set_constr_scenario(
            self.model, self.mod_constr, self.mod_vars, self.opt_model,
            self.params)

        """self.model, self.mod_constr = set_constr_beg(
            self.model, self.mod_constr, self.mod_vars, self.opt_subs,
            self.params)"""

        # set gurobi parameters based on user input (or defaults)
        self.model.Params.LogToConsole = self.grb_params['LogToConsole']
        self.model.Params.OutputFlag = self.grb_params['OutputFlag']
        if self.grb_params['LogFile']:
            self.model.Params.LogFile = os.path.join(
                self.log_dir, self.grb_params['LogFile'])
        self.model.Params.TimeLimit = self.grb_params['TimeLimit']
        self.model.Params.MIPGap = self.grb_params['MIPGap']
        self.model.Params.NumericFocus = self.grb_params['NumericFocus']
        self.model.Params.MIPFocus = self.grb_params['MIPFocus']
        self.model.Params.Aggregate = self.grb_params['Aggregate']
        self.model.Params.PoolSearchMode = self.grb_params['PoolSearchMode']
        self.model.Params.PoolSolutions = self.grb_params['PoolSolutions']
        self.model.Params.PoolGap= self.grb_params['PoolGap']


        self.model.update()

    def run_opti(self):
        """
        Calls gurobi optimize and raises Gurobi Error

        Raises:
            GurobiError: GoodLuck!
        """
        try:
            self.logger.info('START SOLVING MODEL')
            self.model.optimize()

        except gp.GurobiError as e:
            self.logger.error("Error: " + e.message)
            raise GurobiError(10023, e.message)
        #print("hello")
    def postprocess_model(self):
        """
        Postprocessing optimization results.
        If model has a solution, gurobi files and human readable excel file are
        created. Also, a pickle file is provided.

        If the model is either unfeasible or unbounded and
        the argument "infeas_unbound_process" is set to True, a postprocessing
        for further informations is started
        """
        try:
            # case: feasible solution; writes results files
            if not self.model.solCount == 0:
                if self.grb_params['write']:
                    self.model.write(os.path.join(
                        self.log_dir, self.grb_params['write']))
                    self.model.write(os.path.join(
                        self.log_dir,
                        self.grb_params['write'].split('.')[0] + '.mps'))
                aux_log.log_output(self)
                aux_save.store_output_xl(self)
                aux_save.store_output_pkl(self)

            # case: infeasible or unbounded; sets gurobi parameter
            elif self.infeas_unbound_process:
                self.model.Params.DualReductions = 0
                """
                From Gurobi Documentation:
                Disables dual reductions in presolve.
                """
                self.model.Params.InfUnbdInfo = 1
                """
                From Gurobi Documentation:
                Additional info for infeasible/unbounded models.
                !!!! Translates the MIP to an LP !!!!
                Obtained solution is only for querying additional information
                about the infeasible or unbound model
                """
                self.model.optimize()

                if self.model.MIPGap == float("inf"):
                    self.logger.info("Model found to be infeasible")

                    self.model.computeIIS()
                    self.model.write("infeas_model.ilp")
                    list_constr = self.model.getConstrs()
                    infeas_constr = []
                    for c in list_constr:
                        if c.IISConstr:
                            infeas_constr.append(c.ConstrName)
                    self.logger.info(f'''
                        Set of Infeasible Constraints:
                        {infeas_constr}''')
                else:
                    self.logger.info("Model found to be unbounded")
            else:
                self.logger.info("Model is either infeasible or unbounded")

        except AttributeError:
            self.logger.info("Model is unbounded")

    def store_clustered(self):
        """
        Storing clustered data as pickle file such that identical cluster data
        can be accessed without extra clustering process
        """

        self.logger.info("Storing Inputs")

        with open(self.input_filename, "wb") as f_in:
            pkl.dump(self.clustered, f_in, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.len_day, f_in, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.raw_inputs, f_in, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.num_cluster, f_in, pkl.HIGHEST_PROTOCOL)

    def read_clustered(self):
        """
        Reads in clustered data from pickle file
        """

        with open(self.input_filename, "rb") as f_in:
            self.clustered = pkl.load(f_in)
            self.len_day = pkl.load(f_in)
            self.raw_inputs = pkl.load(f_in)
            self.num_cluster = pkl.load(f_in)

    def set_energy_prices(self, p_el: float = None, p_el_hp: float = None,
                          p_gas: float = None, p_pel: float = None):
        if isinstance(p_el, float):
            self.params["eco"]["el"]["el_sta"]["var"] = p_el
        if isinstance(p_el_hp, float):
            self.params["eco"]["el"]["el_hp"]["var"] = p_el_hp
        if isinstance(p_gas, float):
            self.params["eco"]["gas"]["gas_sta"]["var"] = p_gas
        if isinstance(p_pel, float):
            self.params["eco"]["pel"]["pel_sta"]["var"] = p_pel

    @staticmethod
    def get_gradtagszahl(temp_delta: dict, weights: dict) -> float:
        """
        calculates Gradtagszahl based on delta T and weights

        Args:
            temp_delta (dict): outdoor - indoort temperature difference
            weights (dict): cluster weights,
                            i.e. how many days the cluster represents

        Returns:
            Gradtagszahl (float): accumulated temperature difference in a year
            (disregarding days with average temperature above 13 °C and
             hours with temperature above 15 °C)
        """
        g_t = 0
        for day in range(temp_delta.shape[0]):
            # only consider days with average temperature below 13 °C
            if np.mean(temp_delta[day]) > 7:
                # only consider temperatures below 15 °C
                g_t = g_t + sum(temp_delta[day][j]
                                for j in range(temp_delta.shape[1])
                                if temp_delta[day, j] > 5) * weights[day]

        # unit of Gradtagszahl is K*d --> divide by 24 h/d
        return g_t / 24


if __name__ == "__main__":

    proj_name = 'MFH_multiple solutions'

    # OPTIMIZE: Gradtagszahl from raw input data?
    # Building parameters:
    bld_type = "MFH"
    bld_age = "1958_1968"
    location = "Potsdam"
    apart_quantity = 8
    apart_size = 574.8/8
    apart_habitant = 2
    electricity_demand = "medium"
    dhw_demand = "medium"

    tracemalloc.start()
    time_start = time.perf_counter()

    build = Building(
        typ=bld_type,
        age=bld_age,
        loc=location,
        apart_quant=apart_quantity,
        apart_size=apart_size,
        apart_habitant=apart_habitant,
        opt=Options(project_name=proj_name,
                    infeas_unbound_process=True))

    build.opti_model()
    #build.set_energy_prices(p_el=0.25, p_el_hp=0.2, p_gas=0.15, p_pel=0.12)
    ''' build.model.addConstr(build.mod_vars["capacity"]["gas_hy"] <= 10)
    build.model.addConstr(build.mod_vars["soc_nom"]["tes"] <= 4.2)'''
    #build.model.addConstr(build.mod_vars["x"]["gas_hy"] == 0)
    build.run_opti()
    build.postprocess_model()

    # track memory usage
    current, peak = tracemalloc.get_traced_memory()
    build.logger.info(
        f"""\n
        Current memory usage is {round(current / 10 ** 6, 2)}MB
        Peak was {round(peak / 10 ** 6, 2)}MB\n
        Gurobi solving time: {round(build.model.Runtime, 2)}s\n
        Total elapsed time: {round(time.perf_counter()-time_start, 2)} s\n"""
    )

    build.logger.info("finished")
