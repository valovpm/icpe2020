"""system_info.py: Class, describing studied configurable software systems."""

import numpy as np
import pandas as pd

from config_sys import CONFIG_SYS
from config_usr import CONFIG_USR
import output as out

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


class SystemInfo:

    def _load_benchmarking_data(self):
        """
        Loads benchmarking data and averages benchmarked system properties
        (compression time, compression size) for a specified configurable
        software system (self)

        :return: dataframe containing averaged benchmarking data
        """

        # Load benchmarking results CSV
        data = pd.read_csv(f'data/{self.sys_name}.csv')

        # Group and average performance metrics
        cols = ['system', 'target', 'server'] + self.features
        df = data.groupby(cols, as_index=False)
        df = df.aggregate(np.mean)

        # Keep necessary columns
        cols = cols + self.metrics
        df = df[cols]
        return df

    def __init__(self, sys_name: str):
        """
        Loads studied system's configuration and benchmarking data,
        calculates data-based system properties and training sizes

        :param sys_name: Name of a system to be initialized
        """

        # Load system configuration info based on system name:
        # * sys_name: system name
        # * features: system features
        # * metrics: system performance metrics
        # * nf: Number of features
        # * nm: Number of performance metrics
        # * nt: Number of benchmarking targets
        self.sys_name = sys_name
        self.features = CONFIG_SYS['features'][sys_name]
        self.metrics = CONFIG_SYS['metrics'][sys_name]
        self.targets = CONFIG_SYS['targets'][sys_name]
        self.nf = len(self.features)
        self.nm = len(self.metrics)
        self.nt = len(self.targets)

        # Data-based system info:
        # * Load benchmarking data
        # * Unique hardware environments
        # * Source servers (make sure they are valid)
        # * Destination servers (make sure they are valid)
        # * Configuration space size
        self.data = self._load_benchmarking_data()
        self.servers = sorted(self.data['server'].unique())
        self.servers_source = list(set(CONFIG_USR['servers_source']) &
                                   set(self.servers))
        self.servers_dest = list(set(CONFIG_USR['servers_dest']) &
                                 set(self.servers))
        self.space_size = 1
        for f in self.features:
            self.space_size = self.space_size * len(self.data[f].unique())

        # Models' training sample sizes
        min_size = 3  # 3 observations: 2 for training and 1 for validation
        max_size = self.space_size  # Grow to maximal possible size
        self.train_sizes = range(min_size, max_size + 1)
        self.transfer_sizes = range(min_size, max_size + 1)

    @staticmethod
    def get_all_infos():
        """
        Initializes all studied configurable software systems

        :return: a dictionary of initialized system objects
        """

        # Variables:
        # * dbg_msg: print debugging messages
        # * sys_infos: benchmarking results of all systems
        dbg_msg = True
        sys_infos = {}

        # Load benchmarking results for each system
        for sys_name in CONFIG_USR['systems']:
            sys_info = SystemInfo(sys_name)
            sys_infos[sys_name] = sys_info
            out.msg(f'Loaded {sys_name} results. '
                    f'Rows: {len(sys_info.data)}', dbg_msg)

        return sys_infos

    @staticmethod
    def load_models_validation_results(si: 'SystemInfo'):
        """
        Load results of the experiment on prediction and transferring models
        validation for a specified studied configurable software system

        :param si: SystemInfo of a studied system, whose results are loaded
        :return: results of the validation experiment for the selected system
        """
        models_df = pd.read_csv(
            f'results/experiment_models_validation/'
            f'{si.sys_name}_models_validation_df.csv')
        return models_df

    @staticmethod
    def load_approx_frontier_results(si: 'SystemInfo'):
        """
        Load results of the experiment on Pareto Frontier approximation
        for a specified studied configurable software system

        :param si: SystemInfo of a studied system, whose results are loaded
        :return: results of the Pareto approximation experiment for the
        selected system
        """
        models_df = pd.read_csv(
            f'results/experiment_pareto_2d/'
            f'{si.sys_name}_approx_df.csv')
        return models_df

    @staticmethod
    def load_transfer_frontier_results(si: 'SystemInfo'):
        """
        Load results of the experiment on Pareto Frontier transferring
        for a specified studied configurable software system

        :param si: SystemInfo of a studied system, whose results are loaded
        :return: results of the Pareto transferring experiment for the
        selected system
        """
        models_df = pd.read_csv(
            f'results/experiment_pareto_2d/'
            f'{si.sys_name}_transf_df.csv')
        return models_df
