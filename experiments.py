"""experiments.py: contains functions, describing each performed experiment"""

import multiprocessing as mp

import numpy as np
import pandas as pd

from config_usr import CONFIG_USR
from model import DFType, Model
import output as out
from pareto import Pareto, PFType
from system_info import SystemInfo

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


def start_experiments():
    """
    Starts experiments executions for each studied configurable system

    :return: None
    """

    # For each studied software system:
    sys_infos = SystemInfo.get_all_infos()
    for sys_info in sys_infos.values():

        # Perform the following experiments:
        # * Train & validate predictors & transferrers
        # * Build & validate Pareto Frontiers
        experiment_models_validation(sys_info)
        experiment_pareto_2d(sys_info)


# Train & validate all predictors & transferrers #############################
def experiment_models_validation(si: SystemInfo):
    """
    Parallelize experiment on assessing predictors' and transferers' quality
    for a studied configurable software system

    :param si: studied system
    :return: None
    """

    # Vars:
    # * processes: parallel processes for independent experiment repetitions
    # * manager: manager of objects, shared across parallel processes
    # * res_on: export results?
    # * res_dict: dictionary with experimental results from each process
    # * res_dir: directory to export experimental results
    processes = []
    manager = mp.Manager()
    res_on = True
    res_dict = manager.dict()
    res_dir = 'results/experiment_models_validation'

    # Start processes
    for i in range(CONFIG_USR['repetitions']):
        process = mp.Process(target=_experiment_models_validation,
                             args=(si, res_dict, i))
        processes += [process]
        process.start()

    # Join processes
    for process in processes:
        process.join()

    # Finalize resulting dataframe
    full_df = pd.concat(res_dict.values())
    group_cols = [
        'target', 'source_id', 'dest_id', 'predictor', 'transferer',
        'train_size', 'transfer_size', 'metric_a', 'metric_b']
    full_df = full_df.groupby(by=group_cols, as_index=False).mean()
    out.data(full_df, res_dir, f'{si.sys_name}_models_validation_df', res_on)


def _experiment_models_validation(si: SystemInfo, return_dict: dict,
                                  seed: int):
    """
    Execute a single replication of the experiment on assessing predictors'
    and transferers' quality for a studied configurable software system

    :param si: studied system
    :param return_dict: dictionary to aggregate results from each replication
    :param seed: random seed of the current replication
    :return: None
    """

    np.random.seed(seed)

    # VARIABLES ##############################################################
    # * dbg_msg: print debugging messages
    # * dbg_data: export debugging data
    # * dbg_dir: directory to export debugging dataframes
    dbg_msg = True
    dbg_data = False
    dbg_dir = 'debug/experiment_models_validation'

    # EXPERIMENTAL DESIGN CREATION ###########################################
    # (a) Benchmarking targets (text, audio, video, code, ...)
    # (b) Source hardware IDs
    # (c) Destination hardware IDs
    # (d) Predictor name (CART, bagging, forest, ...)
    # (e) Transferer name (Linear, MARS, ...)
    # (f) Predictor training sample size
    # (f) Transferer training sample size
    # (g) Metric A (compression size, compression time, ...)
    # (h) Metric B (compression size, compression time, ...)
    design_lists = \
        [[a, b, c, d, e, f, f, g, h]
         for a in si.targets
         for b in si.servers_source
         for c in si.servers_dest
         for d in CONFIG_USR['predictors']
         for e in CONFIG_USR['transferers']
         for f in si.train_sizes  # Duplicated (train_size & transfer_size)
         for g in si.metrics
         for h in si.metrics
         if g > h  # use metrics' pair only one time
         ]

    # Finalize design dataframe
    design_cols = [
        'target', 'source_id', 'dest_id', 'predictor', 'transferer',
        'train_size', 'transfer_size', 'metric_a', 'metric_b']
    design_df = pd.DataFrame(design_lists, columns=design_cols)

    # Debug info
    out.msg(f'Experimental Design is created: {len(design_df)}', dbg_msg)
    out.data(design_df, dbg_dir, f'{si.sys_name}_01_design_df', dbg_data)

    # EXPAND DESIGN WITH RESULTS #############################################
    models_rows = []

    for row_id, row in design_df.iterrows():
        # Monitoring
        if row_id % 10 == 0:
            out.msg(f'Design row ID: {row_id}', dbg_msg)

        # Resulting rows
        models_row = row.copy()

        # Get UNTRAINED models for each metric
        models_a = Model.get_untrained_models(si, row, 'metric_a')
        models_b = Model.get_untrained_models(si, row, 'metric_b')

        # Validate models for each metric
        models_row = Model.validate_models(models_a, models_row)
        models_row = Model.validate_models(models_b, models_row)

        # Append row
        models_rows.append(models_row)

    # Queue experimental results
    models_df = pd.DataFrame(models_rows)
    return_dict[seed] = models_df


# Build & validate all Pareto Frontiers ######################################
def experiment_pareto_2d(si: SystemInfo):
    """
    Parallelize experiment on approximation and transferring of Pareto fronts
    for a studied configurable software system

    :param si: studied system
    :return: None
    """

    # Vars:
    # * processes: parallel processes for independent experiment repetitions
    # * manager: manager of objects, shared across parallel processes
    # * approx_dict: dictionary with approximate Pareto frontier results
    # * tran_dict: dictionary with transferred Pareto frontier results
    # * res_on: export results?
    # * res_dir: directory to export experimental results
    processes = []
    manager = mp.Manager()
    approx_dict = manager.dict()
    tran_dict = manager.dict()
    res_on = True
    res_dir = 'results/experiment_pareto_2d'

    # Start processes
    for i in range(CONFIG_USR['repetitions']):
        process = mp.Process(target=_experiment_pareto_2d,
                             args=(si, approx_dict, tran_dict, i))
        processes += [process]
        process.start()

    # Join processes
    for process in processes:
        process.join()

    # Finalize resulting dataframe
    approx_full_df = pd.concat(approx_dict.values())
    tran_full_df = pd.concat(tran_dict.values())

    group_cols = [
        'target', 'source_id', 'dest_id', 'predictor', 'transferer',
        'train_size', 'transfer_size', 'metric_a', 'metric_b']
    approx_df = approx_full_df.groupby(by=group_cols, as_index=False).mean()
    transf_df = tran_full_df.groupby(by=group_cols, as_index=False).mean()

    out.data(approx_df, res_dir, f'{si.sys_name}_approx_df', res_on)
    out.data(transf_df, res_dir, f'{si.sys_name}_transf_df', res_on)


def _experiment_pareto_2d(si: SystemInfo, approx_dict: dict,
                          tran_dict: dict, seed: int):
    """
    Execute a single replication of the experiment on approximation and
    transferring of Pareto fronts for a studied configurable software system

    :param si: studied system
    :param approx_dict: dictionary to aggregate results of each replication on
                        approximation of Pareto frontiers
    :param tran_dict: dictionary to aggregate results of each replication on
                      transferring of Pareto frontiers
    :param seed: random seed of the current replication
    :return: None
    """

    np.random.seed(seed)

    # VARIABLES:
    # * dbg_msg: print debugging messages
    # * dbg_data: export debugging dataframes
    # * dbg_dir: export dir for CSV
    dbg_msg = True
    dbg_data = False
    dbg_dir = 'debug/experiment_pareto_2d'
    out.msg(f'Started Pareto experiment on system: {si.sys_name}', dbg_msg)

    # EXPERIMENTAL DESIGN CREATION ###########################################
    # * Benchmarking targets (text, audio, video, code, ...)
    # * Source hardware IDs
    # * Destination hardware IDs
    # * Predictor name (CART, bagging, forest, ...)
    # * Transferer name (Linear, MARS, ...)
    # * Predictor training sample size
    # * Transferer training sample size
    # * Configuration ID
    # * Metric A (compression size, compression time, ...)
    # * Metric B (compression size, compression time, ...)
    design_lists = \
        [[a, b, c, d, e, f, f, i, j]
         for a in si.targets
         for b in si.servers_source
         for c in si.servers_dest
         for d in CONFIG_USR['predictors']
         for e in CONFIG_USR['transferers']
         for f in si.train_sizes  # Duplicated (train_size & transfer_size)
         for i in si.metrics
         for j in si.metrics
         if i > j  # use metrics' pair only one time
         ]

    design_cols = [
        'target', 'source_id', 'dest_id', 'predictor', 'transferer',
        'train_size', 'transfer_size', 'metric_a', 'metric_b']

    design_df = pd.DataFrame(design_lists, columns=design_cols)
    out.data(design_df, dbg_dir, f'{si.sys_name}_01_design_df', dbg_data)
    out.msg(f'Experimental Design is created: {len(design_df)}', dbg_msg)

    # EXPAND DESIGN WITH RESULTS #############################################
    pareto_dfs = []
    approx_rows = []
    tran_rows = []

    for row_id, row in design_df.iterrows():
        # Monitoring
        if row_id % 10 == 0:
            out.msg(f'Design row ID: {row_id}', dbg_msg)

        # PARETO FRONT & CONFUSION MATRIX ####################################
        # Vars:
        # * Optimization objectives
        objectives = {
            'metric_a': 'time_compress',
            'metric_b': 'size_compress'
        }

        # Get Pareto frontier template dataframe
        temp_df = Pareto.get_template_df(si, row, objectives)

        # Train models on USER_DF
        models_a = Model.get_untrained_models(si, row, 'metric_a')
        models_b = Model.get_untrained_models(si, row, 'metric_b')
        Model.train_models(models_a, DFType.USER_DF)
        Model.train_models(models_b, DFType.USER_DF)

        # Predict and transfer all metrics' values using trained models
        Pareto.prepare_metrics(temp_df, models_a, models_b)

        # Build all Pareto fronts
        Pareto.build_fronts(temp_df)

        # Polish temp_df for easier comparison of metrics
        temp_df = Pareto.polish_temp_df(temp_df, si)

        # Generate confusion matrices for approximated & transferred frontiers
        approx_row = Pareto.get_confusion_row(temp_df, row, PFType.APPROX_PF)
        tran_row = Pareto.get_confusion_row(temp_df, row, PFType.TRANSFER_PF)

        # Aggregate resulting row
        pareto_dfs.append(temp_df)
        approx_rows.append(approx_row)
        tran_rows.append(tran_row)

    # COMBINE RESULTS DATAFRAMES #############################################
    # Debug template dataframe
    pareto_df = pd.concat(pareto_dfs, ignore_index=True)
    out.data(pareto_df, dbg_dir, f'{si.sys_name}_02_pareto_df', dbg_data)

    # Finalize experimental results
    approx_df = Pareto.polish_confusion_df(pd.DataFrame(approx_rows))
    tran_df = Pareto.polish_confusion_df(pd.DataFrame(tran_rows))

    # Return experimental results
    approx_dict[seed] = approx_df
    tran_dict[seed] = tran_df
