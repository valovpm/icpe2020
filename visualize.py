"""visualize.py: contains functionality for plotting"""

import math
from pathlib import Path

import matplotlib.pyplot as mpl
import cycler
import numpy as np
import pandas as pd

from config_usr import CONFIG_USR
from model import MName
import output as out
from system_info import SystemInfo

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


def visualize():
    """
    Performs visualization of all necessary plots for
    data exploration and analysis

    :return: None
    """

    # Get studied configurable software systems
    systems = SystemInfo.get_all_infos()
    systems = sorted(systems.values(), key=lambda s: s.sys_name)

    # Plot metric's distributions for all systems and servers
    plot_metric_distros(systems, 'time_compress')

    # Plot models' accuracy results for each system
    plot_models_validation(systems, 'time_compress', 'StandardF2sv2-westus2')

    # Plot measures for Pareto frontiers
    plot_pareto_measures(systems, 'BasicA1-japaneast',
                         'StandardF2sv2-westus2')


# PLOT METRIC'S DISTRIBUTIONS FOR ALL SYSTEMS AND HARDWARE ###################
def plot_metric_distros(systems: list, metric: str):
    """
    For each studied software system, plot distributions of a specified
    performance metric on each available hardware environment

    :param systems: studied configurable software systems
    :param metric: studied performance metric
    :return: None
    """

    # Vars:
    n_sys = len(systems)

    # Figure that contains plots for all systems
    fig = mpl.figure(figsize=(8.5, 2), dpi=600)

    # Add subplot for each system:
    for i_sys, system in enumerate(systems):
        # Select plot data
        space_size = system.space_size
        sys_data = system.data
        data_cols = ['server'] + system.features + [metric]
        sys_data = sys_data[data_cols]

        # Reorganize & sort data
        sys_data = sys_data.pivot_table(index=system.features,
                                        columns='server',
                                        values=metric)
        sys_data = sys_data.sort_values(system.servers[0], ascending=True)
        sys_data.reset_index(drop=True, inplace=True)

        # Add subplot (axes) to figure
        ax = fig.add_subplot(1, n_sys, i_sys+1)
        ax.set_title(system.sys_name)
        ax.tick_params(axis='x', labelsize='8')
        ax.tick_params(axis='y', labelsize='8')

        # Generate x-ticks & x-labels
        x_ticks = np.arange(0, space_size - 1, (space_size - 1) / 10)
        x_ticks = np.append(x_ticks, [space_size - 1])
        ax.set_xlim(0, space_size - 1)
        ax.set_xticks(x_ticks)
        x_labels = np.round(x_ticks, 0).astype(int) + 1
        ax.set_xticklabels(x_labels, rotation='vertical')

        # Generate y-ticks & y-labels
        y_max = sys_data.max().max()
        y_max_tick = int(math.ceil(y_max / 100.0)) * 100
        y_ticks = np.arange(0, y_max_tick + 1, y_max_tick / 10)
        ax.set_ylim(0, y_max_tick)
        ax.set_yticks(y_ticks)

        # Add grid to the plot
        mpl.grid(linestyle=':', linewidth=0.5)

        # Add lines (for each server) to plot
        for server in system.servers:
            x = list(sys_data[server])
            ax.plot(x, linewidth=0.5)

    # Add figure-wide labels
    fig.text(0.5, 0.0, 'Configurations', ha='center', va='center')
    fig.text(0.0, 0.5, 'Compression time', ha='center', va='center',
             rotation='vertical')

    # Save ready figure
    mpl.tight_layout()
    dir_name = 'results/dists_systems_metric'
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    fig.savefig('{0}/dist_systems_{1}'.format(dir_name, metric),
                bbox_inches='tight', pad_inches=0)


# PLOT METRIC'S MODELS' ERRORS ###############################################
def plot_models_validation(systems: list, metric: str, server_dest: str):
    """
    For each studied system, plot metric predictor and transferer validation
    measures, when transferring predictors from every available source
    hardware environment to the specified destination hardware environment

    :param systems: studied configurable software systems
    :param metric: studied performance metric
    :param server_dest: destination hardware environment, where predictors
                        from every available hardware are transferred
    :return: None
    """

    # VARIABLES ##############################################################
    # * dbg_msg: print debugging messages
    # * dbg_data: export debugging data
    # * dbg_dir: directory to export debugging dataframes
    # * Number of systems
    dbg_msg = True
    dbg_data = False
    dbg_dir = 'debug/plot_models_validation'

    # PLOTTING DESIGN CREATION ###############################################
    # (a) Predictor name (linear, tree, etc.)
    # (b) Transferer name (linear, tree, etc.)
    design_lists = \
        [[a, b]
         for a in CONFIG_USR['predictors']
         for b in CONFIG_USR['transferers']]

    # Finalize design dataframe
    design_cols = ['predictor', 'transferer']
    design_df = pd.DataFrame(design_lists, columns=design_cols)

    # Debug info
    out.msg(f'Plotting Design is created: {len(design_df)}', dbg_msg)
    out.data(design_df, dbg_dir, f'design_df', dbg_data)

    # EXPAND DESIGN WITH RESULTS #############################################
    for row_id, row in design_df.iterrows():
        predictor = row['predictor']
        transferer = row['transferer']
        pred_data_full = pd.DataFrame()
        tran_data_full = pd.DataFrame()

        # Create a separate subplot for each system:
        for i_sys, si in enumerate(systems):

            # Load system's models' accuracy results data
            data = SystemInfo.load_models_validation_results(si)
            data['sys_name'] = si.sys_name
            data['space_size'] = si.space_size

            # Filter rows
            data = data[(data['target'].isin(si.targets)) &
                        (data['predictor'] == predictor) &
                        (data['transferer'] == transferer) &
                        (data['dest_id'] == server_dest) &
                        (data['train_size'] >= 3) &
                        (data['transfer_size'] >= 3)]

            # Rename error columns
            data.rename(inplace=True, columns={
                f'{metric}|source_pred|mape_loo_mean': 'pred_error',
                f'{metric}|transferer|mape_loo_mean': 'tran_error'})

            # Filter necessary columns
            data = data[[
                'sys_name', 'space_size', 'source_id', 'dest_id',
                'predictor', 'transferer', 'train_size', 'transfer_size',
                'pred_error', 'tran_error']]

            # Split data for predictor and transferer
            pred_data = data[['sys_name', 'space_size', 'source_id',
                              'predictor', 'train_size', 'pred_error']]

            tran_data = data[['sys_name', 'space_size', 'source_id',
                              'transferer', 'transfer_size', 'tran_error']]

            # Group data
            pred_data = pred_data.groupby(
                by=['sys_name', 'space_size', 'source_id',
                    'predictor', 'train_size'],
                as_index=False).mean()

            tran_data = tran_data.groupby(
                by=['sys_name', 'space_size', 'source_id',
                    'transferer', 'transfer_size'],
                as_index=False).mean()

            # Rename columns to make data homogeneous
            pred_data.rename(columns={'source_id': 'server',
                                      'train_size': 'size',
                                      'pred_error': 'error'}, inplace=True)

            tran_data.rename(columns={'source_id': 'server',
                                      'transfer_size': 'size',
                                      'tran_error': 'error'}, inplace=True)

            # Aggregate data for all systems
            pred_data_full = pd.concat([pred_data_full, pred_data])
            tran_data_full = pd.concat([tran_data_full, tran_data])

        # Debug data
        out.data(pred_data_full, dbg_dir,
                 f'pred_data_full_{predictor}_{transferer}', dbg_data)
        out.data(pred_data_full, dbg_dir,
                 f'pred_data_full_{predictor}_{transferer}', dbg_data)

        # Plot
        _plot_models_validation(MName.SOURCE_PRED, predictor,
                                transferer, metric, pred_data_full)
        _plot_models_validation(MName.TRANSFERER, predictor,
                                transferer, metric, tran_data_full)


def _plot_models_validation(model_name: MName, predictor: str,
                            transferer: str, metric: str, data: pd.DataFrame):
    """
    Perform actual plotting of validation measures for a specified model and
    in a predictor-transferer combination

    :param model_name: predictor or transferer
    :param predictor: predictor model used (linear, tree, etc.)
    :param transferer: transferer model used (linear, tree, etc.)
    :param metric: studied performance metric
    :param data: experimental data fully prepared for plotting
    :return: None
    """

    # Vars:
    # * Unique system names
    # * Number of systems
    # * Figure, containing plots for each system
    sys_names = pd.unique(data.sys_name)
    n_sys = len(sys_names)
    fig = mpl.figure(figsize=(8.5, 2), dpi=600)

    # Create a separate figure for each system: ##############################
    for i_sys, sys_name in enumerate(sys_names):

        # Filter system's data
        sys_data = data[(data.sys_name == sys_name)]
        space_size = sys_data.space_size[0]

        # Plot data (line for each group)
        ax = fig.add_subplot(1, n_sys, i_sys + 1)
        ax.set_title(sys_name)
        ax.tick_params(axis='x', labelsize='8')
        ax.tick_params(axis='y', labelsize='8')

        # Generate x-ticks
        x_ticks = np.arange(0, space_size, space_size / 10)
        x_ticks[0] = 3
        x_ticks = np.append(x_ticks, [space_size])
        if sys_name == 'bzip2' or sys_name == 'gzip':
            x_ticks = np.delete(x_ticks, 2)
        ax.set_xticks(x_ticks)

        # Generate x-labels (Convert x-ticks to % of config space)
        x_label = np.round((x_ticks / space_size) * 100, 0).astype(int)
        ax.set_xticklabels(x_label, rotation='vertical')

        # Fix x-axis
        ax.set_xlim(3, space_size)
        print(x_ticks)
        print(x_label)

        # Fix y-axis
        ax.set_yticks(np.arange(0, 31, 5))
        ax.set_ylim(0, 30)

        mpl.grid(linestyle=':', linewidth=0.5)

        # Specify color cycle
        n = len(pd.unique(sys_data['server']))
        color_map = mpl.get_cmap('tab10')
        color_space = color_map(np.linspace(0, 1, n))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color_space)

        # Add one line for each server
        for srv, srv_df in sys_data.groupby('server'):
            x = list(srv_df['size'])
            y = list(srv_df['error'])
            ax.plot(x, y, linewidth=1)

    # Finalize predictors' figure ############################################
    # Add figure-wide labels
    fig.text(0.5, 0.0,
             f"Training size (% of a system's configuration space size)",
             ha='center', va='center')

    fig.text(0.0, 0.5,
             'MAPE (%)',
             ha='center', va='center', rotation='vertical')

    # Save ready figure
    mpl.tight_layout()
    fig.savefig(
        f'results/plot_models_validation_'
        f'{model_name}_{predictor}_{transferer}_{metric}',
        bbox_inches='tight', pad_inches=0)


# PLOT PARETO CLASSIFICATION MEASURES ########################################
def plot_pareto_measures(systems: list, server_source: str, server_dest: str):
    """
    For each studied system, plot classification quality measures of a Pareto
    frontier, transferred from source to destination hardware environments

    :param systems: studied configurable software systems
    :param server_source: source hardware environment
    :param server_dest: destination hardware environment
    :return: None
    """

    # VARIABLES ##############################################################
    # * dbg_msg: print debugging messages
    # * dbg_data: export debugging dataframes
    # * dbg_dir: directory to export debugging dataframes
    dbg_msg = True
    dbg_data = False
    dbg_dir = 'debug/plot_pareto_measures'

    measures = {
        'increasing': ['TPR', 'TNR', 'PPV', 'NPV', 'MCC'],
        'decreasing': ['FPR', 'FNR', 'FDR', 'FOR']}

    measures_colors = {
        'increasing': ['green', 'red', 'blue', 'orange', 'darkviolet'],
        'decreasing': ['black', 'darkred', 'red', 'orangered']}

    # Load & combine fronts data of all systems ##############################
    approx_fronts = []
    transf_fronts = []

    for si in systems:
        # Approximated frontiers
        front = SystemInfo.load_approx_frontier_results(si)
        front['sys_name'] = si.sys_name
        approx_fronts.append(front)

        # Transferred frontiers
        front = SystemInfo.load_transfer_frontier_results(si)
        front['sys_name'] = si.sys_name
        transf_fronts.append(front)

    # Finalize fronts
    approx_fronts = pd.concat(approx_fronts)
    transf_fronts = pd.concat(transf_fronts)

    # Generate delta_front dataframe
    delta_front = approx_fronts.copy()
    for measure in measures['increasing']:
        delta_front[measure] = delta_front[measure] - transf_fronts[measure]

    # Group fronts
    fronts = {
        'approximated': approx_fronts,
        'transferred': transf_fronts,
        'delta': delta_front}

    # PLOTTING DESIGN CREATION ###############################################
    # * Predictor name (linear, tree, etc.)
    # * Transferer name (linear, tree, etc.)
    # * Pareto Front type (approximated, transferred, etc.)
    # * Measures' type (increasing, decreasing, etc.)
    design_lists = \
        [[predictor, transferer, front_type, measures_type]
         for predictor in CONFIG_USR['predictors']
         for transferer in CONFIG_USR['transferers']
         for front_type in fronts
         for measures_type in measures]

    # Finalize design dataframe
    design_cols = ['predictor', 'transferer', 'front_type', 'measures_type']
    design_df = pd.DataFrame(design_lists, columns=design_cols)

    # Debug info
    out.msg(f'Plotting Design is created: {len(design_df)}', dbg_msg)
    out.data(design_df, dbg_dir, f'design_df', dbg_data)

    # Plot measures ##########################################################
    # Plot grouped measures for all fronts

    for row_id, row in design_df.iterrows():
        predictor = row['predictor']
        transferer = row['transferer']
        front_type = row['front_type']
        measures_type = row['measures_type']

        _plot_grouped_measures(
            systems, predictor, transferer,
            server_source, server_dest,
            front_type, fronts[front_type],
            measures_type, measures[measures_type],
            measures_colors[measures_type])

        _plot_mcc_measure(predictor, transferer,
                          server_source, server_dest,
                          front_type, fronts[front_type])


def _plot_grouped_measures(systems: list, predictor: str, transferer: str,
                           server_source: str, server_dest: str,
                           front_type: str, front: pd.DataFrame,
                           measures_type: str, measures: list,
                           measures_colors: list):
    """
    Perform actual plotting of classification quality measures for a specified
    source-destination combination and Pareto frontier type.

    :param systems: studied configurable software systems
    :param server_source: source (training) server
    :param server_dest: destination (target) server
    :param front_type: approximated, transferred, delta
    :param front: Pareto frontier's data
    :param measures_type: increasing/decreasing
    :param measures: measures' names
    :param measures_colors: measures' colors for plotting
    :return: None
    """

    # Vars:
    # * Number of systems
    # * Figure, containing a plot for each system
    # * MatPlotLib line-styles
    n_sys = len(systems)
    fig = mpl.figure(figsize=(8.5, 2), dpi=600)
    linestyles = ['-', '-', '-', '-', '-']

    # Create a separate figure for each system: ##############################
    for i_sys, si in enumerate(systems):

        # Filter system's data
        sys_data = front[(front.sys_name == si.sys_name) &
                         (front.transferer == transferer) &
                         (front.predictor == predictor) &
                         (front.source_id == server_source) &
                         (front.dest_id == server_dest)]

        # Prepare canvas
        ax = fig.add_subplot(1, n_sys, i_sys + 1)
        ax.set_title(si.sys_name)
        ax.tick_params(axis='x', labelsize='8')
        ax.tick_params(axis='y', labelsize='8')

        # Generate x-ticks
        x_ticks = np.arange(0, si.space_size, si.space_size / 10)
        x_ticks[0] = 3
        x_ticks = np.append(x_ticks, [si.space_size])
        if si.sys_name == 'bzip2' or si.sys_name == 'gzip':
            x_ticks = np.delete(x_ticks, 2)
        ax.set_xticks(x_ticks)

        # Generate x-labels (Convert x-ticks to % of config space)
        x_label = np.round((x_ticks / si.space_size) * 100, 0).astype(int)
        ax.set_xticklabels(x_label, rotation='vertical')

        # Fix y-axes
        if front_type == 'delta':
            ax.set_yticks(np.arange(-0.5, 0.51, 0.1))
            ax.set_xlim(3, si.space_size)
            ax.set_ylim(-0.5, 0.5)
        else:
            ax.set_yticks(np.arange(-0.1, 1.1, 0.1))
            ax.set_xlim(3, si.space_size)
            ax.set_ylim(0, 1)

        mpl.grid(linestyle=':', linewidth=0.5)

        # For each server add 4 lines
        for srv, srv_df in sys_data.groupby('source_id'):
            # For each measure add 1 line
            for i, measure in enumerate(measures):
                x = list(srv_df['train_size'])
                y = list(srv_df[measure])
                ax.plot(x, y, color=measures_colors[i],
                        linewidth=1, linestyle=linestyles[i])

    # Finalize predictors' figure ############################################
    # Add figure-wide labels
    fig.text(0.5, 0.0,
             f"Training size (% of a system's configuration space size)",
             ha='center', va='center')

    # Save ready figure
    mpl.tight_layout()
    fig.savefig(f'results/front_{predictor}_{transferer}_'
                f'{front_type}_{measures_type}',
                bbox_inches='tight', pad_inches=0)


def _plot_mcc_measure(predictor: str, transferer: str,
                      server_source: str, server_dest: str,
                      front_type: str, front: pd.DataFrame):
    """
    Perform actual plotting of Matthews Correlation Coefficient (MCC) for a
    specified source-destination combination and Pareto frontier type.

    :param predictor: studied metric's prediction model
    :param transferer: studied metric's transferring model
    :param server_source: source (training) server
    :param server_dest: destination (target) server
    :param front_type: approximated, transferred, delta
    :param front: Pareto frontier's data
    :return: None
    """

    # Vars:
    # * Unique system names
    # * Number of systems
    # * Figure, containing plots for each system
    sys_names = pd.unique(front.sys_name)
    n_sys = len(sys_names)
    fig = mpl.figure(figsize=(8.5, 2), dpi=600)

    # Create a separate figure for each system: ##############################
    for i_sys, sys_name in enumerate(sys_names):

        # Filter system's data
        sys_data = front[(front.sys_name == sys_name) &
                         (front.predictor == predictor) &
                         (front.transferer == transferer) &
                         (front.source_id == server_source) &
                         (front.dest_id == server_dest)]

        # Plot data (line for each group)
        ax = fig.add_subplot(1, n_sys, i_sys + 1)
        ax.set_title(sys_name)
        ax.tick_params(axis='x', labelsize='8')
        ax.tick_params(axis='y', labelsize='8')

        # Add one line for each server
        for srv, srv_df in sys_data.groupby('source_id'):
            x = list(srv_df['train_size'])
            y = list(srv_df['MCC'])
            ax.plot(x, y, linewidth=1)

    # Finalize predictors' figure ############################################
    # Add figure-wide labels
    fig.text(0.5, 0.0,
             f"Training size (% of a system's configuration space size)",
             ha='center', va='center')

    # Save ready figure
    mpl.tight_layout()
    fig.savefig(f'results/front_{predictor}_{transferer}_{front_type}_MCC',
                bbox_inches='tight', pad_inches=0)
