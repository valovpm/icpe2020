"""output.py: functions to output dataframes to CSV and messages to console"""

from datetime import datetime
import os

import pandas as pd

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


def data(df: pd.DataFrame, dir_name: str, file_name: str, export_on: bool):
    """
    Exports a provided dataframe into a CSV-file to a specified location

    :param df: dataframe for export
    :param dir_name: directory name
    :param file_name: csv-file name
    :param export_on: perform debugging or not
    :return: dataframe for export
    """

    if export_on:

        # Create dir for debugging output
        dir_name = f'{dir_name}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Export debugging df
        path = f'{dir_name}/{file_name}.csv'
        df.to_csv(path)

    return df


def msg(message: str, print_on: bool):
    """
    Prints a specified message to the console

    :param message: a message to print
    :param print_on: perform printing or not
    :return: None
    """

    if print_on:
        dt_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'TIME: {dt_str}; MSG: {message}')
