#!/usr/bin/env python

"""main.py: Contains the main function that starts the analysis execution."""

import config_sys as cfg
import experiments as ex
import visualize as vs

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


def main():
    """
    The main function that starts the overall process of exploring,
    analyzing, and visualizing of Pareto Frontier transferring

    :return: None
    """

    cfg.validate_config_usr()
    ex.start_experiments()
    vs.visualize()


main()
