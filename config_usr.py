"""config_usr.py: Configuration available for end-user of this project."""

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


# Configuration profile for ICPE'2020 paper
PROFILE_PAPER = {
    # System
    'systems': ['bzip2', 'flac', 'gzip', 'x264', 'xz'],

    # Prediction models
    'predictors': ['tree'],

    # Transferring models
    'transferers': ['linear', 'tree'],

    # Number of repetitions for each experiment
    'repetitions': 10,

    # Source servers, used to train predictors
    'servers_source':
        ['BasicA0-japaneast', 'BasicA0-westus2',
         'BasicA1-brazilsouth', 'BasicA1-japaneast', 'BasicA1-westus2',
         'BasicA2-brazilsouth',
         'StandardD2-australiasoutheast',
         'StandardD2v3-australiasoutheast', 'StandardD2v3-southeastasia',
         'StandardE2v3-westeurope',
         'StandardF2sv2-westus2',
         'StandardG1-eastus2'],

    # Destination servers, used to train transferers
    'servers_dest':
        ['StandardF2sv2-westus2'],
}


# Minimalistic configuration profile
# Analyzes minimal amount of software & hardware,
# necessary to comprehensively test the analysis scripts
PROFILE_MIN = {
    # Systems
    'systems': ['bzip2', 'gzip'],

    # Prediction models
    'predictors': ['tree'],

    # Transferring models
    'transferers': ['linear', 'tree'],

    # Number of repetitions for each experiment
    'repetitions': 10,

    # Source servers, used to train predictors
    'servers_source':
        ['BasicA0-japaneast',
         'BasicA1-brazilsouth'],

    # Destination servers, used to train transferers
    'servers_dest':
        ['StandardF2sv2-westus2',
         'StandardG1-eastus2'],
}


# Full configuration profile
# Analyzes all possible combinations of software systems, hardware servers,
# predictors, transferers,
PROFILE_FULL = {
    # Systems
    'systems': ['bzip2', 'flac', 'gzip', 'x264', 'xz'],

    # Prediction models
    'predictors': ['tree'],

    # Transferring models
    'transferers': ['linear', 'tree'],

    # Number of repetitions for each experiment
    'repetitions': 10,

    # Source servers, used to train predictors
    'servers_source':
        ['BasicA0-brazilsouth', 'BasicA0-japaneast', 'BasicA0-westus2',
         'BasicA1-brazilsouth', 'BasicA1-japaneast', 'BasicA1-westus2',
         'BasicA2-brazilsouth', 'BasicA2-japaneast', 'BasicA2-westus2',
         'StandardA0-canadacentral', 'StandardA0-japaneast',
         'StandardA1-canadacentral', 'StandardA1-japaneast',
         'StandardA1v2-southcentralus', 'StandardA1v2-westcentralus',
         'StandardA2-canadacentral', 'StandardA2-japaneast',
         'StandardA2v2-southcentralus', 'StandardA2v2-westcentralus',
         'StandardD1-australiasoutheast',
         'StandardD1v2-centralus', 'StandardD1v2-southindia',
         'StandardD2-australiasoutheast',
         'StandardD2v2-centralus', 'StandardD2v2-southindia',
         'StandardD2v3-australiasoutheast', 'StandardD2v3-southeastasia',
         'StandardE2v3-westeurope',
         'StandardF1-eastus', 'StandardF1-southindia',
         'StandardF2-eastus', 'StandardF2-southindia',
         'StandardF2sv2-westus2',
         'StandardG1-eastus2'],


    # Destination servers, used to train transferers
    'servers_dest':
        ['BasicA0-brazilsouth', 'BasicA0-japaneast', 'BasicA0-westus2',
         'BasicA1-brazilsouth', 'BasicA1-japaneast', 'BasicA1-westus2',
         'BasicA2-brazilsouth', 'BasicA2-japaneast', 'BasicA2-westus2',
         'StandardA0-canadacentral', 'StandardA0-japaneast',
         'StandardA1-canadacentral', 'StandardA1-japaneast',
         'StandardA1v2-southcentralus', 'StandardA1v2-westcentralus',
         'StandardA2-canadacentral', 'StandardA2-japaneast',
         'StandardA2v2-southcentralus', 'StandardA2v2-westcentralus',
         'StandardD1-australiasoutheast',
         'StandardD1v2-centralus', 'StandardD1v2-southindia',
         'StandardD2-australiasoutheast',
         'StandardD2v2-centralus', 'StandardD2v2-southindia',
         'StandardD2v3-australiasoutheast', 'StandardD2v3-southeastasia',
         'StandardE2v3-westeurope',
         'StandardF1-eastus', 'StandardF1-southindia',
         'StandardF2-eastus', 'StandardF2-southindia',
         'StandardF2sv2-westus2',
         'StandardG1-eastus2'],
}


CONFIG_USR = PROFILE_PAPER
