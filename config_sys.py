"""config_sys.py: Contains systems configuration information."""

from config_usr import CONFIG_USR

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


CONFIG_SYS = {
    # Configurable software systems that are benchmarked and supported
    'systems': ['bzip2', 'flac', 'gzip', 'x264', 'xz'],

    # Systems' configurable features
    'features': {
        'bzip2': ['fastbest', 'small'],
        'flac':  ['fastbest', 'verify', 'lax', 'replay_gain',
                  'lpc_optimization'],
        'gzip':  ['fastbest', 'rsyncable'],
        'x264':  ['b-adapt', 'me', 'no-mbtree', 'no-scenecut',
                  'rc-lookahead', 'ref', 'subme', 'trellis'],
        'xz':    ['fastbest', 'sparse', 'extreme', 'check']
    },

    # Systems' performance metrics
    'metrics': {
        'bzip2': ['time_compress', 'size_compress'],
        'flac':  ['time_compress', 'size_compress'],
        'gzip':  ['time_compress', 'size_compress'],
        'x264':  ['time_compress', 'size_compress'],
        'xz':    ['time_compress', 'size_compress'],
    },

    # Systems' targets
    'targets': {
        'bzip2': ['benchmark/targets/text/enwik9_10'],
        'flac':  ['benchmark/targets/audio/ghosts.wav'],
        'gzip':  ['benchmark/targets/text/enwik9_10'],
        'x264':  ['benchmark/targets/video/sintel_trailer_2k_480p24.y4m'],
        'xz':    ['benchmark/targets/text/enwik9_10'],
    },

    # Prediction models
    'predictors': ['tree'],

    # Transferring models
    'transferers': ['linear', 'tree'],

    # Systems' servers to keep
    # Notice: not every system is measured on every server
    'servers':
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

    'servers_ids':
        ['BscA0-4171HE', 'BscA0-2660', 'BscA0-2673v3',
         'BscA1-4171HE', 'BscA1-2660', 'BscA1-2673v3',
         'BscA2-4171HE', 'BscA2-2660', 'BscA2-2673v3',
         'StdA0-2673v3', 'StdA0-2660',
         'StdA1-2673v3', 'StdA1-2660',
         'StdA1v2-2660', 'StdA1v2-2673v3',
         'StdA2-2673v3', 'StdA2-2660',
         'StdA2v2-2660', 'StdA2v2-2673v3',
         'StdD1-2660',
         'StdD1v2-2673v3', 'StdD1v2-2673v4',
         'StdD2-2660',
         'StdD2v2-2673v3', 'StdD2v2-2673v4',
         'StdD2v3-2673v4', 'StdD2v3-2673v3',
         'StdE2v3-2673v4',
         'StdF1-2673v3', 'StdF1-2673v4',
         'StdF2-2673v3', 'StdF2-2673v4',
         'StdF2sv2-8168',
         'StdG1-2698Bv3'],
}


def validate_config_usr():
    """
    Checks whether a configuration specified by an end-user is valid

    :return: None
    """

    # Check that configuration members are non-empty
    tuples = {
        'systems': 'Studied systems are not specified',
        'predictors': 'Predictor models are not specified',
        'transferers': 'Transferring models are not specified',
        'repetitions': 'Amount of experiment repetitions is not set',
        'servers_source': 'Source servers are not specified',
        'servers_dest': 'Destination servers are not specified'
    }

    for key in tuples.keys():
        check_nonempty(key, tuples[key])

    # Check types
    tuples = {
        'systems': (list, '"Systems" should be a List of Strings'),
        'predictors': (list, '"Predictors" should be a List of Strings'),
        'transferers': (list, '"Transferers" should be a List of Strings'),
        'repetitions': (int, '"Repetitions" should be of Integer type'),
        'servers_source': (list, '"Servers_source" should be a List of Str'),
        'servers_dest': (list, '"Servers_dest" should be a List of Strings'),
    }

    for key in tuples.keys():
        check_type(key, tuples[key])

    # Check that configuration members have valid values
    tuples = {
        'systems': ('systems', 'Invalid system is specified'),
        'predictors': ('predictors', 'Invalid predictor is specified'),
        'transferers': ('transferers', 'Invalid transferer is specified'),
        'servers_source': ('servers', 'Invalid source-server is specified'),
        'servers_dest': ('servers', 'Invalid destination-server is specified'),
    }

    for key in tuples.keys():
        check_legal(key, tuples[key])


def check_nonempty(key, msg):
    """
    Checks whether all necessary user configuration options are specified

    :return: None
    """
    if key not in CONFIG_USR:
        raise ValueError(msg)


def check_type(key, tpl):
    """
    Checks whether each user configuration option has a proper type

    :return: None
    """

    # Extract configuration type and message from tuple
    config_type = tpl[0]
    config_msg = tpl[1]

    if not isinstance(CONFIG_USR[key], config_type):
        raise ValueError(config_msg)


def check_legal(key_usr, tpl):
    """
    Checks that configuration options have legal values
    E.g. specified servers have been benchmarked

    :return: None
    """
    key_sys = tpl[0]
    msg = tpl[1]

    for s in CONFIG_USR[key_usr]:
        if s not in CONFIG_SYS[key_sys]:
            raise ValueError(msg)
