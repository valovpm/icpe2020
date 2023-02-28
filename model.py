"""model.py: Class, describing models used for prediction and transferring."""

from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

import output as out
from system_info import SystemInfo

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


class MName(Enum):
    """
    Enum, containing different NAMES of regression models:

    * source_pred: model is a PREDICTOR of an analyzed system property
                   on a SOURCE hardware environment

    * dest_pred: model is a PREDICTOR of an analyzed system property
                 on a DESTINATION hardware environment

    * transferer: model is a TRANSFERER of an analyzed system property
                  from SOURCE to DESTINATION hardware environments

    * delta: a special keyword, meaning that a DELTA between PREDICTOR and
             TRANSFERER results for a particular system property is analyzed
    """

    SOURCE_PRED = 'source_pred'
    DEST_PRED = 'dest_pred'
    TRANSFERER = 'transferer'
    DELTA = 'delta'

    def __str__(self):
        return self.value


class MType(Enum):
    """
    Enum, containing valid model types:

    * tree: decision tree regressor model

    * linear: ordinary least squares linear regression model
    """
    TREE = 'tree'
    LINEAR = 'linear'

    def __str__(self):
        return self.value


class DFType(Enum):
    """
    Enum, containing valid dataframe types:

    * full_df: contains all studied system's data for a particular
               benchmarking target and hardware environment.

    * user_df: contains a sample of benchmarking data from full_df that
               might be available to an end user of the studied software
               system in a real-world setting.
               In our study, the size of user_df is determined by the
               experimental design.

    * rest_df: contains all benchmarking data from full_df that
               wasn't captured by user_df

    """
    FULL_DF = 'full_df'
    USER_DF = 'user_df'
    REST_DF = 'rest_df'

    def __str__(self):
        return self.value


class Suffix(Enum):
    """
    Suffixes, used to distinguish between property values and Pareto frontiers
    across different hardware environments
    """

    # Actual and predicted properties (or frontiers) on source environment
    SOURCE = '_s'
    SOURCE_PRED = '_sp'

    # Transferred actual and predicted properties (or frontiers)
    # from source to destination hardware environments
    SOURCE_TRANSFER = '_st'
    SOURCE_PRED_TRANSFER = '_spt'

    # Actual and predicted properties on destination machine
    DEST = '_d'
    DEST_PRED = '_dp'

    def __str__(self):
        return self.value


class ValidType(Enum):
    """
    Enum, containing validation methods used:

    * loocv: leave-one-out cross-validation

    * hv: holdout validation
    """

    LOOCV = 'loocv'
    HV = 'hv'

    def __str__(self):
        return self.value


class Model:
    """
    Class, describing models used for prediction and transferring.
    """

    def __init__(self, name: MName, mtype: MType, features: list,
                 metric: str, dfs: dict):
        """
        Initialize a model with necessary properties and dataframes

        :param name: model name (e.g. predictor, transferer, etc.)
        :param mtype: model type (e.g. linear, tree, etc.)
        :param features: configuration features
        :param metric: performance metric to be predicted
        :param dfs: dictionary of full, user and rest dataframes
        """
        self.name = name
        self.mtype = mtype
        self.features = features
        self.metric = metric
        self.dfs = dfs

        # Trained regressor & validation measures dictionary
        self.regressor = None
        self.measures = None

    @classmethod
    def get_untrained_models(cls, si: SystemInfo, row: pd.Series,
                             metric_col: str):
        """
        Prepare all necessary models for a specified experimental design row:

        * Metric predictor on source hardware environment
        * Metric predictor on destination hardware environment
        * Metric transferer from source to destination environments

        Note that these models are yet to be trained!

        :param si: system info
        :param row: design row
        :param metric_col: modelled performance metric
        :return: a dictionary of models prepared for training, along with
                 necessary training and validation data
        """
        # Vars:
        # dbg_data: export debugging dataframes
        dbg_data = False

        # Get model dataframes: full, training, testing
        # For: source and destination machines, and transfer training
        models_dfs = cls.__get_models_dfs(si, row)

        # Export dataframes to CSV
        if dbg_data:
            for key_model, df_dict in models_dfs.items():
                for key_df, df in df_dict.items():
                    out.data(df, '02_models_pipe',
                             f'iter:{row.iter_id}__{key_model}:{key_df}',
                             dbg_data)

        # Initialize models
        source_pred = Model(
            MName.SOURCE_PRED, MType(row.predictor), si.features,
            row[metric_col], models_dfs[MName.SOURCE_PRED])

        dest_pred = Model(
            MName.DEST_PRED, MType(row.predictor), si.features,
            row[metric_col], models_dfs[MName.DEST_PRED])

        transferer = Model(
            MName.TRANSFERER, MType(row.transferer), si.features,
            row[metric_col], models_dfs[MName.TRANSFERER])

        # Aggregate models
        models = {
            MName.SOURCE_PRED: source_pred,
            MName.DEST_PRED: dest_pred,
            MName.TRANSFERER: transferer
        }

        return models

    @staticmethod
    def __get_models_dfs(si: SystemInfo, row: pd.Series):
        """
        Generate all necessary dataframes for the experimental design row

        :param si: studied configurable software system
        :param row: experimental design instance (row)
        :return: a dictionary of prepared dataframes
        """

        # Vars:
        tdf = si.data

        # SOURCE #############################################################
        # Get all data for SOURCE machine
        source_df = (tdf[(tdf['target'] == row.target) &
                         (tdf['server'] == row.source_id)]
                     .reset_index(drop=True))

        # Get data available to user
        source_user_df = (source_df
                          .sample(row.train_size)
                          .sort_index())

        # Get all the remaining data
        source_rest_df = (source_df[~source_df.isin(source_user_df)]
                          .dropna(axis=0, how='all'))

        # DESTINATION ########################################################
        # Get all data for DESTINATION machine
        dest_df = (tdf[(tdf['target'] == row.target) &
                       (tdf['server'] == row.dest_id)]
                   .reset_index(drop=True))

        # Get data available to user
        drop_cols = [name + '_r' for name in source_user_df.columns.values]
        dest_user_df = (
            pd.merge(dest_df.reset_index(), source_user_df, how='right',
                     on=si.features, suffixes=('', '_r'))
            .drop(drop_cols, axis=1, errors='ignore')
            .set_index('index'))

        # Get all the remaining data
        dest_rest_df = (dest_df[~dest_df.isin(dest_user_df)]
                        .dropna(axis=0, how='all'))

        # TRANSFER ###########################################################
        # Get all data for transferring
        merge_cols = ['system', 'target'] + si.features
        suffixes = (Suffix.SOURCE, Suffix.DEST)
        transfer_df = (
            pd.merge(source_df, dest_df, how='left',
                     on=merge_cols, suffixes=suffixes)
            .reset_index(drop=True))

        # Prepare columns to be dropped after right-merge
        transfer_cols = list(transfer_df.columns.values)
        source_cols = list(source_user_df.columns.values)
        drop_cols = [x for x in source_cols if x not in transfer_cols]

        # Get data available to users
        transfer_user_df = (
            pd.merge(transfer_df.reset_index(), source_user_df, how='right',
                     on=merge_cols, suffixes=('', '_r'))
            .drop(drop_cols, axis=1, errors='ignore')
            .set_index('index')
            .sample(row.transfer_size)
            .sort_index())

        # Get all the remaining data
        transfer_rest_df = (transfer_df[~transfer_df.isin(transfer_user_df)]
                            .dropna(axis=0, how='all'))

        # AGGREGATE FRAMES ###################################################
        models_dfs = {
            MName.SOURCE_PRED: {
                DFType.FULL_DF: source_df,
                DFType.USER_DF: source_user_df,
                DFType.REST_DF: source_rest_df,
            },

            MName.DEST_PRED: {
                DFType.FULL_DF: dest_df,
                DFType.USER_DF: dest_user_df,
                DFType.REST_DF: dest_rest_df,
            },

            MName.TRANSFERER: {
                DFType.FULL_DF: transfer_df,
                DFType.USER_DF: transfer_user_df,
                DFType.REST_DF: transfer_rest_df,
            },
        }

        return models_dfs

    @classmethod
    def train_models(cls, models, df_type: 'DFType'):
        """
        Train prepared models using specified dataframe type

        :param models: a dictionary of prepared untrained models
        :param df_type: dataframe type to be used for training
        :return: None
        """

        for model in models.values():
            # Get training df
            train_df = model.dfs[df_type]

            # Train model
            cls.__train_regressor(model, train_df)

    @staticmethod
    def __train_regressor(model: 'Model', train_df: pd.DataFrame):
        """
        Train model on the specified training dataframe

        :param model: model prepared for training
        :return: None
        """

        regressor, x, y = None, None, None

        # Prepare regressor for fitting
        if model.mtype == MType.TREE:
            # Fit regression tree
            regressor = DecisionTreeRegressor(
                criterion='squared_error',  # mean squared error
                splitter='best',  # use the best split, not random
                max_depth=None,  # (None) grow to the max possible size
                min_samples_split=2,  # split a node if you can
                min_samples_leaf=1,  # require only 1 config per leaf
                min_weight_fraction_leaf=0,  # do not weight leafs
                max_features=None,  # consider all features during split
                random_state=0,  # fix random number generation
                max_leaf_nodes=None,  # number of leaf nodes is unlimited
                min_impurity_decrease=0,  # improvement required to split
                # presort=True  # presort training data to speed up
            )

        elif model.mtype == MType.LINEAR:
            # Fit linear regression
            regressor = LinearRegression(
                fit_intercept=True,  # calculate intercept
                # normalize=False,  # do not normalize data
                copy_X=True,  # do not overwrite data
                n_jobs=1  # do not parallelize computation
            )

        else:
            raise Exception('Wrong regression model type!')

        # Prepare data for fitting predictors
        if model.name == MName.SOURCE_PRED or model.name == MName.DEST_PRED:
            x = train_df[model.features]
            y = train_df[model.metric]

        # Prepare data for fitting transferers
        elif model.name == MName.TRANSFERER:
            x = train_df['{0}{1}'.format(model.metric, Suffix.SOURCE)]
            x = x.values.reshape(-1, 1)
            y = train_df['{0}{1}'.format(model.metric, Suffix.DEST)]

        regressor.fit(x, y)
        model.regressor = regressor

    @classmethod
    def validate_models(cls, models: dict, result_row: pd.Series):
        """
        Validate all trained models and add their validation measures
        to the resulting row

        :param models: prepared and trained models, ready for validation
        :param result_row: resulting row of validation measures
        :return: updated resulting row
        """

        for model in models.values():
            # Validate each model
            cls.__validate_model(model)

            # Add validation measures to the result
            result_row = pd.concat([result_row, model.measures])

        return result_row

    @classmethod
    def __validate_model(cls, model: 'Model'):
        """
        Perform a comprehensive validation of a trained model using different
        validation strategies

        :param model: prepared and trained model
        :return: None
        """

        # Dataframes to perform validation upon
        user_df = model.dfs[DFType.USER_DF]  # For 'Leave-one-out' validation
        rest_df = model.dfs[DFType.REST_DF]  # For 'Hold-out' validation
        full_df = model.dfs[DFType.FULL_DF]  # For 'Sanity-check' validation

        # LOO Validation #####################################################
        loo = LeaveOneOut()
        loo_dfs = []
        for train_index, test_index in loo.split(user_df):
            # Train model using user_df
            loo_train_df = user_df.iloc[train_index, :]
            cls.__train_regressor(model, loo_train_df)

            # Test model on 1 config from user_df (loo validation)
            loo_test_df = user_df.iloc[test_index, :]
            loo_dfs += [Model.__test_model(model, loo_test_df)]

        # Finalize LOO
        loo_df = pd.concat(loo_dfs)
        loo_measures = {}
        for m in loo_df:
            loo_measures[f'{model.metric}|{m}_loo_mean'] = \
                np.round(loo_df[m].mean(), 2)
        loo_measures = pd.Series(loo_measures)

        # Hold-out Validation ################################################
        cls.__train_regressor(model, user_df)
        hov_df = Model.__test_model(model, rest_df)
        hov_measures = {}
        for m in hov_df:
            hov_measures[f'{model.metric}|{m}_hov_mean'] = \
                np.round(hov_df[m].mean(), 2)
        hov_measures = pd.Series(hov_measures)

        # Sanity validation ##################################################
        cls.__train_regressor(model, user_df)
        san_df = Model.__test_model(model, full_df)
        # san_df = pd.concat(san_dfs)
        san_measures = {}
        for m in san_df:
            san_measures[f'{model.metric}|{m}_san_mean'] = \
                np.round(san_df[m].mean(), 2)
        san_measures = pd.Series(san_measures)

        model.measures = pd.concat([loo_measures, hov_measures, san_measures])

    @classmethod
    def __test_model(cls, model: 'Model', valid_test_df: pd.DataFrame):
        """
        Test trained model on a particular validation dataframe and
        acquire validation measures values

        :param model: trained regression model
        :return: a dataframe of validation measures
        """

        # Predictors:
        if model.name == MName.SOURCE_PRED or model.name == MName.DEST_PRED:
            x_true = valid_test_df[model.features]
            y_true = valid_test_df[model.metric]
            y_pred = np.NaN
            if len(x_true.index) > 0:
                y_pred = model.regressor.predict(x_true)

        # Transferers:
        elif model.name == MName.TRANSFERER:
            metric_col = '{0}{1}'.format(model.metric, Suffix.SOURCE)
            x_true = valid_test_df[metric_col]
            x_true = x_true.values.reshape(-1, 1)

            metric_col = '{0}{1}'.format(model.metric, Suffix.DEST)
            y_true = valid_test_df[metric_col].to_numpy()

            y_pred = np.NaN
            if x_true.size > 0:
                y_pred = model.regressor.predict(x_true)

        # Other
        else:
            raise Exception('Invalid model name!')

        # Finalize measures
        mape = cls.__mean_absolute_percentage_error(y_true, y_pred)
        mape = [mape] if not isinstance(mape, (list,)) else None
        measures = {f'{model.name}|mape': mape}

        # We need df (and not dictionary) because of pd.concat later
        measures_df = pd.DataFrame(data=measures)
        return measures_df

    @staticmethod
    def __mean_absolute_percentage_error(y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error (MAPE)

        :param y_true: a list of actual metric values
        :param y_pred: a list of predicted metric values
        :return: a value of MAPE error
        """
        error = np.abs((y_true - y_pred) / y_true)

        if not error.size == 0:
            error = np.round(np.mean(error) * 100, 2)
        else:
            error = 0

        return error
