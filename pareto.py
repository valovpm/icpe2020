"""pareto.py: functionality for building and analyzing Pareto frontiers"""

from enum import Enum
import math

import numpy as np
import pandas as pd

from model import MName, Model, Suffix
from system_info import SystemInfo

__author__ = "Pavel Valov"
__copyright__ = "Copyright 2020, Transferring Pareto Frontiers across " \
                "Heterogeneous Hardware Environments"
__license__ = "MIT"
__maintainer__ = "Pavel Valov"
__email__ = "valov.pm@gmail.com"
__status__ = "Production"


class PFType(Enum):
    """
    Enum, containing different types of Pareto Frontiers:

    * approx_pf: frontier approximated using a separate predictor model for
                 each studied system performance metric

    * transfer_pf: frontier transferred using a separate transferring model
                   for each studied system performance metric

    """

    APPROX_PF = 'approx_pf'
    TRANSFER_PF = 'transfer_pf'

    def __str__(self):
        return self.value


class Pareto:
    """
    Class, containing methods for building and analyzing Pareto Frontiers
    """

    @staticmethod
    def get_template_df(si: SystemInfo, row: pd.Series, objectives: dict):
        """
        Generates a 'template' dataframe for the experiment on Pareto frontier
        transferring. This template is later expanded with experimental
        results.

        :param si: studied configurable software system
        :param row: experimental design row
        :param objectives: optimization objectives for Pareto Frontier
        :return: template dataframe
        """

        # VARS:
        # * System data frame
        # * Filtering columns
        df = si.data
        filter_cols = ['system', 'target', 'metric_a', 'metric_b'] + \
            si.features + ['server', 'ma', 'mb']

        # Source machine data
        source_df = (df[(df['target'] == row.target) &
                        (df['server'] == row.source_id)]
                     .reset_index(drop=True))
        source_df['metric_a'] = objectives['metric_a']
        source_df['metric_b'] = objectives['metric_b']
        source_df.rename({objectives['metric_a']: 'ma',
                          objectives['metric_b']: 'mb'},
                         axis='columns', inplace=True)
        source_df = source_df[filter_cols]

        # Destination machine data
        dest_df = (df[(df['target'] == row.target) &
                      (df['server'] == row.dest_id)]
                   .reset_index(drop=True))
        dest_df['metric_a'] = objectives['metric_a']
        dest_df['metric_b'] = objectives['metric_b']
        dest_df.rename({objectives['metric_a']: 'ma',
                        objectives['metric_b']: 'mb'},
                       axis='columns', inplace=True)
        dest_df = dest_df[filter_cols]

        # Transfer data
        merge_cols = ['system', 'target', 'metric_a', 'metric_b'] \
            + si.features
        suffixes = (Suffix.SOURCE, Suffix.DEST)
        transfer_df = (pd.merge(source_df, dest_df, how='left',
                                on=merge_cols,
                                suffixes=suffixes
                                ).reset_index(drop=True))

        return transfer_df

    @staticmethod
    def __predict_metric(temp_df: pd.DataFrame, model: Model, new_col: str):
        """
        Extends template dataframe with a new column, containing values of
        an approximated performance metric

        :param temp_df: template dataframe for Pareto transferring experiment
        :param model: predictor used for metric approximation
        :param new_col: name of a new column, containing approximated metric
        :return: None
        """
        temp_df[new_col] = \
            model.regressor.predict(temp_df[model.features])

    @staticmethod
    def __transfer_metric(temp_df: pd.DataFrame, model: Model,
                          from_col: str, new_col: str):
        """
        Extends template dataframe with a new column, containing values of
        a transferred performance metric

        :param temp_df: template dataframe for Pareto transferring experiment
        :param model: transferer model used for metric approximation across
                      different hardware environments (servers)
        :param from_col: column containing metric values that need to be
                         transferred
        :param new_col: name of a new column, containing transferred metric
        :return: None
        """
        x_true = temp_df[from_col].values.reshape(-1, 1)
        temp_df[new_col] = model.regressor.predict(x_true)

    @classmethod
    def prepare_metrics(cls, temp_df: pd.DataFrame,
                        models_a: dict, models_b: dict):
        """
        Prepares template dataframe for Pareto Frontier construction and
        analysis by approximating and transferring all necessary performance
        metrics.

        :param temp_df: template dataframe for Pareto transferring experiment
        :param models_a: predictors and transferrers for performance metric A
        :param models_b: predictors and transferrers for performance metric B
        :return: None
        """

        # Predict all properties on source and destination ###################
        cls.__predict_metric(temp_df, models_a[MName.SOURCE_PRED],
                             new_col=f'ma{Suffix.SOURCE_PRED}')

        cls.__predict_metric(temp_df, models_a[MName.DEST_PRED],
                             new_col=f'ma{Suffix.DEST_PRED}')

        cls.__predict_metric(temp_df, models_b[MName.SOURCE_PRED],
                             new_col=f'mb{Suffix.SOURCE_PRED}')

        cls.__predict_metric(temp_df, models_b[MName.DEST_PRED],
                             new_col=f'mb{Suffix.DEST_PRED}')

        # Transfer all properties from source to destination #################
        cls.__transfer_metric(temp_df, models_a[MName.TRANSFERER],
                              from_col=f'ma{Suffix.SOURCE}',
                              new_col=f'ma{Suffix.SOURCE_TRANSFER}')

        cls.__transfer_metric(temp_df, models_a[MName.TRANSFERER],
                              from_col=f'ma{Suffix.SOURCE_PRED}',
                              new_col=f'ma{Suffix.SOURCE_PRED_TRANSFER}')

        cls.__transfer_metric(temp_df, models_b[MName.TRANSFERER],
                              from_col=f'mb{Suffix.SOURCE}',
                              new_col=f'mb{Suffix.SOURCE_TRANSFER}')

        cls.__transfer_metric(temp_df, models_b[MName.TRANSFERER],
                              from_col=f'mb{Suffix.SOURCE_PRED}',
                              new_col=f'mb{Suffix.SOURCE_PRED_TRANSFER}')

    @staticmethod
    def __build_front(temp_df: pd.DataFrame, objectives: list, col: str):
        """
        Analyzes system configurations based on specified optimization
        objectives and selects Pareto-optimal ones, thus forming a new
        Pareto Frontier.

        :param temp_df: template dataframe that is expanded with a new column,
                        specifying Pareto Frontier information
        :param objectives: column names of metrics to be used as optimization
                           objectives when building a Pareto Frontier
        :param col: name of a column that is used to store information about
                    a constructed Pareto Frontier
        :return: None
        """

        # Generate matrices for each metric ##################################
        # Metric A's matrices
        # * Extract Metric-A values for each configuration
        # * Make Metric-A row-matrix (each row contains all A's values)
        # * Make Metric-A column-matrix (each column contains all A's values)
        ma = temp_df[objectives[0]].values
        ma_rows = np.repeat(ma[np.newaxis, :], ma.size, axis=0)
        ma_cols = np.repeat(ma[:, np.newaxis], ma.size, axis=1)

        # Metric B's matrices
        # * Extract Metric-B values for each configuration
        # * Make Metric-B row-matrix (each row contains all B's values)
        # * Make Metric-B column-matrix (each column contains all B's values)
        mb = temp_df[objectives[1]].values
        mb_rows = np.repeat(mb[np.newaxis, :], mb.size, axis=0)
        mb_cols = np.repeat(mb[:, np.newaxis], mb.size, axis=1)

        # Generate Pareto frontier ###########################################
        # How many metrics each config optimizes, compared to other configs?
        optimized_metrics = (np.less(ma_rows, ma_cols) +
                             np.less(mb_rows, mb_cols))

        # For each config, which other configs did it fail to optimize?
        optimize_failed = np.equal(optimized_metrics, 0)

        # How many metrics each config degraded, compared to other configs?
        degraded_metrics = (np.greater(ma_rows, ma_cols) +
                            np.greater(mb_rows, mb_cols))

        # For each config, which other configs did it degrade?
        degrade_success = np.greater(degraded_metrics, 0)

        # For each config, which other configs did it fail to optimize and
        # on the contrary degraded their metric(s)?
        # For each config, which other configs are DOMINATING this config?
        dominating = np.logical_and(optimize_failed, degrade_success)

        # Get Pareto Frontier
        dominating_reduced = np.logical_or.reduce(dominating, 0)
        temp_df[col] = np.logical_not(dominating_reduced)

    @classmethod
    def build_fronts(cls, temp_df: pd.DataFrame):
        """
        Constructs all possible approximated and transferred Pareto Frontiers

        :param temp_df: template dataframe that is expanded with new columns,
                        containing Pareto Frontier information
        :return: None
        """

        for s in Suffix:
            objectives = [f'ma{s}', f'mb{s}']
            cls.__build_front(temp_df, objectives, f'pf{s}')

    @staticmethod
    def polish_temp_df(temp_df: pd.DataFrame, si: SystemInfo):
        """
        Polish the template dataframe by reindexing and keeping only
        necessary columns

        :param temp_df: template dataframe
        :param si: studied configurable software system
        :return: polished dataframe
        """

        # Order cols #########################################################
        cols = ['system', 'target', 'server_s', 'server_d'] + \
               ['metric_a', 'metric_b'] + si.features + \
               ['ma_s', 'ma_st', 'ma_sp', 'ma_spt', 'ma_d', 'ma_dp'] + \
               ['mb_s', 'mb_st', 'mb_sp', 'mb_spt', 'mb_d', 'mb_dp'] + \
               ['pf_s', 'pf_st', 'pf_sp', 'pf_spt', 'pf_d', 'pf_dp']

        return temp_df.reindex(columns=cols, copy=False)

    @staticmethod
    def get_confusion_row(temp_df: pd.DataFrame, row: pd.Series,
                          pareto_type: 'PFType'):
        """
        Analyze a specified Pareto Frontier and calculate classification
        measures (confusion matrix) for it

        :param temp_df: template dataframe, containing Pareto Frontiers
        :param row: experimental design row
        :param pareto_type: analyze approximated or transferred frontier
        :return: experimental design row, expanded with classification
                 measures
        """

        # Classification quality measures ####################################
        # Basic measures:
        # * Population (POP): total num of configs
        # * Condition Positive (P): num of pareto-optimal configs
        # * Condition Negative (N): num of non-optimal configs
        # * Predicted Condition Positive (PP): num of configs,
        #       classified as pareto-optimal
        # * Predicted Condition Negative (PN): num of configs,
        #       classified as non-optimal
        measures = ['POP', 'P', 'N', 'PP', 'PN']

        # * True Positive (TP): num of configs, classified
        #       correctly as pareto-optimal (hit)
        # * False Positive (FP): num of configs, classified
        #       incorrectly as pareto-optimal (false alarm, Type I error)
        # * False Negative (FN): num of configs, classified
        #       incorrectly as non-optimal configs (miss, Type II error)
        # * True negative (TN): num of configs, classified
        #       correctly as non-optimal configs (correct rejection)
        measures += ['TP', 'FP', 'FN', 'TN']

        # * True Positive Rate (TPR): proportion of configs,
        #       correctly classified as pareto-optimal (sensitivity, recall,
        #       probability of detection)
        # * False Positive Rate (FPR): proportion of configs,
        #       incorrectly classified as pareto-optimal (fall-out,
        #       probability of false alarm)
        # * False Negative Rate (FNR): proportion of configs
        #       incorrectly classified as non-optimal (miss rate)
        # * True Negative Rate (TNR): proportion of configs,
        #       correctly classified as non-optimal (specificity)
        measures += ['TPR', 'FPR', 'FNR', 'TNR']

        # * Positive Predictive Value (PPV): proportion of optimally
        #       classified configs that are actually optimal (precision)
        # * False Discovery Rate (FDR): proportion of optimally
        #       classified configs, that are actually non-optimal
        # * Negative Predictive Value (NPV): proportion of non-optimally
        #       classified configs that are actually non-optimal
        # * False Omission Rate (FOR): proportion of non-optimally
        #       classified configs, that are actually optimal
        measures += ['PPV', 'FDR', 'FOR', 'NPV']

        # * F1 score (F1)
        # * Matthews correlation coefficient (MCC)
        measures += ['F1', 'MCC']
        measures_vals = []

        # Start populating confusion results
        confusion_dict = {}
        for st in row.iteritems():
            confusion_dict[st[0]] = st[1]

        tdf = temp_df

        # Confusion Matrix ###################################################
        #
        # Layout:
        #   | POP | P   | N   |
        #   | PP  | TP  | FP  | PPV | FDR |
        #   | PN  | FN  | TN  | FOR | NPV |
        #         | TPR | FPR | F1        |
        #         | FNR | TNR | MCC       |

        # Columns
        actual, target = '', ''
        if pareto_type == PFType.APPROX_PF:
            actual = 'pf_s'
            target = 'pf_sp'
        elif pareto_type == PFType.TRANSFER_PF:
            actual = 'pf_d'
            target = 'pf_spt'

        # Basic measures:
        # * Population (POP)
        # * Condition Positive (P)
        # * Condition Negative (N)
        # * Predicted Condition Positive (PP)
        # * Predicted Condition Negative (PN)
        pop = tdf.shape[0]
        p = tdf[tdf[actual]].shape[0]
        n = tdf[~tdf[actual]].shape[0]
        pp = tdf[tdf[target]].shape[0]
        pn = tdf[~tdf[target]].shape[0]
        measures_vals += [pop, p, n, pp, pn]

        # Classification Results:
        # * True Positive (TP)
        # * False Positive (FP)
        # * False Negative (FN)
        # * True Negative (TN)
        tp = tdf[(tdf[actual]) &
                 (tdf[actual] == tdf[target])].shape[0]

        fp = tdf[(~tdf[actual]) &
                 (tdf[actual] != tdf[target])].shape[0]

        fn = tdf[(tdf[actual]) &
                 (tdf[actual] != tdf[target])].shape[0]

        tn = tdf[(~tdf[actual]) &
                 (tdf[actual] == tdf[target])].shape[0]

        measures_vals += [tp, fp, fn, tn]

        # Classification Results Rates:
        # * True Positive Rate (TPR)
        # * False Positive Rate (FPR)
        # * True Negative Rate (TNR)
        # * False Negative Rate (FNR)
        tpr = tp / p if p > 0 else float('nan')
        fpr = fp / n if n > 0 else float('nan')
        fnr = fn / p if p > 0 else float('nan')
        tnr = tn / n if n > 0 else float('nan')
        measures_vals += [tpr, fpr, fnr, tnr]

        # Predicted Results Rates
        # * Positive Predictive Value (PPV)
        # * False Discovery Rate (FDR)
        # * Negative Predictive Value (NPV)
        # * False Omission Rate (FOR)
        ppv = tp / (tp + fp) \
            if (tp + fp) > 0 else float('nan')

        fdr = fp / (fp + tp) \
            if (fp + tp) > 0 else float('nan')

        npv = tn / (tn + fn) \
            if (tn + fn) > 0 else float('nan')

        fo_rate = fn / (fn + tn) \
            if (fn + tn) > 0 else float('nan')

        measures_vals += [ppv, fdr, fo_rate, npv]

        # Additional Measures:
        # * F1 score - harmonic mean of sensitivity and precision
        # * Matthews Correlation Coefficient (MCC)
        f1 = (2 * ppv * tpr) / (ppv + tpr) \
            if (ppv + tpr) > 0 else float('nan')

        mcc_d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_d \
            if mcc_d > 0 else float('nan')

        measures_vals += [f1, mcc]

        # Populate confusion row #############################################
        for i, m in enumerate(measures):
            m_val = np.round(measures_vals[i], 2)
            confusion_dict[m] = m_val

        confusion_row = pd.Series(confusion_dict)
        return confusion_row

    @staticmethod
    def polish_confusion_df(confusion_df: pd.DataFrame):
        """
        Polish resulting dataframe of the Pareto Frontier transferring
        experiment, by keeping and ordering only necessary columns

        :param confusion_df: dataframe, containing confusion matrices for
                             each instance of the experimental design
        :return: polished resulting dataframe
        """

        # Confusion Matrix Layout:
        #   | POP | P   | N   |
        #   | PP  | TP  | FP  | PPV | FDR |
        #   | PN  | FN  | TN  | FOR | NPV |
        #         | TPR | FPR | F1        |
        #         | FNR | TNR | MCC       |

        cols = ['target', 'metric_a', 'metric_b',
                'source_id', 'dest_id', 'predictor', 'transferer',
                'train_size', 'transfer_size']

        cols += ['POP', 'P', 'N', 'PP', 'PN'] + \
                ['TP', 'FP', 'FN', 'TN'] + \
                ['TPR', 'FPR', 'FNR', 'TNR'] + \
                ['PPV', 'FDR', 'FOR', 'NPV'] + \
                ['F1', 'MCC']

        return confusion_df[cols]
