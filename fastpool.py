import sys
from copy import deepcopy
from six import iteritems, string_types, integer_types
import os

if sys.version_info >= (3, 3):
    from collections.abc import Iterable, Sequence, Mapping, MutableMapping
else:
    from collections import Iterable, Sequence, Mapping, MutableMapping

from collections import OrderedDict, defaultdict

import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
from enum import Enum
from operator import itemgetter
from threading import Lock

if platform.system() == 'Linux':
    try:
        ctypes.CDLL('librt.so')
    except Exception:
        pass

try:
    from pandas import DataFrame, Series
except ImportError:
    class DataFrame(object):
        pass

    class Series(object):
        pass

import scipy.sparse


_typeof = type

import catboost
from catboost.plot_helpers import save_plot_file, try_plot_offline
from catboost import _catboost
from catboost.metrics import BuiltinMetric
from catboost.core import _get_features_indices

_PoolBase = _catboost._PoolBase
_CatBoost = _catboost._CatBoost
_MetricCalcerBase = _catboost._MetricCalcerBase
_cv = _catboost._cv
_set_logger = _catboost._set_logger
_reset_logger = _catboost._reset_logger
_configure_malloc = _catboost._configure_malloc
CatBoostError = _catboost.CatBoostError
_metric_description_or_str_to_str = _catboost._metric_description_or_str_to_str
is_classification_objective = _catboost.is_classification_objective
is_cv_stratified_objective = _catboost.is_cv_stratified_objective
is_regression_objective = _catboost.is_regression_objective
is_multiregression_objective = _catboost.is_multiregression_objective
is_survivalregression_objective = _catboost.is_survivalregression_objective
is_groupwise_metric = _catboost.is_groupwise_metric
is_ranking_metric = _catboost.is_ranking_metric
_PreprocessParams = _catboost._PreprocessParams
_check_train_params = _catboost._check_train_params
_MetadataHashProxy = _catboost._MetadataHashProxy
_NumpyAwareEncoder = _catboost._NumpyAwareEncoder
FeaturesData = _catboost.FeaturesData
_have_equal_features = _catboost._have_equal_features
SPARSE_MATRIX_TYPES = _catboost.SPARSE_MATRIX_TYPES
MultiRegressionCustomMetric = _catboost.MultiRegressionCustomMetric
MultiRegressionCustomObjective = _catboost.MultiRegressionCustomObjective
fspath = _catboost.fspath
_eval_metric_util = _catboost._eval_metric_util


from contextlib import contextmanager  # noqa E402


_configure_malloc()
_catboost._library_init()

INTEGER_TYPES = (integer_types, np.integer)
FLOAT_TYPES = (float, np.floating)
STRING_TYPES = (string_types,)
ARRAY_TYPES = (list, np.ndarray, DataFrame, Series)

if sys.version_info >= (3, 6):
    PATH_TYPES = STRING_TYPES + (os.PathLike,)
elif sys.version_info >= (3, 4):
    from pathlib import Path
    PATH_TYPES = STRING_TYPES + (Path,)
else:
    PATH_TYPES = STRING_TYPES



class Pool(catboost.Pool):
    """
    Pool used in CatBoost as a data structure to train model from.
    """

    def __init__(
        self,
        data,
        label=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        column_description=None,
        pairs=None,
        delimiter='\t',
        has_header=False,
        ignore_csv_quoting=False,
        weight=None,
        group_id=None,
        group_weight=None,
        subgroup_id=None,
        pairs_weight=None,
        baseline=None,
        feature_names=None,
        thread_count=-1,
        log_cout=sys.stdout,
        log_cerr=sys.stderr
    ):
        """
        Pool is an internal data structure that is used by CatBoost.
        You can construct Pool from list, numpy.ndarray, pandas.DataFrame, pandas.Series.

        Parameters
        ----------
        data : list or numpy.ndarray or pandas.DataFrame or pandas.Series or FeaturesData or string or pathlib.Path
            Data source of Pool.
            If list or numpy.ndarrays or pandas.DataFrame or pandas.Series, giving 2 dimensional array like data.
            If FeaturesData - see FeaturesData description for details, 'cat_features' and 'feature_names'
              parameters must be equal to None in this case
            If string or pathlib.Path, giving the path to the file with data in catboost format.
              If string starts with "quantized://", the file has to contain quantized dataset saved with Pool.save().

        label : list or numpy.ndarrays or pandas.DataFrame or pandas.Series, optional (default=None)
            Label of the training data.
            If not None, giving 1 or 2 dimensional array like data with floats.
            If data is a file, then label must be in the file, that is label must be equals to None

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        column_description : string or pathlib.Path, optional (default=None)
            ColumnsDescription parameter.
            There are several columns description types: Label, Categ, Num, Auxiliary, DocId, Weight, Baseline, GroupId, Timestamp.
            All columns are Num as default, it's not necessary to specify
            this type of columns. Default Label column index is 0 (zero).
            If None, Label column is 0 (zero) as default, all data columns are Num as default.
            If string or pathlib.Path, giving the path to the file with ColumnsDescription in column_description format.

        pairs : list or numpy.ndarray or pandas.DataFrame or string or pathlib.Path
            The pairs description.
            If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of winner object in the training set. The second element of the pair is
            the index of loser object in the training set.
            If string or pathlib.Path, giving the path to the file with pairs description.

        delimiter : string, optional (default='\t')
            Delimiter to use for separate features in file.
            Should be only one symbol, otherwise would be taken only the first character of the string.

        has_header : bool optional (default=False)
            If True, read column names from first line.

        ignore_csv_quoting : bool optional (default=False)
            If True ignore quoting '"'.

        weight : list or numpy.ndarray, optional (default=None)
            Weight for each instance.
            If not None, giving 1 dimensional array like data.

        group_id : list or numpy.ndarray, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.

        group_weight : list or numpy.ndarray, optional (default=None)
            Group weight for each instance.
            If not None, giving 1 dimensional array like data.

        subgroup_id : list or numpy.ndarray, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.

        pairs_weight : list or numpy.ndarray, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.ndarray, optional (default=None)
            Baseline for each instance.
            If not None, giving 2 dimensional array like data.

        feature_names : list or string or pathlib.Path, optional (default=None)
            If list - list of names for each given data_feature.
            If string or pathlib.Path - path with scheme for feature names data to load.
            If this parameter is None and 'data' is pandas.DataFrame feature names will be initialized
              from DataFrame's column names.
            Must be None if 'data' parameter has FeaturesData type

        thread_count : int, optional (default=-1)
            Thread count for data processing.
            If -1, then the number of threads is set to the number of CPU cores.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        """
        if data is not None:
            cat_features = _get_features_indices(cat_features, feature_names)
            self._init_pool(data, label, cat_features, text_features, embedding_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names, thread_count)
        super(catboost.Pool, self).__init__()

    # def _check_files(self, data, column_description, pairs):
    #     """
    #     Check files existence.
    #     """
    #     data = fspath(data)
    #     column_description = fspath(column_description)
    #     pairs = fspath(pairs)
    #     if data.find('://') == -1 and not os.path.isfile(data):
    #         raise CatBoostError("Invalid data path='{}': file does not exist.".format(data))
    #     if column_description is not None and column_description.find('://') == -1 and not os.path.isfile(column_description):
    #         raise CatBoostError("Invalid column_description path='{}': file does not exist.".format(column_description))
    #     if pairs is not None and pairs.find('://') == -1 and not os.path.isfile(pairs):
    #         raise CatBoostError("Invalid pairs path='{}': file does not exist.".format(pairs))

    def _check_delimiter(self, delimiter):
        if not isinstance(delimiter, STRING_TYPES):
            raise CatBoostError("Invalid delimiter type={} : must be str().".format(type(delimiter)))
        if len(delimiter) < 1:
            raise CatBoostError("Invalid delimiter length={} : must be > 0.".format(len(delimiter)))

    def _check_column_description_type(self, column_description):
        """
        Check type of column_description parameter.
        """
        if not isinstance(column_description, PATH_TYPES):
            raise CatBoostError("Invalid column_description type={}: must be str() or pathlib.Path().".format(type(column_description)))

    def _check_string_feature_type(self, features, features_name):
        """
        Check type of cat_feature parameter.
        """
        if not isinstance(features, (list, np.ndarray)):
            raise CatBoostError("Invalid {} type={}: must be list() or np.ndarray().".format(features_name, type(features)))

    def _check_string_feature_value(self, features, features_count, features_name):
        """
        Check values in cat_feature parameter. Must be int indices.
        """
        for indx, feature in enumerate(features):
            if not isinstance(feature, INTEGER_TYPES):
                raise CatBoostError("Invalid {}[{}] = {} value type={}: must be int().".format(features_name, indx, feature, type(feature)))
            if feature >= features_count:
                raise CatBoostError("Invalid {}[{}] = {} value: index must be < {}.".format(features_name, indx, feature, features_count))

    def _check_pairs_type(self, pairs):
        """
        Check type of pairs parameter.
        """
        if not isinstance(pairs, (list, np.ndarray, DataFrame)):
            raise CatBoostError("Invalid pairs type={}: must be list(), np.ndarray() or pd.DataFrame.".format(type(pairs)))

    def _check_pairs_value(self, pairs):
        """
        Check values in pairs parameter. Must be int indices.
        """
        for pair_id, pair in enumerate(pairs):
            if (len(pair) != 2):
                raise CatBoostError("Length of pairs[{}] isn't equal to 2.".format(pair_id))
            for i, index in enumerate(pair):
                if not isinstance(index, INTEGER_TYPES):
                    raise CatBoostError("Invalid pairs[{}][{}] = {} value type={}: must be int().".format(pair_id, i, index, type(index)))

    def _check_data_type(self, data):
        """
        Check type of data.
        """
        if not isinstance(data, (PATH_TYPES, ARRAY_TYPES, SPARSE_MATRIX_TYPES, FeaturesData)):
            raise CatBoostError(
                "Invalid data type={}: data must be list(), np.ndarray(), DataFrame(), Series(), FeaturesData " +
                " scipy.sparse matrix or filename str() or pathlib.Path().".format(type(data))
            )

    def _check_data_empty(self, data):
        """
        Check that data is not empty (0 objects is ok).
        note: already checked if data is FeatureType, so no need to check again
        """
        pass

    def _check_label_type(self, label):
        """
        Check type of label.
        """
        pass

    def _check_label_empty(self, label):
        """
        Check label is not empty.
        """
        pass

    def _check_label_shape(self, label, samples_count):
        """
        Check label length and dimension.
        """
        pass

    def _check_baseline_type(self, baseline):
        """
        Check type of baseline parameter.
        """
        pass

    def _check_baseline_shape(self, baseline, samples_count):
        """
        Check baseline length and dimension.
        """
        pass

    def _check_weight_type(self, weight):
        """
        Check type of weight parameter.
        """
        pass


    def _check_weight_shape(self, weight, samples_count):
        """
        Check weight length.
        """
        pass

    def _check_group_id_type(self, group_id):
        """
        Check type of group_id parameter.
        """
        pass

    def _check_group_id_shape(self, group_id, samples_count):
        """
        Check group_id length.
        """
        pass

    def _check_group_weight_type(self, group_weight):
        """
        Check type of group_weight parameter.
        """
        pass

    def _check_group_weight_shape(self, group_weight, samples_count):
        """
        Check group_weight length.
        """
        pass

    def _check_subgroup_id_type(self, subgroup_id):
        """
        Check type of subgroup_id parameter.
        """
        pass

    def _check_subgroup_id_shape(self, subgroup_id, samples_count):
        """
        Check subgroup_id length.
        """
        pass

    def _check_feature_names(self, feature_names, num_col=None):
        pass

    def _check_thread_count(self, thread_count):
        pass

    def slice(self, rindex):
        if not isinstance(rindex, ARRAY_TYPES):
            raise CatBoostError("Invalid rindex type={} : must be list or numpy.ndarray".format(type(rindex)))
        slicedPool = Pool(None)
        slicedPool._take_slice(self, rindex)
        return slicedPool

    def set_pairs(self, pairs):
        self._check_pairs_type(pairs)
        if isinstance(pairs, DataFrame):
            pairs = pairs.values
        self._check_pairs_value(pairs)
        self._set_pairs(pairs)
        return self

    def set_feature_names(self, feature_names):
        self._check_feature_names(feature_names)
        self._set_feature_names(feature_names)
        return self

    def set_baseline(self, baseline):
        self._check_baseline_type(baseline)
        baseline = self._if_pandas_to_numpy(baseline)
        baseline = np.reshape(baseline, (self.num_row(), -1))
        self._check_baseline_shape(baseline, self.num_row())
        self._set_baseline(baseline)
        return self

    def set_weight(self, weight):
        self._check_weight_type(weight)
        weight = self._if_pandas_to_numpy(weight)
        self._check_weight_shape(weight, self.num_row())
        self._set_weight(weight)
        return self

    def set_group_id(self, group_id):
        self._check_group_id_type(group_id)
        group_id = self._if_pandas_to_numpy(group_id)
        self._check_group_id_shape(group_id, self.num_row())
        self._set_group_id(group_id)
        return self

    def set_group_weight(self, group_weight):
        self._check_group_weight_type(group_weight)
        group_weight = self._if_pandas_to_numpy(group_weight)
        self._check_group_weight_shape(group_weight, self.num_row())
        self._set_group_weight(group_weight)
        return self

    def set_subgroup_id(self, subgroup_id):
        self._check_subgroup_id_type(subgroup_id)
        subgroup_id = self._if_pandas_to_numpy(subgroup_id)
        self._check_subgroup_id_shape(subgroup_id, self.num_row())
        self._set_subgroup_id(subgroup_id)
        return self

    def set_pairs_weight(self, pairs_weight):
        self._check_weight_type(pairs_weight)
        pairs_weight = self._if_pandas_to_numpy(pairs_weight)
        self._check_weight_shape(pairs_weight, self.num_pairs())
        self._set_pairs_weight(pairs_weight)
        return self

    def save(self, fname):
        """
        Save the quantized pool to a file.

        Parameters
        ----------
        fname : string or pathlib.Path
            Output file name.
        """
        if not self.is_quantized():
            raise CatBoostError('Pool is not quantized')

        if not isinstance(fname, PATH_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str() or pathlib.Path().".format(type(fname)))

        self._save(fname)

    def quantize(self, ignored_features=None, per_float_feature_quantization=None, border_count=None,
                 max_bin=None, feature_border_type=None, sparse_features_conflict_fraction=None,
                 nan_mode=None, input_borders=None, task_type=None, used_ram_limit=None, random_seed=None, **kwargs):
        pass

    def _if_pandas_to_numpy(self, array):
        if isinstance(array, Series):
            array = array.values
        if isinstance(array, DataFrame):
            array = np.transpose(array.values)[0]
        return array

    def _label_if_pandas_to_numpy(self, label):
        if isinstance(label, Series):
            label = label.values
        if isinstance(label, DataFrame):
            label = label.values
        return label

    def _read(
        self,
        pool_file,
        column_description,
        pairs,
        feature_names_path,
        delimiter,
        has_header,
        ignore_csv_quoting,
        thread_count,
        quantization_params=None,
        log_cout=sys.stdout,
        log_cerr=sys.stderr
    ):
        pass


class CatBoost(catboost.CatBoost):

    def predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'Exponent' : return Exponent of raw formula value.
            - 'RMSEWithUncertainty': return standard deviation for RMSEWithUncertainty loss function
              (logarithm of the standard deviation is returned by default).

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)


    def _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type="CPU"):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type)

        predictions = self._base_predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
        return predictions[0] if data_is_single_object else predictions


    def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
        return data, False
