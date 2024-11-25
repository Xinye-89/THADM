# -*- coding: utf-8 -*-

import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import _check_sample_weight
from sklearn.cluster._dbscan_inner import dbscan_inner

@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=False,
)
def detector(
    X,
    eps=0.5,
    *,
    min_samples=5,
    metric="minkowski",
    metric_params=None,
    algorithm="auto",
    leaf_size=30,
    p=2,
    sample_weight=None,
    n_jobs=None,
):

    est = DETECTOR(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs,
    )
    est.fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_


class DETECTOR(ClusterMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "eps": [Interval(Real, 0.0, None, closed="neither")],
        "min_samples": [Interval(Integral, 1, None, closed="left")],
        "metric": [
            StrOptions(set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        "metric_params": [dict, None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0.0, None, closed="left"), None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    @_fit_context(
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, sample_weight=None):

        X = self._validate_data(X, accept_sparse="csr")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == "precomputed" and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            X = X.copy()  # copy to avoid in-place modification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())

        neighbors_model = NearestNeighbors(
            radius=self.eps,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        else:
            n_neighbors = np.array(
                [np.sum(sample_weight[neighbors]) for neighbors in neighborhoods]
            )

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):

        self.fit(X, sample_weight=sample_weight)
        return self.labels_

    def _more_tags(self):
        return {"pairwise": self.metric == "precomputed"}

def detector_eval(feats,labels,eps,min_samples):
    # eps=1.8
    # min_samples=15
    dbs=DETECTOR(eps=eps,metric='minkowski',p=2,min_samples=min_samples)
    dbs.fit(feats)
    # print(Counter(dbs.labels_))
    pred_org=dbs.labels_
    pred=[0 if x<0 else 1 for x in pred_org]
    result=pd.DataFrame({'A':labels,'B':pred}).value_counts().reset_index()
    result['method']='thadm' 
    result['eps']=eps 
    result['min_samples']=min_samples
    result['feats_shape']=str(feats.shape)
    accuracy=accuracy_score(labels,pred)
    precision=precision_score(labels,pred)
    recall=recall_score(labels,pred)
    f_score=f1_score(labels,pred)
    auc=roc_auc_score(labels,pred)
    return result,['thadm',str(feats.shape),eps,min_samples,accuracy,precision,recall,f_score,auc]

