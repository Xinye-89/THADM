# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
import torch.nn as nn
import torch
from karateclub.estimator import Estimator

from numbers import Integral, Real
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
    _UnstableArchMixin,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

def barycenter_weights(X, Y, indices, reg=1e-3):
    
    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = check_array(Y, dtype=FLOAT_DTYPES)
    indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, assume_a="pos")
        B[i, :] = w / np.sum(w)
    return B


def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X, ind, reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))


def null_space(
    M, k, k_skip=1, eigen_solver="arpack", tol=1e-6, max_iter=100, random_state=None
):
    
    if eigen_solver == "auto":
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = "arpack"
        else:
            eigen_solver = "dense"

    if eigen_solver == "arpack":
        v0 = _init_arpack_v0(M.shape[0], random_state)
        try:
            eigen_values, eigen_vectors = eigsh(
                M, k + k_skip, sigma=0.0, tol=tol, maxiter=max_iter, v0=v0
            )
        except RuntimeError as e:
            raise ValueError(
                "Error in determining null-space with ARPACK. Error message: "
                "'%s'. Note that eigen_solver='arpack' can fail when the "
                "weight matrix is singular or otherwise ill-behaved. In that "
                "case, eigen_solver='dense' is recommended. See online "
                "documentation for more information." % e
            ) from e

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == "dense":
        if hasattr(M, "toarray"):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, subset_by_index=(k_skip, k + k_skip - 1), overwrite_a=True
        )
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


def locally_linear_embedding(
    X,
    *,
    n_neighbors,
    n_components,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-6,
    max_iter=100,
    method="standard",
    hessian_tol=1e-4,
    modified_tol=1e-12,
    random_state=None,
    n_jobs=None,
):
    
    if eigen_solver not in ("auto", "arpack", "dense"):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if method not in ("standard", "hessian", "modified", "ltsa"):
        raise ValueError("unrecognized method '%s'" % method)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError(
            "output dimension must be less than or equal to input dimension"
        )
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples,  but n_samples = %d, n_neighbors = %d"
            % (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = eigen_solver != "dense"

    if method == "standard":
        W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs
        )

        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[:: M.shape[0] + 1] += 1  # W = W - I = W - I

    return null_space(
        M,
        n_components,
        k_skip=1,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )

class MNE(
    Estimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _UnstableArchMixin,
    BaseEstimator,
):
    def __init__(self, dimensions=16, alphas=[0.5, 0.5], seed=42, n_components=2, n_neighbors=5):
        self.dimensions = dimensions
        self.alphas = alphas
        self.seed = seed
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        
        self.weights = nn.Parameter(torch.tensor(145.0))

    def _create_D_inverse(self, graph):
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0 / graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_smoothing_matrix(self, graph):
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
        D_inverse = self._create_D_inverse(graph)
        A_hat = D_inverse.dot(A)
        return A_hat

    def _create_embedding(self, A_hat):
        sd = 1 / self.dimensions
        base_embedding = np.random.normal(0, sd, (A_hat.shape[0], self.dimensions))
        base_embedding, _ = np.linalg.qr(base_embedding)
        embedding = np.zeros(base_embedding.shape)
        alpha_sum = sum(self.alphas)
        for alpha in self.alphas:
            base_embedding = A_hat.dot(base_embedding)
            embedding = embedding + alpha * base_embedding
        embedding = embedding / alpha_sum
        embedding = (embedding - embedding.mean(0)) / embedding.std(0)
        return embedding

    def fit(self, graph, feats, df_train, hash1, hash2):
        np.random.seed(self.seed)
        # 训练RandNE模型
        A_hat = self._create_smoothing_matrix(graph)
        embed = self._create_embedding(A_hat)
        
        length=len(set(df_train['patient_id2']))
        tmp1=pd.DataFrame(np.concatenate((df_train[['patient_id']].drop_duplicates().values,embed[0:length]),axis=1),columns=['patient_id']+['embed'+str(x) for x in range(1,embed.shape[1]+1)])
        tmp2=pd.DataFrame(np.concatenate((df_train[['hospital_id']].drop_duplicates().values,embed[length:]),axis=1),columns=['hospital_id']+['embed'+str(x) for x in range(1,embed.shape[1]+1)])
        tmp1=pd.merge(df_train[['patient_id']],tmp1,on='patient_id',how='left')
        tmp2=pd.merge(df_train[['hospital_id']],tmp2,on='hospital_id',how='left')
        
        # 将RandNE的输出结果与原始特征拼接
        feats2 = np.concatenate((tmp1.iloc[:,1:].values, tmp2.iloc[:,1:].values, feats), axis=1)
        
        # 训练LLE模型
        local_embedding, _ = locally_linear_embedding(
            feats2, n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver='dense'
        )
        
        # 与Hash Coding结合
        feats_con=torch.cat((
            torch.tensor(hash1),
            torch.tensor(hash2),
            torch.tensor(local_embedding)*self.weights    
        ),dim=1)
        print(feats_con.shape)
        return feats_con.detach().numpy()































