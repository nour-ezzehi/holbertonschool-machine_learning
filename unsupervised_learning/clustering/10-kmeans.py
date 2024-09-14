#!/usr/bin/env python3
""" 10. Hello, sklearn! """
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    kmeans_ = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_.fit(X)
    C = kmeans_.cluster_centers_
    clss = kmeans_.labels_

    return C, clss
