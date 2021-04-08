# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-07-24 09:55:34
# Last modified: 2021-04-08 09:30:52

"""
Some utility functions, including loading and saving data.
"""

import collections
from typing import List, Tuple, TextIO

import numpy as np
import torch

from directprobe.distanceQ import DistanceQ


Pair = collections.namedtuple('Pair', ['Entity', 'Label'])


def load_entities(path: TextIO):
    reval = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split('\t')
            reval.append(Pair(*s))
    return reval


def load_labels(path: TextIO):
    reval = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            reval.add(line.strip())
    return reval


def load_embeddings(path: TextIO):
    reval = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            vec = [float(v) for v in s]
            reval.append(vec)
    return reval


def write_predictions(
        path: TextIO,
        cluster_list: List[List[Tuple[int, str, float]]],
        real_labels: List[str]):
    """ Write distances to the file.
    """
    assert len(cluster_list) == len(real_labels)
    with open(path, 'w', encoding='utf8') as f:
        for i, label_dis_pair_list in enumerate(cluster_list):
            line = [str(real_labels[i])]
            for cls_id, label, dis in label_dis_pair_list:
                s = '{a}-{b},{c:0.4f}'.format(
                        a=str(cls_id), b=str(label), c=dis)
                line.append(s)
            line = '\t'.join(line)
            f.write(line+'\n')


def write_clusters(path: TextIO, q: DistanceQ):
    """Write down the clusters.
    """
    ans = [-1] * len(q.fix_embeddings)
    ans = np.array(ans)
    indices = torch.nonzero(q.active).reshape(-1)
    indices = indices.cpu().numpy().tolist()
    for i, idx in enumerate(indices):
        t = q.clusters[idx]
        ans[t.indices] = i

    with open(path, 'w', encoding='utf8') as f:
        for i in ans:
            f.write(str(i)+'\n')


def load_clusters(path: TextIO) -> List[List[int]]:
    """ Load the clusters from the file.

    Return:
        reval[i] is the list of points that belong
        to cluster i.
    """
    cluster_labels = []
    with open(path, encoding='utf8') as f:
        for line in f:
            cluster_labels.append(int(line.strip()))
    cluster_num = max(cluster_labels)+1
    reval = [[] for _ in range(cluster_num)]

    for i, v in enumerate(cluster_labels):
        reval[v].append(i)
    return reval


def assign_labels(
        clusters_indices: List[List[int]],
        annotation: np.array) -> List[List[int]]:
    """ Assign labele to each cluster.
    """
    labels = []
    for cls in clusters_indices:
        labs = [annotation[i] for i in cls]
        labels.append(labs)
    return labels


def map_to_label(
        idx2label: dict,
        cluster_list: List[List[Tuple[int, int, float]]],
        real_labels: np.array
        ) -> Tuple[List[List[Tuple[int, str, float]]], List[str]]:
    """Map the int label to str label.

    Args:
        idx2label: A dictionary from int to str.
        cluster_list: cluster_list[i][j] is a tuple of
                      (cluster_id, label, dis),
                      represents the distance between test point i and the
                      cluster with label.
        real_labels: np.array. The real int labels for each test point.
    """
    assert len(cluster_list) == len(real_labels)
    real_labels = [idx2label[v] for v in real_labels]
    return_list = []
    for i, label_dis_pair_list in enumerate(cluster_list):
        s = [(cls_id, idx2label[label], dis)
             for cls_id, label, dis in label_dis_pair_list]
        return_list.append(s)
    return return_list, real_labels


def write_convex_dis(
        path: str,
        label_pairs: List[Tuple[str, str]],
        diss: List[float]):
    """Write distances between clusters into the file.
    """
    assert len(label_pairs) == len(diss)
    with open(path, 'w', encoding='utf8') as f:
        for (cls_i, label_i, cls_j, label_j), dis in zip(label_pairs, diss):
            s = '({a}-{b}, {c}-{d}): {e:0.4f}\n'.format(
                    a=str(cls_i), b=str(label_i),
                    c=str(cls_j), d=str(label_j),
                    e=dis)
            f.write(s)


def write_dis_inside_convex(
        path: TextIO,
        mean_std: List[Tuple[float, float]],
        labels: List[str]):
    assert len(mean_std) == len(labels)
    with open(path, 'w') as f:
        for i in range(len(labels)):
            tag = labels[i]
            mean, std = mean_std[i]
            s = '{a}: {b:.4f}\t{c:.4f}\n'
            s = s.format(a=str(tag), b=mean, c=std)
            f.write(s)
