# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.8.0
#
# Date: 2020-12-29 13:46:09
# Last modified: 2021-04-08 09:29:58

"""
Analyzing functions.
"""

import logging
from typing import List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

import numpy as np
from directprobe.distanceQ import DistanceQ
from directprobe.space import Space

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(self):
        pass

    def predict(self, q, ann, embeds):
        return self.points2convex(q, ann, embeds)

    def points2convex(
            self,
            q: DistanceQ,
            ann: np.array,
            embeds: np.array
            ) -> Tuple[float, List[List[Tuple[int, int, float]]]]:
        """
        Make predictions for `embeds` based on the distances
        between each point in `embeds` and all the clusters.

        Returns:
            - List((cluster_id, major_label, distance)):
                    the ranking of clusters for each test point
                    based on the distance.
        """
        assert len(ann) == len(embeds)
        clusters = q.clusters

        logger.info('Computing the distances...')
        return_list = []
        correct = 0
        for i, (label, vec) in tqdm(
                enumerate(zip(ann, embeds)), total=len(ann)):
            data = []
            diss = []
            # select all the points belong to cluster j
            for j in range(len(clusters)):
                cls = clusters[j]
                indexs = torch.LongTensor(cls.indices)
                vecs = q.fix_embeddings[indexs]
                vecs = vecs.cpu().numpy()
                data.append((vecs, vec))
            diss = Parallel(n_jobs=30, prefer='processes', verbose=0,
                            batch_size='auto')(
                delayed(Space.point2hull)(X1, X2) for X1, X2 in data)

            diss = np.array(diss)
            sorted_indices = np.argsort(diss)
            preds = [
                    (j, clusters[j].major_label, diss[j])
                    for j in sorted_indices]

            if preds[0][1] == label:
                correct += 1
            return_list.append(preds)
        acc = correct / len(ann)
        return acc, return_list

    def convex2convex(
            self,
            q: DistanceQ
            ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """Return the minimum distance between the clusters.
        """
        data = []
        clusters = q.clusters

        indices = [i for i in range(len(q.clusters))]

        # Prepare the embeddings
        for i in range(len(q.clusters)):
            cls = q.clusters[i]
            indexs = torch.LongTensor(cls.indices)
            vecs = q.fix_embeddings[indexs]
            vecs = vecs.cpu().numpy()
            data.append(vecs)

        indexs = [(i, j) for i in indices for j in indices
                  if i < j]
        # Only compute the distances between clusters with different labels
        indexs = [(i, j) for i, j in indexs
                  if clusters[i].major_label != clusters[j].major_label]

        data = [(data[i], data[j]) for i, j in indexs]
        labels = [(i, clusters[i].major_label, j, clusters[j].major_label)
                  for i, j in indexs]

        diss = Parallel(n_jobs=10, prefer='processes', verbose=0,
                        batch_size=1)(
            delayed(Space.hull2hull)(X1, X2) for X1, X2 in data)
        return diss, labels

    def convex2convex_pair_wise(
            self,
            q: DistanceQ
            ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """Return the minimum pair-wise distance between the clusters.
        """
        data = []
        clusters = q.clusters

        indices = [i for i in range(len(q.clusters))]

        # Prepare the embeddings
        for i in range(len(q.clusters)):
            cls = q.clusters[i]
            indexs = torch.LongTensor(cls.indices)
            vecs = q.fix_embeddings[indexs]
            # vecs = vecs.cpu().numpy()
            data.append(vecs)

        indexs = [(i, j) for i in indices for j in indices
                  if i < j]
        indexs = [(i, j) for i, j in indexs
                  if clusters[i].major_label != clusters[j].major_label]

        data = [(data[i], data[j]) for i, j in indexs]
        labels = [(clusters[i].major_label, clusters[j].major_label)
                  for i, j in indexs]

        diss = []
        logger.info('Computing the pair-wise distance...')
        for X1, X2 in tqdm(data):
            dis = torch.mean(torch.cdist(X1, X2))
            diss.append(dis)

        return diss, labels

    def convexhulls(
            self,
            q: DistanceQ) -> Tuple[List[int], List[Tuple[float, float]]]:
        """Computing the average distance and STD distance inside each cluster.
        """
        data = []
        labels = []
        for i in range(len(q.clusters)):
            cls = q.clusters[i]
            indexs = torch.LongTensor(cls.indices)
            vecs = q.fix_embeddings[indexs]
            data.append(vecs)
            labels.append(cls.major_label)
        mean_std = []
        re_labels = []
        for tag, embeds in tqdm(zip(labels, data), total=len(data)):
            pdist = torch.nn.functional.pdist(embeds)
            # If there is only one point in one cluster
            if len(embeds) == 1:
                continue
            # If thare are only two points in one cluster
            elif len(pdist) == 1:
                continue
            else:
                mean = torch.mean(pdist)
                std = torch.std(pdist)
            mean_std.append((mean, std))
            re_labels.append(tag)
        return re_labels, mean_std
