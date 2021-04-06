# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-07-24 10:36:21
# Last modified: 2021-04-06 15:51:21

"""
Main enterance.
"""
from typing import Tuple
import logging
import logging.config
import configparser

import numpy as np
import ExAssist as EA

from probing import utils
from probing.config import Config
from probing.probing import Probe
from probing.clusters import Cluster
from probing.distanceQ import DistanceQ
from probing.analyzer import Analyzer
import probing.logconfig as cfg

logger = logging.getLogger(__name__)


def loading(
        config: Config) -> Tuple[np.array, np.array, np.array]:
    """Loading all the necessary input files.

    This function load 3 files:
        - entities: A file contains the entities and labels.
                    One entity per line.
        - label_set_path: A file contains all the possible labels.
                          We have a separate file because in some cases,
                          not all the labels occure in the training set.
        - embeddings_path: A file contains all the embeddings.
                           A vector per line.
    """
    path = config.entities_path
    logger.info('Load entities from ' + path)
    entities = utils.load_entities(path)

    # For debugging
    n = len(entities)
    # n = 200
    annotations = [entities[i].Label for i in range(n)]
    entities = [entities[i] for i in range(n)]

    s = 'Finish loading {a} entities...'
    s = s.format(a=str(len(entities)))
    logger.info(s)

    labels = sorted(list(utils.load_labels(config.label_set_path)))
    label2idx = {labels[i]: i for i in range(len(labels))}
    annotations = [label2idx[t] for t in annotations]

    logger.info('Label size={a}'.format(a=str(len(labels))))

    embeddings_path = config.embeddings_path
    logger.info('Loading embeddings from ' + embeddings_path)
    embeddings = utils.load_embeddings(embeddings_path)
    embeddings = embeddings[:n]
    logger.info('Finish loading embeddings...')

    assert len(embeddings) == n

    annotations = np.array(annotations)
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    return annotations, labels, embeddings


def load_test(config: Config):
    path = config.test_entities_path
    logger.info('Load entities from ' + path)
    entities = utils.load_entities(path)

    # For debugging
    n = len(entities)
    # n = 30
    annotations = [entities[i].Label for i in range(n)]
    entities = [entities[i] for i in range(n)]

    s = 'Finish loading {a} entities...'
    s = s.format(a=str(len(entities)))
    logger.info(s)

    labels = sorted(list(utils.load_labels(config.label_set_path)))
    label2idx = {labels[i]: i for i in range(len(labels))}
    annotations = [label2idx[t] for t in annotations]

    embeddings_path = config.test_embeddings_path
    logger.info('Loading embeddings from ' + embeddings_path)
    embeddings = utils.load_embeddings(embeddings_path)
    embeddings = embeddings[:n]
    logger.info('Finish loading embeddings...')

    assert len(embeddings) == n

    annotations = np.array(annotations)
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    assert len(annotations) == len(embeddings)
    return annotations, embeddings, label2idx


def probe(config):
    annotations, labels, embeddings = loading(config)
    probe = Probe(config)

    clusters = [Cluster([i], [label]) for
                i, label in enumerate(annotations)]

    logger.info('Initialize the Distance Queue...')
    q = DistanceQ(config, embeddings, clusters, len(labels))
    q = probe.probing(q)
    assist = EA.getAssist('Probing')
    assist.result['final number'] = len(q)
    logger.info('Dumping the clusters...')
    utils.write_clusters(config.cluster_path,  q)
    logger.info('Finish dumping the clusters...')

    config.probing_cluster_path = config.cluster_path
    prediction(config)


def prediction(config):
    s = 'Loading the clusters from {a}'
    s = s.format(a=str(config.probing_cluster_path))
    logger.info(s)
    annotations, labels, embeddings = loading(config)
    clusters_indices = utils.load_clusters(config.probing_cluster_path)

    labels_list = utils.assign_labels(clusters_indices, annotations)
    assert len(clusters_indices) == len(labels_list)
    clusters = [Cluster(indices, labs) for
                indices, labs in zip(clusters_indices, labels_list)]
    q = DistanceQ(config, embeddings, clusters, len(labels))
    logger.info('Finish loading the clusters...')

    analyzer = Analyzer()
    annotations, embeddings, label2idx = load_test(config)
    idx2label = {value: key for key, value in label2idx.items()}

    assist = EA.getAssist('Probing')
    logger.info('Start prediction...')
    acc, cluster_list = analyzer.predict(q, annotations, embeddings)
    logger.info('Acc={a}'.format(a=str(acc)))
    assist.result['acc'] = acc
    cluster_list, real_labels = utils.map_to_label(
            idx2label, cluster_list, annotations)
    logger.info('Writing predictions to file...')
    utils.write_predictions(
            config.prediction_path, cluster_list, real_labels)

    logger.info('Computing the distances between clusters...')
    diss, label_pairs = analyzer.convex2convex(q)
    label_pairs = [(cls_i, idx2label[label_i], cls_j, idx2label[label_j])
                   for cls_i, label_i, cls_j, label_j in label_pairs]
    logger.info('Writing pair-wise distances...')
    utils.write_convex_dis(config.dis_path, label_pairs, diss)

    assist.result['MinConvexDistance'] = str(np.min(diss))
    assist.result['MaxConvexDistance'] = str(np.max(diss))
    assist.result['AverageConvexDistance'] = str(np.mean(diss))
    logger.info('MinConvexDistance={a}'.format(a=str(np.min(diss))))
    logger.info('MaxConvexDistance={a}'.format(a=str(np.max(diss))))
    logger.info('AverageConvexDistance={a}'.format(a=str(np.mean(diss))))


def main():
    assist = EA.getAssist('Probing')

    # Assist is used for developping experiments
    # For real using application, it needs to be deactivated
    assist.deactivate()

    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    config.read('./config.ini', encoding='utf8')

    assist.set_config(config)
    with EA.start(assist) as assist:
        config = Config(assist.config)
        cfg.set_log_path(config.log_path)
        logging.config.dictConfig(cfg.LOGGING_CONFIG)
        if config.mode == 'prediction':
            prediction(config)
        elif config.mode == 'probing':
            probe(config)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()', sort='cumulative')
    main()
