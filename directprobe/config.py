# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-08-14 13:44:20
# Last modified: 2021-04-06 10:33:21

"""
Load configuation from file.
"""

from pathlib import Path

import torch


class Config:
    def __init__(self, config):
        self._get_runpath(config)

        self._get_data(config)

        self._get_clustering(config)

    def _get_clustering(self, config):
        self.mode = config.mode
        self.probing_cluster_path = config.probing_cluster_path
        self.enable_cuda = bool(config.enable_cuda)
        self.rate = float(config.rate)
        cuda = self.enable_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else
                                   'cpu')

    def _get_runpath(self, config):
        output_path = Path(config.output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        else:
            if len(list(output_path.iterdir())) != 0:
                raise Exception('The results directory is non-empty!')
        self.cluster_path = output_path / 'clusters.txt'
        self.log_path = output_path / 'log.txt'
        self.prediction_path = output_path / 'prediction.txt'
        self.dis_path = output_path / 'dis.txt'

    def _get_data(self, config):
        self.entities_path = config.entities_path
        self.test_entities_path = config.test_entities_path
        self.label_set_path = config.label_set_path
        self.embeddings_path = config.embeddings_path
        self.test_embeddings_path = config.test_embeddings_path
