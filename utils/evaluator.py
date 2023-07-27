import os
import json
import time

from logging import getLogger
from utils.eval_funcs import top_k
# from utils.eval_funcs import get_bleu
from utils.abstract_evaluator import AbstractEvaluator
allowed_metrics = ['Precision', 'Recall', 'F1', 'MRR', 'MAP', 'NDCG']
from collections import defaultdict

class TrajLocPredEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config['metrics']  # 评估指标, 是一个 list
        self.config = config
        self.topk = config['topk']
        self.result = {}
        # 兼容全样本评估与负样本评估
        self.evaluate_method = config['evaluate_method']
        self.intermediate_result = defaultdict(float)
        self._check_config()
        self._logger = getLogger()
        self.steps_to_evaluate = config['evaluate_steps_index']

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in allowed_metrics:
                raise ValueError('the metric is not allowed in \
                    TrajLocPredEvaluator')

    def collect(self, batch):
        """
        Args:
            batch (dict): contains three keys: uid, loc_true, and loc_pred.
            uid (list): 来自于 batch 中的 uid，通过索引可以确定 loc_true 与 loc_pred
                中每一行（元素）是哪个用户的一次输入。
            loc_true (list): 期望地点(target)，来自于 batch 中的 target。
                对于负样本评估，loc_pred 中第一个点是 target 的置信度，后面的都是负样本的
            loc_pred (matrix): 实际上模型的输出，batch_size * output_dim.
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        if(type(self.topk) == type(0)):
            total = len(batch['loc_true'])
            self.intermediate_result['total'] += total
            for step in self.steps_to_evaluate:
                hit, rank, dcg = top_k(batch['loc_pred'], batch['loc_true'], self.topk, step)
                self.intermediate_result['hit_s'+str(step)] += hit
                self.intermediate_result['rank_s'+str(step)] += rank
                self.intermediate_result['dcg_s'+str(step)] += dcg
        elif(type(self.topk) == type([])):
            total = len(batch['loc_true'])
            self.intermediate_result['total'] += total
            for step in self.steps_to_evaluate:
                for idx in range(len(self.topk)):
                    hit, rank, dcg = top_k(batch['loc_pred'], batch['loc_true'], self.topk[idx], step)
                    self.intermediate_result['hit' + str(self.topk[idx]) +'_s'+str(step)] += hit
                    self.intermediate_result['rank' + str(self.topk[idx])+'_s'+str(step)] += rank
                    self.intermediate_result['dcg' + str(self.topk[idx])+'_s'+str(step)] += dcg

    def evaluate(self):
        if(type(self.topk) == type(0)):
            for step in self.steps_to_evaluate:
                precision_key = 'Precision@{}_s{}'.format(self.topk, step)
                precision = self.intermediate_result['hit_s'+str(step)] / (
                        self.intermediate_result['total'] * self.topk)
                if 'Precision' in self.metrics:
                    self.result[precision_key] = precision
                # recall is used to valid in the trainning, so must exit
                recall_key = 'Recall@{}_s{}'.format(self.topk, step)
                recall = self.intermediate_result['hit_s'+str(step)] \
                        / self.intermediate_result['total']
                self.result[recall_key] = recall
                if 'F1' in self.metrics:
                    f1_key = 'F1@{}_s{}'.format(self.topk, step)
                    if precision + recall == 0:
                        self.result[f1_key] = 0.0
                    else:
                        self.result[f1_key] = (2 * precision * recall) / (precision +
                                                                        recall)
                if 'MRR' in self.metrics:
                    mrr_key = 'MRR@{}_s{}'.format(self.topk, step)
                    self.result[mrr_key] = self.intermediate_result['rank_s'+str(step)] \
                                        / self.intermediate_result['total']
                if 'MAP' in self.metrics:
                    map_key = 'MAP@{}_s{}'.format(self.topk, step)
                    self.result[map_key] = self.intermediate_result['rank_s'+str(step)] \
                                        / self.intermediate_result['total']
                if 'NDCG' in self.metrics:
                    ndcg_key = 'NDCG@{}_s{}'.format(self.topk, step)
                    self.result[ndcg_key] = self.intermediate_result['dcg_s'+str(step)] \
                                            / self.intermediate_result['total']
        elif(type(self.topk) == type([])):
            
            for step in self.steps_to_evaluate:
                for k in self.topk:
                    precision_key = 'Precision@{}_s{}'.format(k, step)
                    precision = self.intermediate_result['hit' + str(k)+'_s'+str(step)] / (
                            self.intermediate_result['total'] * k)
                    if 'Precision' in self.metrics:
                        self.result[precision_key] = precision
                    # recall is used to valid in the trainning, so must exit
                    recall_key = 'Recall@{}_s{}'.format(k, step)
                    recall = self.intermediate_result['hit' + str(k)+'_s'+str(step)] \
                            / self.intermediate_result['total']
                    self.result[recall_key] = recall
                    if 'F1' in self.metrics:
                        f1_key = 'F1@{}_s{}'.format(k, step)
                        if precision + recall == 0:
                            self.result[f1_key] = 0.0
                        else:
                            self.result[f1_key] = (2 * precision * recall) / (precision +
                                                                            recall)
                    if 'MRR' in self.metrics:
                        mrr_key = 'MRR@{}_s{}'.format(k, step)
                        self.result[mrr_key] = self.intermediate_result['rank' + str(k)+'_s'+str(step)] \
                                            / self.intermediate_result['total']
                    if 'MAP' in self.metrics:
                        map_key = 'MAP@{}_s{}'.format(k, step)
                        self.result[map_key] = self.intermediate_result['rank' + str(k)+'_s'+str(step)] \
                                            / self.intermediate_result['total']
                    if 'NDCG' in self.metrics:
                        ndcg_key = 'NDCG@{}_s{}'.format(k, step)
                        self.result[ndcg_key] = self.intermediate_result['dcg' + str(k)+'_s'+str(step)] \
                                                / self.intermediate_result['total']

        return self.result

    def save_result(self, save_path, filename=None):
        self.evaluate()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if filename is None:
            # 使用时间戳
            filename = time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        self._logger.info('evaluate result is {}'.format(json.dumps(self.result, indent=1)))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') \
                as f:
            json.dump(self.result, f)

    def clear(self):
        self.result = {}
        self.intermediate_result.clear()
