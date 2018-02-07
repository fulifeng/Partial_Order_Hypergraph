import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error


class FoldEvaluator:
    def __init__(self, ground_truth_fname, fold_count):
        self.origin_ranking = np.matrix(np.genfromtxt(ground_truth_fname,
                                                      delimiter=',',
                                                      dtype=float)).T
        self.y_fold = []
        for i in range(1, fold_count + 1):
            self.y_fold.append(
                np.matrix(
                    np.genfromtxt(ground_truth_fname.replace('.csv', '_' +
                                                             str(i) + '.csv'),
                                  delimiter=',', dtype=float)).T)

    def evaluate(self, generate_ranking_fold):
        fold_count = len(generate_ranking_fold)
        performances = np.zeros([fold_count, 7], dtype=float)
        for i in range(fold_count):
            performances[i] = self._evaluate(generate_ranking_fold[i], i)
        average_performance = np.mean(performances, axis=0)
        return {
            'nmse': average_performance[0],
            'tau': [average_performance[1], average_performance[2]],
            'rho': [average_performance[3], average_performance[4]]
        }

    def _evaluate(self, generated_ranking, fold_index):
        test_generated = []
        test_ground_truth = []
        for i in xrange(self.y_fold[fold_index].shape[0]):
            if self.y_fold[fold_index][i, 0] < 1e-9:
                test_generated.append(generated_ranking[i, 0])
                test_ground_truth.append(self.origin_ranking[i, 0])
        test_generated = np.asarray(test_generated)
        test_ground_truth = np.asarray(test_ground_truth)

        tau, tau_pvalue = sps.kendalltau(test_generated,
                                         test_ground_truth)

        rho, rho_pvalue = sps.spearmanr(test_generated,
                                        test_ground_truth, axis=None)

        nmse = (np.linalg.norm(test_generated - test_ground_truth) ** 2) / \
               (np.linalg.norm(test_ground_truth) ** 2)
        return np.asarray([nmse, tau, tau_pvalue, rho, rho_pvalue],
                          dtype=float)

    def single_fold_evaluate(self, generated_ranking, fold_index):
        test_generated = []
        test_ground_truth = []
        for i in xrange(self.y_fold[fold_index].shape[0]):
            if self.y_fold[fold_index][i, 0] < 1e-9:
                test_generated.append(generated_ranking[i, 0])
                test_ground_truth.append(self.origin_ranking[i, 0])
        test_generated = np.asarray(test_generated)
        test_ground_truth = np.asarray(test_ground_truth)

        tau, tau_pvalue = sps.kendalltau(test_generated,
                                         test_ground_truth)

        rho, rho_pvalue = sps.spearmanr(test_generated,
                                        test_ground_truth, axis=None)

        nmse = (np.linalg.norm(test_generated - test_ground_truth) ** 2) / \
               (np.linalg.norm(test_ground_truth) ** 2)

        return {'tau': [tau, tau_pvalue], 'rho': [rho, rho_pvalue], 'nmse': nmse}

    def compare(self, current_performance, origin_performance):
        is_better = {}
        for metric_name in origin_performance.iterkeys():
            if metric_name == 'tau' or metric_name == 'rho':
                if current_performance[metric_name] > \
                        origin_performance[metric_name]:
                    is_better[metric_name] = True
                else:
                    is_better[metric_name] = False
            else:
                if current_performance[metric_name] < \
                        origin_performance[metric_name]:
                    is_better[metric_name] = True
                else:
                    is_better[metric_name] = False
        return is_better
