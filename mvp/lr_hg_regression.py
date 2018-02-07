import copy
from evaluator import FoldEvaluator
import numpy as np
import numpy.linalg as LA
import os
import sys
from time import time
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from util.util_io import calc_conv_laplacian, calc_hyper_laplacian, \
    calc_conv_laplacian_drop, calc_hyper_laplacian_similarity, \
    mvp_print_performance


class SemiGraphRerank:
    def __init__(self, feature_fname, origin_rank_fname, fold_count,
                 parameters):
        self.feature = np.genfromtxt(feature_fname, dtype=float, delimiter=',')
        self.fold_count = fold_count
        print 'feature shape:', self.feature.shape
        self.origin_ranking = np.genfromtxt(origin_rank_fname,
                                            delimiter=',', dtype=float)
        print 'origin ranking shape:', self.origin_ranking.shape

        self.parameters = copy.copy(parameters)
        self.L = []
        if parameters['graph_type'] == 'convention':
            self.L = np.matrix(calc_conv_laplacian(self.feature,
                                                   parameters['normalize']))
        elif parameters['graph_type'] == 'hyper':
            self.L = np.matrix(calc_hyper_laplacian(self.feature,
                                                    parameters['k'],
                                                    parameters['normalize']))
        elif parameters['graph_type'] == 'conv_drop':
            self.L = np.matrix(calc_conv_laplacian_drop(self.feature,
                                                    parameters['k'],
                                                    parameters['normalize']))
        elif parameters['graph_type'] == 'hyper_sim':
            self.L = np.matrix(calc_hyper_laplacian_similarity(self.feature,
                                                    parameters['k'],
                                                    parameters['normalize']))
        else:
            print 'unexpected graph_type:', parameters['graph_type']
            exit()
        self.n = len(self.origin_ranking)
        self.y_fold = []
        self.f_fold = []
        self.w = np.matrix(np.zeros((self.feature.shape[1], 1), dtype=float))
        self.y_mask_folds = []
        for i in range(1, fold_count + 1):
            self.f_fold.append(np.matrix(np.random.rand(self.n, 1)))

            y_array = np.genfromtxt(
                origin_rank_fname.replace('.csv', '_' + str(i) + '.csv'),
                delimiter=',', dtype=float
            )
            self.y_fold.append(np.matrix(y_array).T)

            train_mask = np.zeros(self.feature.shape[0])
            for i in xrange(self.feature.shape[0]):
                if abs(y_array[i]) > 1e-10:
                    train_mask[i] = 1.0
            self.y_mask_folds.append(np.matrix(np.diag(train_mask)))

    def ranking(self):
        for i in range(self.fold_count):
            self._ranking(i)
        return self.f_fold

    def update_model(self, model_parameters):
        # need to construct new Laplacian matrix
        if 'k' in model_parameters.keys() and (not model_parameters['k'] ==
            self.parameters['k']):
            # update the Laplacian matrix
            if model_parameters['graph_type'] == 'convention':
                self.L = np.matrix(calc_conv_laplacian(self.feature,
                                        self.parameters['normalize']))
            elif model_parameters['graph_type'] == 'hyper':
                if model_parameters['k'] > self.n - 1:
                    return False
                self.L = np.matrix(calc_hyper_laplacian(self.feature,
                                        model_parameters['k'],
                                        model_parameters['normalize']))
            elif model_parameters['graph_type'] == 'conv_drop':
                if model_parameters['k'] > self.n - 1:
                    return False
                self.L = np.matrix(calc_conv_laplacian_drop(self.feature,
                                        model_parameters['k'],
                                        model_parameters['normalize']))
            elif model_parameters['graph_type'] == 'hyper_sim':
                if model_parameters['k'] > self.n - 1:
                    return False
                self.L = np.matrix(calc_hyper_laplacian_similarity(
                    self.feature, model_parameters['k'],
                    model_parameters['normalize']))
            else:
                print 'shit graph_type:', model_parameters['graph_type']
                exit()
        # update model parameters
        for parameter_kv in model_parameters.iteritems():
            self.parameters[parameter_kv[0]] = parameter_kv[1]
        return True

    def _loss(self, fold_index):
        loss_gr = (0.5 * (self.f_fold[fold_index].T * self.L
                          * self.f_fold[fold_index]))[0, 0]
        prediction = self.feature * self.w

        loss_gr = 0.5 * (prediction.T * self.L * prediction)[0, 0]
        loss_hr = 0.5 * self.parameters['lam_i'] * (
            (prediction - self.y_fold[fold_index]).T
            * self.y_mask_folds[fold_index]
            * (prediction - self.y_fold[fold_index])
        )[0, 0]
        return [loss_gr + loss_hr, loss_hr, loss_gr]

    def _ranking(self, fold_index):
        time_start = time()
        b = self.parameters['lam_i'] * (
            self.feature.T * self.y_mask_folds[fold_index]
            * self.y_fold[fold_index]
        )
        self.w = np.matrix(
            LA.solve(self.feature.T * self.L * self.feature
                     + self.parameters['lam_i'] * self.feature.T
                     * self.y_mask_folds[fold_index] * self.feature, b)
        )
        self.f_fold[fold_index] = self.feature * self.w
        print 'time:', '{:.4f}'.format(time() - time_start)

if __name__ == '__main__':
    feature_path = os.path.join(sys.path[0], 'data', 'mvp')
    fold_count = 10
    parameter = {'normalize': False, 'graph_type': 'hyper_sim', 'k': 2,
                 'lam_i': 0.01}
    ofname = 'lr_hg.pred'
    fold_evaluate = FoldEvaluator(
        os.path.join(feature_path, 'ground_truth.csv'), fold_count
    )
    rerank = SemiGraphRerank(
        os.path.join(feature_path, 'feature.csv'),
        os.path.join(feature_path, 'ground_truth.csv'), fold_count, parameter)
    generated_ranking = rerank.ranking()
    np.savetxt(ofname, generated_ranking, fmt='%.8f')

    nmse = 0.0
    tau = 0.0
    rho = 0.0
    for i in xrange(fold_count):
        print('-------------------------')
        performance = fold_evaluate.single_fold_evaluate(generated_ranking[i],
                                                         i)
        nmse += performance['nmse']
        tau += performance['tau'][0]
        rho += performance['rho'][0]
        mvp_print_performance(performance)
    print('average performance')
    print('nmse:', nmse / fold_count, 'tau:', tau / fold_count, 'rho:',
          rho / fold_count)
