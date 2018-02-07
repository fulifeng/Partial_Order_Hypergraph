import argparse
import copy
from evaluator import FoldEvaluator
import numpy as np
import numpy.linalg as LA
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from util.util_io import calc_conv_laplacian, calc_hyper_laplacian, \
    calc_conv_laplacian_drop, calc_hyper_laplacian_similarity, \
    mvp_print_performance


class GraphRerank:
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
            print 'shit graph_type:', parameters['graph_type']
            exit()
        # self.y = np.matrix(self.origin_ranking).T
        self.n = len(self.origin_ranking)
        # self.f = np.matrix(np.random.rand(self.n, 1))
        self.y_fold = []
        self.f_fold = []
        for i in range(1, fold_count + 1):
            self.y_fold.append(np.matrix(np.genfromtxt(origin_rank_fname.replace('.csv',
                                                '_' + str(i) + '.csv'),
                                             delimiter=',',
                                             dtype=float)).T)
            self.f_fold.append(np.matrix(np.random.rand(self.n, 1)))

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
        loss_hr = 0.5 * self.parameters['lam_i'] * np.power(
            LA.norm(self.f_fold[fold_index] - self.y_fold[fold_index]), 2)
        return [loss_gr + loss_hr, loss_hr, loss_gr]

    def _ranking(self, fold_index):
        print self._loss(fold_index)
        b = self.parameters['lam_i'] * self.y_fold[fold_index]
        # aaaa = self.L[0, 0]
        # temp = self.L + (self.parameters['lam_i']) * np.matrix(
        #     np.identity(self.n))
        self.f_fold[fold_index] = np.matrix(LA.solve(
            self.L + (self.parameters['lam_i']) * np.matrix(
                np.identity(self.n)), b))
        print 'final loss:', self._loss(fold_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='simple',
                        help='denote the type of graph')
    args = parser.parse_args()

    if args.type == 'simple':
        parameter = {'graph_type': 'conv_drop', 'lam_i': 400, 'k': 200,
                     'normalize': True}
        ofname = 'simple_graph.pred'
    elif args.type == 'hyper':
        parameter = {'graph_type': 'hyper_sim', 'lam_i': 1e4, 'k': 3,
                     'normalize': False}
        ofname = 'hypergraph.pred'

    feature_path = os.path.join(sys.path[0], 'data', 'mvp')
    fold_count = 10
    fold_evaluate = FoldEvaluator(
        os.path.join(feature_path, 'ground_truth.csv'), fold_count
    )
    rerank = GraphRerank(os.path.join(feature_path, 'feature.csv'),
                         os.path.join(feature_path, 'ground_truth.csv'),
                         fold_count, parameter)
    generated_ranking = rerank.ranking()
    np.savetxt(ofname, generated_ranking, fmt='%.8f')

    nmse = 0.0
    tau = 0.0
    rho = 0.0
    for i in xrange(fold_count):
        print '-------------------------'
        performance = fold_evaluate.single_fold_evaluate(generated_ranking[i],
                                                         i)
        nmse += performance['nmse']
        tau += performance['tau'][0]
        rho += performance['rho'][0]
        mvp_print_performance(performance)
    print 'average performance'
    print 'nmse:', nmse / fold_count, 'tau:', tau / fold_count, 'rho:', \
        rho / fold_count