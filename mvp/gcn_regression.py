from __future__ import division
from __future__ import print_function

import copy
from evaluator import FoldEvaluator
import os
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import time
import tensorflow as tf
from util.util_io import calc_gcn_adjacency, mvp_print_performance

from gcn.utils import *
from gcn.models import GCN

# Set random seed
seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class GCNRanking():
    # model parameters: epochs, hidden1, dropout, learning_rate, weight_decay
    def __init__(self, feature_fname, origin_rank_fname, fold_count,
                 model_parameters, evaluator, verbose=True):
        self.fold_count = fold_count
        self.parameters = copy.copy(model_parameters)
        self.evaluator = evaluator
        self.verbose = verbose

        self.feature = np.genfromtxt(feature_fname, dtype=float, delimiter=',')
        self.feature = normalize(self.feature)
        print('feature shape:', self.feature.shape)

        self.adj = calc_gcn_adjacency(self.feature, k=200)

        self.ground_truth = np.genfromtxt(origin_rank_fname, delimiter=',',
                                          dtype=float)

        # read ground truth by fold and construct train, val, test, lable
        # and mask
        self.y_train_folds = []
        self.y_val_folds = []
        self.train_mask_folds = []
        self.val_mask_folds = []
        for i in range(1, fold_count + 1):
            y_train_array = np.genfromtxt(
                origin_rank_fname.replace('.csv', '_' + str(i) + '.csv'),
                delimiter=',', dtype=float
            )
            y_train = np.zeros((self.feature.shape[0], 1), dtype=float)
            for i in xrange(self.feature.shape[0]):
                y_train[i][0] = y_train_array[i]
            y_val = np.zeros(y_train.shape, dtype=float)

            train_mask = np.zeros(self.feature.shape[0])
            val_mask = np.zeros(self.feature.shape[0])
            for i in xrange(self.feature.shape[0]):
                if abs(y_train[i][0]) < 1e-10:
                    y_val[i][0] = self.ground_truth[i]
                    val_mask[i] = 1.0
                else:
                    train_mask[i] = 1.0

            train_mask = np.array(train_mask, dtype=np.bool)
            val_mask = np.array(val_mask, dtype=np.bool)
            self.y_train_folds.append(copy.copy(y_train))
            self.y_val_folds.append(copy.copy(y_val))
            self.train_mask_folds.append(copy.copy(train_mask))
            self.val_mask_folds.append(copy.copy(val_mask))

        self.n = self.feature.shape[0]
        self.f_fold = []
        for i in range(1, fold_count + 1):
            self.f_fold.append(np.matrix(np.random.rand(self.n, 1)))

        # set flag parameters
        flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
        flags.DEFINE_integer('max_degree', 3,
                             'Maximum Chebyshev polynomial degree.')
        flags.DEFINE_integer('early_stopping', 100,
                             'Tolerance for early stopping (# of epochs).')
        flags.DEFINE_integer('epochs', self.parameters['epochs'],
                             'Number of epochs to train.')
        flags.DEFINE_float('learning_rate', self.parameters['learning_rate'],
                           'Initial learning rate.')
        flags.DEFINE_integer('hidden1', self.parameters['hidden1'],
                             'Number of units in hidden layer 1.')
        flags.DEFINE_integer('hidden2', 32,
                             'Number of units in hidden layer 2.')
        flags.DEFINE_float('dropout', self.parameters['dropout'],
                           'Dropout rate (1 - keep probability).')
        flags.DEFINE_float('weight_decay', self.parameters['weight_decay'],
                           'Weight for L2 loss on embedding matrix.')

    def _ranking(self, fold_index):
        if FLAGS.model == 'gcn':
            support = [self.adj]
            num_supports = 1
            model_func = GCN
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'support': [tf.placeholder(tf.float32) for _ in
                        range(num_supports)],
            'features': tf.placeholder(
                tf.float32, shape=(None, self.feature.shape[1])
            ),
            'labels': tf.placeholder(
                tf.float32, shape=(
                    None, self.y_train_folds[fold_index].shape[1]
                )
            ),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        model = model_func(placeholders, input_dim=self.feature.shape[1],
                           logging=True)

        # Initialize session
        sess = tf.Session()

        # Define model evaluation function
        def evaluate(features, support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support, labels,
                                                mask, placeholders)
            outs_val = sess.run(
                [model.loss, model.accuracy, model.outputs],
                feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test), \
                   outs_val[2]

        # Init variables
        sess.run(tf.global_variables_initializer())

        cost_val = []

        # Train model
        best_performance = {'tau': [-1.0, 7.0e-05], 'nmse': 1e20,
                            'rho': [-1.0, 9.2e-05]}
        best_generated_ranking = []
        best_epoch = 0
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(
                self.feature, support, self.y_train_folds[fold_index],
                self.train_mask_folds[fold_index], placeholders
            )
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy,
                             model.outputs, model.embeddings, model.vars],
                            feed_dict=feed_dict)

            # Validation
            cost, acc, duration, generated_ranking = evaluate(
                self.feature, support, self.y_val_folds[fold_index],
                self.val_mask_folds[fold_index], placeholders
            )
            cost_val.append(cost)
            current_performance = self.evaluator.single_fold_evaluate(
                generated_ranking, fold_index
            )
            is_better = self.evaluator.compare(current_performance,
                                               best_performance)

            # Print results
            if self.verbose:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
                      "{:.8f}".format(outs[1]),
                      "train_mse=", "{:.8f}".format(outs[2]), "val_loss=",
                      "{:.8f}".format(cost),
                      "val_mse=", "{:.8f}".format(acc), "time=",
                      "{:.4f}".format(time.time() - t))

            if is_better['tau']:
                best_performance = copy.copy(current_performance)
                if self.verbose:
                    print('better TAU performance:', current_performance)
                best_generated_ranking = copy.copy(generated_ranking)
                best_epoch = epoch

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
                    cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break
        if self.verbose:
            print('@epoch %d best performance:' % best_epoch, best_performance)
        if best_epoch + 1 == self.parameters['epochs']:
            print('not converge yet')
        print("Optimization Finished!")
        return best_generated_ranking

    def ranking(self):
        for i in xrange(self.fold_count):
            print('----------------------------------------')
            self.f_fold[i] = copy.copy(self._ranking(i))
        return self.f_fold

    def update_model(self, model_parameters):
        # need to construct new adjacency matrix
        if 'k' in model_parameters.keys() and \
                (not model_parameters['k'] == self.parameters['k']):
            # update the adjacency matrix
            if model_parameters['k'] > self.n:
                return False
            self.adj = calc_gcn_adjacency(self.feature, model_parameters['k'])
        for parameter_kv in model_parameters.iteritems():
            self.parameters[parameter_kv[0]] = parameter_kv[1]
            FLAGS.__setattr__(parameter_kv[0], parameter_kv[1])
        return True

if __name__ == '__main__':
    feature_path = os.path.join(sys.path[0], 'data', 'mvp')
    fold_count = 10
    parameter = {'epochs': 50, 'learning_rate': 0.001, 'weight_decay': 0,
                 'dropout': 0.1, 'hidden1': 256}
    ofname = 'gcn.pred'
    print(parameter)
    fold_evaluate = FoldEvaluator(
        os.path.join(feature_path, 'ground_truth.csv'), fold_count
    )
    gcn_rank = GCNRanking(
        os.path.join(feature_path, 'feature.csv'),
        os.path.join(feature_path, 'ground_truth.csv'), fold_count, parameter,
        fold_evaluate, verbose=False
    )
    generated_ranking = gcn_rank.ranking()
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
