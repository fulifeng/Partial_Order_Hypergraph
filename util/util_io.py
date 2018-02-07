# -*- coding: utf-8 -*-
import numpy as np
try:
    set
except NameError:
    from sets import Set as set
from sklearn import metrics


def calc_conv_laplacian(X, normalize=True):
    # # find data missing lines
    # miss_index = []
    # for i in range(X.shape[0]):
    #     if abs(np.amax(X[i]) + 1) < 1e-8:
    #         miss_index.append(i)
    # missings = len(miss_index)
    # print '#missing data:', missings
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    # for ind in miss_index:
    #     euc_dis[ind] = 0.0
    #     euc_dis[:, ind] = 0.0
    median = float(np.median(euc_dis))
    # As there are missing data, we cannot use the median function directly, as the real median is in the percentage position:
    #   50 + missing_percentage / 2
    # pos_real_median = 50 + 50.0 * (2.0 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0])
    # median = np.percentile(euc_dis, 50 + 50 * (2 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0]))
    infinity_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    infinity_matrix -= np.identity(X.shape[0])
    # calculate Laplacian matrix
    # assign 0 to missing lines
    # for ind in miss_index:
    #     infinity_matrix[ind] = 0.0
    #     infinity_matrix[:, ind] = 0.0
    if normalize:
        degree = np.sum(infinity_matrix, axis=0)
        for i in range(len(degree)):
            if abs(degree[i]) > 1e-8:
                degree[i] = 1.0 / degree[i]
        np.sqrt(degree, degree)
        d_neg_half_power = np.diag(degree)
        return np.identity(X.shape[0]) - np.dot(np.dot(d_neg_half_power, infinity_matrix), d_neg_half_power)
    else:
        # degree = np.diag(np.sum(infinity_matrix, axis=0))
        # return degree - infinity_matrix
        return np.diag(np.sum(infinity_matrix, axis=0)) - infinity_matrix


def calc_conv_laplacian_drop(X, k, normalize=True):
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    median = float(np.median(euc_dis))
    infinity_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    infinity_matrix -= np.identity(X.shape[0])
    # with this code, we drop the nodes after the one with the k-th biggest similarity value
    for i in range(infinity_matrix.shape[0]):
        pos_sort = np.argsort(infinity_matrix[i, :])
        for j in range(X.shape[0] - k):
            # here j + missings + 1 is because missing lines and the diagonal entry are assigned with 0.0
            infinity_matrix[pos_sort[j], i] = 0.0
            infinity_matrix[i, pos_sort[j]] = 0.0

    # calculate Laplacian matrix
    degree = np.sum(infinity_matrix, axis=0)
    for i in range(len(degree)):
        if abs(degree[i]) > 1e-8:
            degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    d_neg_half_power = np.diag(degree)
    if normalize:
        degree = np.sum(infinity_matrix, axis=0)
        for i in range(len(degree)):
            if abs(degree[i]) > 1e-8:
                degree[i] = 1.0 / degree[i]
        np.sqrt(degree, degree)
        d_neg_half_power = np.diag(degree)
        return np.identity(X.shape[0]) - np.dot(np.dot(d_neg_half_power, infinity_matrix), d_neg_half_power)
    else:
        degree = np.diag(np.sum(infinity_matrix, axis=0))
        return degree - infinity_matrix


'''
    The return is in ndarray type
'''
def calc_hyper_laplacian(X, k, normalize=True):
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    median = float(np.median(euc_dis))
    infinity_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    # infinity_matrix -= np.identity(X.shape[0])

    '''
        construct the matrix to calculate the Hypergraph Laplacian
        D_e is edge degree matrix
        D_v is vertex degree matrix
        H is the incidence matrix
        W is edge weight matrix
    '''
    # construct H
    H = np.zeros(infinity_matrix.shape, dtype=float)
    # H = np.identity(infinity_matrix.shape[0], dtype=float)
    for i in range(infinity_matrix.shape[0]):
        pos_sort = np.argsort(infinity_matrix[i, :])
        for j in range(k):
            H[pos_sort[infinity_matrix.shape[0] - j - 1], i] = 1.0

    # subtract identity here aims to include the note when construct an edge
    # with its (k-1)-nn neighbors
    infinity_matrix -= np.identity(X.shape[0])

    # construct W
    # this is element wise multiplication, won't work if H is not square
    W_neighbor = infinity_matrix * H
    # H_sum_raw = np.sum(H, axis=1)
    w = np.sum(W_neighbor, axis=0)
    W = np.diag(w)
    if normalize:
        # # construct D_e
        # D_e_inverse = np.identity(infinity_matrix.shape[0], dtype=float) * (1.0 / k)
        # for i in miss_index:
        #     D_e_inverse[i, i] = 0.0
        # construct D_v
        d_v = np.dot(H, w.T)
        # d_v = np.dot(w, H.T)
        for i in range(len(d_v)):
            if abs(d_v[i]) > 1e-8:
                d_v[i] = 1.0 / d_v[i]
        np.sqrt(d_v, d_v)
        D_v_neg_half_power = np.diag(d_v)
        # # calculate Laplacian matrix
        A = np.dot(H, W)
        # # A = np.dot(A, D_e_inverse)
        A = np.dot(A, H.T)
        # A = np.dot(np.dot(D_v_neg_half_power, A), D_v_neg_half_power)
        for i in range(X.shape[0]):
            A[i, i] = 0.0
        return np.identity(X.shape[0]) - np.dot(np.dot(D_v_neg_half_power, A),
                                                D_v_neg_half_power)
        # return np.identity(X.shape[0]) - np.dot(
        #                                     np.dot(
        #                                         # np.dot(
        #                                             np.dot(
        #                                                 np.dot(
        #                                                     D_v_neg_half_power,
        #                                                 H),
        #                                             W),
        #                                         # D_e_inverse),
        #                                     H.T),
        #                                 D_v_neg_half_power)
    else:
        S = np.dot(np.dot(H, W), H.T)
        SD = np.diag(np.sum(S, axis=0))
        Temp = SD - S
        return SD - S


'''
    The return is in ndarray type
'''
def calc_hyper_laplacian_similarity(X, k, normalize=True, return_H=False):
    # find data missing lines
    # miss_index = []
    # miss_index = set()
    # for i in range(X.shape[0]):
    #     if abs(np.amax(X[i]) + 1) < 1e-8:
    #         # miss_index.append(i)
    #         miss_index.add(i)
    # missings = len(miss_index)
    # print '#missing data:', missings
    # calculate median
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    # for ind in miss_index:
    #     euc_dis[ind] = 0.0
    #     euc_dis[:, ind] = 0.0
    median = float(np.median(euc_dis))
    # As there are missing data, we cannot use the median function directly, as the real median is in the percentage position:
    #   50 + missing_percentage / 2
    # pos_real_median = 50 + 50.0 * (2.0 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0])
    # median = np.percentile(euc_dis, 50 + 50 * (2 * missings * X.shape[0] - missings * missings) / (X.shape[0] * X.shape[0]))
    infinity_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    # infinity_matrix -= np.identity(X.shape[0])

    '''
        construct the matrix to calculate the Hypergraph Laplacian
        D_e is edge degree matrix
        D_v is vertex degree matrix
        H is the incidence matrix
        W is edge weight matrix
    '''
    # assign 0 to missing lines
    # for ind in miss_index:
    #     infinity_matrix[ind] = 0.0
    #     infinity_matrix[:, ind] = 0.0
    # construct H
    H = np.zeros(infinity_matrix.shape, dtype=float)
    # H = np.identity(infinity_matrix.shape[0], dtype=float)
    for i in range(infinity_matrix.shape[0]):
        # if i in miss_index:
        #     continue
        pos_sort = np.argsort(infinity_matrix[i, :])
        for j in range(k):
            H[pos_sort[infinity_matrix.shape[0] - j - 1], i] = 1.0

    # subtract identity here aims to include the note when construct an edge
    # with its (k-1)-nn neighbors
    infinity_matrix -= np.identity(X.shape[0])

    # construct W
    # this is element wise multiplication, won't work if H is not square
    # W_neighbor = infinity_matrix * H
    # w = np.sum(W_neighbor, axis=0)
    w = np.sum(infinity_matrix * H, axis=0) * (1.0 / (k ** 2))
    W = np.diag(w)
    if normalize:
        # # construct D_e
        # D_e_inverse = np.identity(infinity_matrix.shape[0], dtype=float) * (1.0 / k)
        # for i in miss_index:
        #     D_e_inverse[i, i] = 0.0
        # construct D_v
        d_v = np.dot(H, w.T)
        # d_v = np.dot(w, H.T)
        for i in range(len(d_v)):
            if abs(d_v[i]) > 1e-8:
                d_v[i] = 1.0 / d_v[i]
        np.sqrt(d_v, d_v)
        D_v_neg_half_power = np.diag(d_v)
        # # calculate Laplacian matrix
        A = np.dot(H, W)
        # # A = np.dot(A, D_e_inverse)
        A = np.dot(A, H.T)
        A = infinity_matrix * A
        # A = np.dot(np.dot(D_v_neg_half_power, A), D_v_neg_half_power)
        # for i in range(X.shape[0]):
        #     A[i, i] = 0.0
        if return_H:
            return np.identity(X.shape[0]) - np.dot(np.dot(D_v_neg_half_power,
                                                           A),
                                                    D_v_neg_half_power), H
        else:
            return np.identity(X.shape[0]) - np.dot(np.dot(D_v_neg_half_power,
                                                           A),
                                                    D_v_neg_half_power)
        # return np.identity(X.shape[0]) - np.dot(
        #                                     np.dot(
        #                                         # np.dot(
        #                                             np.dot(
        #                                                 np.dot(
        #                                                     D_v_neg_half_power,
        #                                                 H),
        #                                             W),
        #                                         # D_e_inverse),
        #                                     H.T),
        #                                 D_v_neg_half_power)
    else:
        S = np.dot(np.dot(H, W), H.T)
        S = infinity_matrix * S
        SD = np.diag(np.sum(S, axis=0))
        if return_H:
            return SD - S, H
        else:
            return SD - S


'''
    'H' denotes the incidence matrix of conventional hypergraph.
    'partial_orders' are several lists, each list contains 'n' numbers
        corresponding to 'n' samples, whether two samples satisfy a partial
        order depends on the magnitude of their numbers.
    'order_directions' are binary values (True/False). The order direction of a
        partial order is True if it has same direction as target ranking, i.e.,
        if f_i > f_j, po_i > po_j is expected.
'''
# def calc_partial_order_incidence(partial_orders, drop_ratios, order_directions,
#                                  H, place_holder=12345678.12345678):
#     assert len(partial_orders) == len(drop_ratios), 'count of partial orders' \
#                                                     'and drop ratio not equal'
#     assert len(order_directions) == len(drop_ratios), 'count of order ' \
#                                                       'directions and drop ' \
#                                                       'ratios not equal'
#     print '#partial orders:', len(drop_ratios)
#     print '#samples:', H.shape[0]
#     n = H.shape[0]
#     partial_order_incidences = []
#     C = np.dot(H, H.T)      # C is the cooccurrence matrix
#     for partial_order_index, partial_order in enumerate(partial_orders):
#         drop_ratio = drop_ratios[partial_order_index]
#
#         order_direction = order_directions[partial_order_index]
#         partial_order_matrix = np.zeros(C.shape, dtype=float)
#         satisfied_pairs = 0
#         for i in range(n):
#             if abs(partial_order[i] - place_holder) < 1e-8:
#                 continue
#             for j in range(i + 1, n):
#                 '''
#                     Example of the construction rule:
#                         sample_i better than sample_j, f_i < f_j, po_i > po_j,
#                         thus the 'order_direction' is False
#                         partial_order_matrix_ji should be 1
#                 '''
#                 if abs(partial_order[j] - place_holder) < 1e-8:
#                     continue
#                 if order_direction:
#                     if partial_order[i] > partial_order[j]:
#                         partial_order_matrix[i][j] = partial_order[i] \
#                                                      - partial_order[j]
#                     elif partial_order[i] < partial_order[j]:
#                         partial_order_matrix[j][i] = partial_order[j] \
#                                                      - partial_order[i]
#                 else:
#                     if partial_order[i] < partial_order[j]:
#                         partial_order_matrix[i][j] = partial_order[j] \
#                                                      - partial_order[i]
#                     elif partial_order[i] > partial_order[j]:
#                         partial_order_matrix[j][i] = partial_order[i] \
#                                                      - partial_order[j]
#                 satisfied_pairs += 1
#         print '#satisfied pairs:', satisfied_pairs
#         cat_percentage = (n ** 2 - satisfied_pairs * (1.0 - drop_ratio)) \
#                          * 100 / (n ** 2)
#         cat_percentile = np.percentile(partial_order_matrix, cat_percentage)
#         print 'cat percentile: %.6f at percentage: %.3f' % (cat_percentile,
#                                                             cat_percentage)
#         for i in range(n):
#             for j in range(n):
#                 if partial_order_matrix[i][j] > cat_percentile:
#                     partial_order_matrix[i][j] = 1.0
#                 else:
#                     partial_order_matrix[i][j] = 0.0
#         partial_order_matrix *= C
#         partial_order_incidences.append(copy.copy(partial_order_matrix))
#     return partial_order_incidences
def calc_partial_order_incidence(partial_order, drop_ratio, order_direction,
                                 H, place_holder=12345678.12345678):
    print '#samples:', H.shape[0]
    n = H.shape[0]
    C = np.dot(H, H.T)      # C is the cooccurrence matrix
    positive_pairs = 0
    for i in xrange(n):
        for j in xrange(n):
            if C[i][j] > 1e-8:
                positive_pairs += 1
    print 'cooccure pairs in hypergraph:', positive_pairs
    partial_order_matrix = np.zeros(C.shape, dtype=float)
    satisfied_pairs = 0
    for i in range(n):
        if abs(partial_order[i] - place_holder) < 1e-8:
            continue
        for j in range(i + 1, n):
            '''
                Example of the construction rule:
                    sample_i better than sample_j, f_i < f_j, po_i > po_j,
                    thus the 'order_direction' is False
                    partial_order_matrix_ji should be 1
            '''
            if abs(partial_order[j] - place_holder) < 1e-8:
                continue
            if order_direction:
                if partial_order[i] > partial_order[j]:
                    partial_order_matrix[i][j] = partial_order[i] \
                                                 - partial_order[j]
                elif partial_order[i] < partial_order[j]:
                    partial_order_matrix[j][i] = partial_order[j] \
                                                 - partial_order[i]
            else:
                if partial_order[i] < partial_order[j]:
                    partial_order_matrix[i][j] = partial_order[j] \
                                                 - partial_order[i]
                elif partial_order[i] > partial_order[j]:
                    partial_order_matrix[j][i] = partial_order[i] \
                                                 - partial_order[j]
            satisfied_pairs += 1
    print '#satisfied pairs:', satisfied_pairs
    # print 'maximum: %.6f minimum: %.6f' % (np.max(partial_order_matrix),
    #                                        np.min(partial_order_matrix))
    # for i in xrange(1000):
    #     print i, np.percentile(partial_order_matrix, i * 0.1)
    cat_percentage = (n ** 2 - satisfied_pairs * (1.0 - drop_ratio)) \
                     * 100 / (n ** 2)
    cat_percentile = np.percentile(partial_order_matrix, cat_percentage)
    print 'cat percentile: %.6f at percentage: %.3f' % (cat_percentile,
                                                        cat_percentage)
    for i in xrange(n):
        for j in xrange(n):
            if partial_order_matrix[i][j] > cat_percentile:
                partial_order_matrix[i][j] = 1.0
            else:
                partial_order_matrix[i][j] = 0.0
            # if partial_order_matrix[i][j] < cat_percentile:
            #     partial_order_matrix[i][j] = 0.0
    partial_order_matrix *= C
    positive_pairs = 0
    for i in xrange(n):
        for j in xrange(n):
            if partial_order_matrix[i][j] > 1e-8:
                positive_pairs += 1
    print 'pairs after merging with hypergraph:', positive_pairs
    return partial_order_matrix


def calc_gcn_adjacency(X, k):
    euc_dis = metrics.pairwise.euclidean_distances(X, X)
    median = float(np.median(euc_dis))
    adjacency_matrix = metrics.pairwise.rbf_kernel(X, gamma= 0.5 / (median ** 2))
    # with this code, we drop the nodes after the one with the k-th biggest
    # similarity value
    for i in range(adjacency_matrix.shape[0]):
        pos_sort = np.argsort(adjacency_matrix[i, :])
        for j in range(X.shape[0] - k):
            # entry are assigned with 0.0
            adjacency_matrix[pos_sort[j], i] = 0.0
            adjacency_matrix[i, pos_sort[j]] = 0.0

    degree = np.sum(adjacency_matrix, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    d_neg_half_power = np.diag(degree)
    return np.dot(np.dot(d_neg_half_power, adjacency_matrix), d_neg_half_power)


def cur_print_performance(performance):
    print 'mae:\t', performance['mae']
    print 'tau:\t', performance['tau'][0]
    print 'rho:\t', performance['rho'][0]


def mvp_print_performance(performance):
    print 'nmse:\t', performance['nmse']
    print 'tau:\t', performance['tau'][0]
    print 'rho:\t', performance['rho'][0]