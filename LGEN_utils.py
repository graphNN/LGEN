import dgl
import torch
import scipy.sparse as sp
import numpy as np

import sys
import pickle as pkl
import networkx as nx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def cate_grouping(labels):
    group = {}
    num_classes = labels.max().int() + 1
    for i in range(num_classes):
        group[i] = torch.nonzero(labels == i, as_tuple=True)[0]

    return group


def category_features(features, labels, mask):
    c = int(labels.max() + 1)
    group = cate_grouping(labels[mask[0]])

    cate_features_ = torch.zeros(c, features.shape[1]).to(features.device)

    for i in range(c):
        if len(group[i]) == 0:
            cate_features_[i] = features[mask[0]].mean(0)
        else:
            cate_features_[i] = features[mask[0]][group[i]].mean(0)

    return cate_features_


def features_augmentation(features, labels, mask):
    cate_features_ = category_features(features, labels, mask)
    cont = torch.zeros(features.shape).to(features.device)
    for i in range(features.shape[0]):
        if i in mask[2]:
            cont[i] = features[i] * 1
        elif i in mask[1]:
            cont[i] = cate_features_[labels[i]]

        elif i in mask[0]:
            cont[i] = cate_features_[labels[i]]
        else:
            cont[i] = features[i] * 1
    return cont


def edge_rand_prop(g, edge_drop_rate):
    src, dst = g.edges()
    edges_num = len(src)

    drop_rates = torch.FloatTensor(np.ones(edges_num) * edge_drop_rate)
    masks = torch.tensor(torch.bernoulli(1. - drop_rates), dtype=torch.int).to(g.device)

    remove_edges_index = torch.nonzero(masks == 0, as_tuple=True)
    new_g = dgl.remove_edges(g, remove_edges_index[0])

    return new_g


def propagate_adj(adj):
    D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)
    A = sparse_mx_to_torch_sparse_tensor(A)

    return A


def consis_loss(logps, tem, lam):
    logps = torch.exp(logps)
    sharp_logps = (torch.pow(logps, 1. / tem) / torch.sum(torch.pow(logps, 1. / tem), dim=1, keepdim=True)).detach()
    loss = torch.mean((logps - sharp_logps).pow(2).sum(1)) * lam

    return loss


file_dir_citation = "/diskvdb/rui/mycode/duanrui_0110/three_distance/data"
# file_dir_citation = 'LGEN/data'
def load_data_citation(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(file_dir_citation, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    features = torch.FloatTensor(np.array(features.todense()))
    # features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features, labels, idx_train, idx_val, idx_test, adj

