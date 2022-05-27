
import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import random
import time
import argparse
import optuna

from LGEN_models import LGEN
from LGEN_utils import edge_rand_prop, consis_loss, propagate_adj
from LGEN_utils import features_augmentation, load_data_citation


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', help='cora citeseer')

parser.add_argument('--num_layers', type=int, default=2, help='')
parser.add_argument('--edge_sample', type=float, default=1, help='edge sampling rate.')
parser.add_argument('--num_graphs', type=int, default=3, help='')
parser.add_argument('--input_dropout', type=float, default=0.4,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_dropout', type=float, default=0.7,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--tem', type=float, default=0.1, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.5, help='Lamda')
parser.add_argument('--a', type=float, default=0.1, help='LGEN alpha')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--feat_aug', type=bool, default=True)
parser.add_argument('--only_aug', type=bool, default=False)

parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')

args = parser.parse_args()

device = torch.device("cuda:0")

A, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(args.dataset)
g = dgl.from_scipy(adj).to(device)
features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)
mask = [train_mask, val_mask, test_mask]

if args.feat_aug:
    cont = features_augmentation(features, labels, mask)
    if args.only_aug:
        features = cont
    else:
        features = torch.cat([features, cont], dim=1)


def test_cat(trial):

    # edge_dropout = trial.suggest_int('edge_dropout', 0, 4)
    # input_dropout = trial.suggest_int('input_dropout', 3, 9)
    # hidden_dropout = trial.suggest_int('hidden_dropout', 3, 9)
    # num_graphs = trial.suggest_int('num_graphs', 2, 4)
    #
    # tem = trial.suggest_int('tem', 1, 5)
    # lam = trial.suggest_int('lam', 5, 15)
    # alpha = trial.suggest_int('alpha', 0, 5)

    # hidden = trial.suggest_int('hidden', 3, 6)
    # num_layers = trial.suggest_int('num_layers', 2, 6)

    hidden = args.hidden

    edge_dropout = 1 - args.edge_sample
    input_dropout = args.input_dropout
    hidden_dropout = args.hidden_dropout
    num_graphs = args.num_graphs
    tem = args.tem
    lam = args.lam
    alpha = args.a
    num_layers = args.num_layers

    test_accs = []
    for i in range(args.seed, args.seed + 10):
        # seed
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        adj_list = []
        feat_list = []

        for j in range(num_graphs):
            new_g = edge_rand_prop(dgl.remove_self_loop(g), edge_dropout)
            new_g = dgl.add_self_loop(new_g)
            new_adj = propagate_adj(new_g.adj(scipy_fmt='csr'))
            adj_list.append(new_adj.to(device))
        feat_list.append(features)

        model = LGEN(input_dim=features.shape[1], hidden=hidden, classes=(int(labels.max()) + 1),
                       num_graphs=num_graphs,
                       dropout=[input_dropout, hidden_dropout],
                       num_layers=num_layers, activation=True, alpha=alpha, use_bn=False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.to(device)

        t0 = time.time()
        best_acc, best_val_acc, best_test_acc, best_val_loss = 0, 0, 0, float("inf")
        for epoch in range(args.epochs):
            model.train()
            t1 = time.time()
            outputs = model(adj_list, feat_list)
            outputs_ = F.log_softmax(outputs, dim=1)
            train_loss = F.cross_entropy(outputs_[train_mask], labels[train_mask])

            cosi_loss = consis_loss(outputs_, tem, lam)

            optimizer.zero_grad()
            # train_loss.backward()
            (train_loss + cosi_loss).backward()
            optimizer.step()

            model.eval()  # val
            with torch.no_grad():
                outputs = model(adj_list, feat_list)
                outputs_ = F.log_softmax(outputs, dim=1)

                train_loss_ = F.cross_entropy(outputs_[train_mask], labels[train_mask]).item()
                train_pred = outputs_[train_mask].max(dim=1)[1].type_as(labels[train_mask])
                train_correct = train_pred.eq(labels[train_mask]).double()
                train_correct = train_correct.sum()
                train_acc = (train_correct / len(labels[train_mask])) * 100

                val_loss = F.cross_entropy(outputs[val_mask], labels[val_mask]).item()
                val_pred = outputs[val_mask].max(dim=1)[1].type_as(labels[val_mask])
                correct = val_pred.eq(labels[val_mask]).double()
                correct = correct.sum()
                val_acc = (correct / len(labels[val_mask])) * 100

            model.eval()  # test
            with torch.no_grad():
                # outputs = model(g_list, features, order_attn)
                # outputs = F.log_softmax(outputs, dim=1)
                test_loss = F.cross_entropy(outputs_[test_mask], labels[test_mask]).item()
                test_pred = outputs_[test_mask].max(dim=1)[1].type_as(labels[test_mask])
                correct = test_pred.eq(labels[test_mask]).double()
                correct = correct.sum()
                test_acc = (correct / len(labels[test_mask])) * 100

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_test_acc = test_acc
                    bad_epoch = 0

                else:
                    bad_epoch += 1

            epoch_time = time.time() - t1
            if (epoch + 1) % 50 == 0:
                print('Epoch: {:3d}'.format(epoch), 'Train loss: {:.4f}'.format(train_loss_),
                      '|Train accuracy: {:.2f}%'.format(train_acc), '||Val loss: {:.4f}'.format(val_loss),
                      '||Val accuracy: {:.2f}%'.format(val_acc), '||Time: {:.2f}'.format(epoch_time))

            if bad_epoch == args.patience:
                break

        _time = time.time() - t0
        print('\n', 'Test accuracy:', best_test_acc)
        print('Time of training model:', _time)
        print('End of the training !')
        print('-' * 100)

        test_accs.append(best_test_acc.item())

    print(test_accs)
    print(f'Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}')

    return np.mean(test_accs)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(test_cat, n_trials=1)  # 搜索次数

    # print("Number of finished trials: ", len(study.trials))
    #
    # print("Best trial:")
    # trial = study.best_trial
    #
    # print("  Value: ", trial.value)
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

