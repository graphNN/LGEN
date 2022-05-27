
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
from load_geom import load_geom
from LGEN_utils import features_augmentation


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='wisconsin', help='texas, cornell, wisconsin, chameleon')

parser.add_argument('--num_layers', type=int, default=2, help='')
parser.add_argument('--edge_sample', type=float, default=0, help='edge sampling rate.')
parser.add_argument('--num_graphs', type=int, default=2, help='')  # cora:4
parser.add_argument('--input_dropout', type=float, default=0.6,  # cora:0.6  webkb:0.5/0.5
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_dropout', type=float, default=0.4,  # cora:0.6  pub:0.8
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--tem', type=float, default=0.2, help='Sharpening temperature')  # cora:0.2
parser.add_argument('--lam', type=float, default=0., help='Lamda')  # cora:0.5
parser.add_argument('--a', type=float, default=1, help='LGEN alpha')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--feat_aug', type=bool, default=True)
parser.add_argument('--only_aug', type=bool, default=False)

parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')  # 0.005
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')  # 5e-4

parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')  # 64
parser.add_argument('--gin_hidden', type=int, default=8, help='MoNet:Number of hidden units.')
parser.add_argument('--appnp_hidden', type=int, default=8, help='APPNP:Number of hidden units.')
parser.add_argument('--num_heads1', type=int, default=8, help='GAT:number of head')
parser.add_argument('--num_heads2', type=int, default=1, help='GAT:number of head')

parser.add_argument('--gin_agg_type', type=str, default='mean', help='gin:sum mean max')
parser.add_argument('--sage_agg_type', type=str, default='gcn', help='sage:mean, gcn, pool, lstm')
parser.add_argument('--K', type=int, default=10, help='APPNP inter')  # paper:10
parser.add_argument('--alpha', type=float, default=0.1, help='APPNP alpha')

args = parser.parse_args()

device = torch.device("cuda:0")

def test_cat(trial):

    # edge_dropout = trial.suggest_int('edge_dropout', 0, 5)
    # input_dropout = trial.suggest_int('input_dropout', 0, 7)
    # hidden_dropout = trial.suggest_int('hidden_dropout', 0, 7)
    # num_graphs = trial.suggest_int('num_graphs', 1, 4)  # 1?
    # tem = trial.suggest_int('tem', 1, 5)
    # lam = trial.suggest_int('lam', 0, 10)
    # hidden = trial.suggest_int('hidden', 4, 7)
    # alpha = trial.suggest_int('alpha', 0, 10)
    # num_layers = trial.suggest_int('num_layers', 2, 4)
    # need_feat_aug = trial.suggest_int('need_feat_aug', 0, 1)

    hidden = args.hidden

    edge_dropout = 1 - args.edge_sample
    input_dropout = args.input_dropout
    hidden_dropout = args.hidden_dropout
    num_graphs = args.num_graphs
    tem = args.tem
    lam = args.lam
    alpha = args.a
    num_layers = args.num_layers

    need_feat_aug = args.feat_aug

    if need_feat_aug:
        feat_aug = True
    else:
        feat_aug = False

    all_test_accs = []
    for j in range(10):
        dataset_split = f'/diskvdb/rui/mycode/duanrui_0110/second/geom-gcn-master/geom-gcn-master/splits/{args.dataset}_split_0.6_0.2_{j}.npz'
        g, features, labels, train_mask, val_mask, test_mask = load_geom(args.dataset, dataset_split,
                                                                         train_percentage=None, val_percentage=None,
                                                                         embedding_mode=None, embedding_method=None,
                                                                         embedding_method_graph=None,
                                                                         embedding_method_space=None)

        g = dgl.add_self_loop(g).to(device)
        features = torch.tensor(features).to(device)
        labels = torch.tensor(labels).to(device)
        train_mask = torch.tensor(train_mask).to(device)
        val_mask = torch.tensor(val_mask).to(device)
        test_mask = torch.tensor(test_mask).to(device)
        mask = [train_mask, val_mask, test_mask]

        if feat_aug:
            cont = features_augmentation(features, labels, mask)
            if args.only_aug:
                features = cont
            else:
                features = torch.cat([features, cont], dim=1)

        test_accs = []
        for i in range(args.seed, args.seed + 5):
            # seed
            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)

            feats = []
            adj_list = []
            for i in range(num_graphs):
                new_g = edge_rand_prop(dgl.remove_self_loop(g), edge_dropout)
                new_g = dgl.add_self_loop(new_g)
                new_adj = propagate_adj(new_g.adj(scipy_fmt='csr'))
                adj_list.append(new_adj.to(device))
            feats.append(features)

            model = LGEN(input_dim=features.shape[1], hidden=hidden, classes=(int(labels.max()) + 1),
                        num_graphs=num_graphs,
                        dropout=[input_dropout, hidden_dropout],
                        num_layers=num_layers, alpha=alpha, activation=True, use_bn=False)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model.to(device)

            t0 = time.time()
            best_acc, best_val_acc, best_test_acc, best_val_loss = 0, 0, 0, float("inf")
            for epoch in range(args.epochs):
                model.train()
                t1 = time.time()
                outputs = model(adj_list, feats)
                outputs_ = F.log_softmax(outputs, dim=1)
                train_loss = F.cross_entropy(outputs_[train_mask], labels[train_mask])

                cosi_loss = consis_loss(outputs_, tem, lam)

                optimizer.zero_grad()
                # train_loss.backward()
                (train_loss + cosi_loss).backward()
                optimizer.step()

                model.eval()  # val
                with torch.no_grad():
                    outputs = model(adj_list, feats)
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
        all_test_accs.append(np.mean(test_accs))
        print(test_accs)
        print(f'Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}')
        print('-' * 100)
    print(all_test_accs)
    print(f'all Average test accuracy: {np.mean(all_test_accs)} ± {np.std(all_test_accs)}')
    return np.mean(all_test_accs)


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

