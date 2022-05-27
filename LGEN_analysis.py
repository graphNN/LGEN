
from LGEN_utils import load_data_citation
from load_geom import load_geom
import dgl

import numpy as np
import matplotlib.pyplot as plt


def data_analysis():
    dataset = 'texas'  # texas, cornell, wisconsin, chameleon, film, squirrel, cora, citeseer, corafull, coauthorCS
    if dataset in {'texas', 'cornell', 'wisconsin', 'chameleon', 'film', 'squirrel'}:
        dataset_split = f'/diskvdb/rui/mycode/duanrui_0110/second/geom-gcn-master/geom-gcn-master/splits/{dataset}_split_0.6_0.2_{0}.npz'
        g, features, labels, train_mask, val_mask, test_mask = load_geom(dataset, dataset_split, train_percentage=None,
                                                                         val_percentage=None,
                                                                         embedding_mode=None, embedding_method=None,
                                                                         embedding_method_graph=None,
                                                                         embedding_method_space=None)
    if dataset in {'cora', 'citeseer', 'pubmed'}:
        A, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(dataset)
        g = dgl.from_scipy(adj)
        g = dgl.remove_self_loop(g)

    src, dst = g.edges()

    print('classes and features:', int(labels.max()) + 1, features.shape[1])
    print('train, val and test:', len(train_mask), len(val_mask), len(test_mask))
    print('nodes:', len(g.nodes()))
    print('edges:', len(src))
    h = 0
    for i in range(len(src)):
        if labels[src[i]] == labels[dst[i]]:
            h = h + 1
    print('h:', h / len(src))


def curve(x, y, name):
    plt.subplot(111)
    plt.grid(linestyle="-")
    for i in range(len(name)):
        plt.plot(x, y[i], label=name[i])  # dataset
        plt.scatter(x, y[i], s=20)
    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x)
    plt.yticks(np.arange(20, 90, step=10))
    plt.legend(loc=3, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10, fontweight='bold', style='normal')  # bold
    plt.show()


# parameter: p
# x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# y2 = [43.57, 43.51, 46.40, 49.34, 52.26, 55.50, 57.43, 59.74, 62.52, 64.30, 66.86]  # chameleon
# y4 = [24.91, 27.64, 49.16, 46.03, 48.99, 56.99, 61.35, 66.14, 71.50, 73.48, 76.22]  # citeseer
# y3 = [58.11, 60.05, 64.06, 68.39, 73.15, 77.01, 79.99, 82.12, 83.42, 84.86, 85.76]  # cora
# y1 = [22.74, 29.09, 30.00, 33.23, 35.27, 37.71, 41.65, 45.43, 47.87, 50.52, 54.22]  # squirrel
# y0 = [84.86, 72.70, 61.62, 56.76, 55.68, 55.68, 52.70, 55.14, 55.41, 49.73, 49.73]  # cornell


# parameter: C
x = [1, 2, 3, 4, 5, 6, 7, 8]
y2 = [66.86, 65.26, 62.52, 60.35, 58.16, 55.22, 53.00, 50.15]  # chameleon
y4 = [64.18, 70.03, 76.22, 73.78, 73.45, 47.05, 37.34, 27.22]  # citeseer
y3 = [85.20, 85.76, 85.68, 85.58, 85.29, 85.31, 85.26, 84.87]  # cora
y1 = [54.22, 49.64, 33.09, 30.29, 25.10, 19.96, 19.88, 19.84]  # squirrel
y0 = [84.86, 84.05, 82.43, 80.81, 79.73, 81.08, 78.65, 79.73]  # cornell


name = ['Cornell', 'Squirrel', 'Chameleon', 'Cora', 'Citesee']
y = []
y.append(y0)
y.append(y1)
y.append(y2)
y.append(y3)
y.append(y4)

curve(x, y, name)


