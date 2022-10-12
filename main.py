import time
import scipy.sparse
import argparse
import numpy as np

from model4both import Node2vecModel
from model4undirect import Node2vecModel as UndirectModel

import torch
import dgl

def UndirectGraph():
    row = [0,1,1,2,0,2]
    col = [1,0,2,1,2,0]
    graph = dgl.graph((row,col))
    graph.edata['w'] = torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0]).float()
    return graph

def UndirectGraph1():
    row = [1,2,2,3,3,1]
    col = [2,1,3,2,1,3]
    graph = dgl.graph((row,col))
    graph.edata['w'] = torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0]).float()
    return graph

def DirectGraph():
    row = torch.tensor([0,1,2])
    col = torch.tensor([1,2,3])
    row += 1
    col += 1

    graph = dgl.graph((row,col))
    graph.edata['w'] = torch.tensor([1.0,2.0,3.0]).float()
    return graph

def Test(graph):
    # config
    #===================================================
    embedding_dim = 3
    walk_length = 5
    p = 0.25
    q = 4.0
    num_walks = 1
    epochs = 1
    batch_size = 2
    device = 'cpu'
    #===================================================

    trainer = Node2vecModel(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        p=p,
        q=q,
        num_walks=num_walks,
        weight_name = 'w',
        device=device,
    )

    trainer.train(
        epochs=epochs, batch_size=batch_size, learning_rate=0.01
    )

    node_features = trainer.embedding()
    nodes = node_features.detach().to('cpu').numpy()
    print(nodes)

def TestUndirect(graph):
    # config
    #===================================================
    embedding_dim = 3
    walk_length = 5
    p = 0.25
    q = 4.0
    num_walks = 1
    epochs = 1
    batch_size = 2
    device = 'cpu'
    #===================================================

    trainer = UndirectModel(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        p=p,
        q=q,
        num_walks=num_walks,
        weight_name = 'w',
        device=device,
    )

    trainer.train(
        epochs=epochs, batch_size=batch_size, learning_rate=0.01
    )

    node_features = trainer.embedding()
    nodes = node_features.detach().to('cpu').numpy()
    print(nodes)

if __name__ == "__main__":
    g1 = DirectGraph()
    g2 = UndirectGraph()
    g3 = UndirectGraph1()

    print('=============================================================================')
    print('test model4both.py')
    print('========================')
    print('directed graph, node id begin with 1')
    Test(g1)
    print('========================')
    print('undirected graph, node id begin with 0. This usage is wrong!!!, node id must begin with 1 ')
    Test(g2)
    print('========================')
    print('undirected graph, node id begin with 1')
    Test(g3)

    print('=============================================================================')
    print('test model4undirect.py')
    print('========================')
    print('undirected graph, node id begin with 0')
    TestUndirect(g2)
    print('========================')
    print('undirected graph, node id begin with 1, wrong')
    # TestUndirect(g3)
