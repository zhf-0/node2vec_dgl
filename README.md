# node2vec_dgl
`node2vec` for directed and undirected graphs based on `DGL`. The program in [/example/pytorch/node2vec](https://github.com/dmlc/dgl/tree/master/examples/pytorch/node2vec) can't deal with directed graphs, detail can be found in [dgl issue #4696](https://github.com/dmlc/dgl/issues/4696)

- `model4undirect.py`: `node2vec` program for undirected graphs which is almost the same as  [examples/pytorch/node2vec/model.py](https://github.com/dmlc/dgl/blob/master/examples/pytorch/node2vec/model.py) but remove `eval` related parameters and functions
- `model4both.py`: `node2vec` program for directed and undirected graphs
- `main.py`: basic test function



## `model4undirect.py` 

If you want to use `model4undirect.py` for undirected graphs, the node id must begin with 0, otherwise the program will fail because of  the same error in the issue 4696 "IndexError: index out of range in self" (you can reproduce the error by uncommenting line 124 in `main.py`). The reason is that `node2vec`  walk will begin with every node in the graph. When the walk begin with node id 0, there will be no successor and next node id in `trace` tensor is -1, which is a wrong index for `self.embedding` . 

Besides, the same error will occur again when node id is inconsecutive. For example, there are edges between node `0-1`, `1-2`, `4-5`, but no edge connected to node 3. When `node2vec` walk begin with node 3, there will be no successor.

## `model4both.py`

If you want to use `model4both.py`, the node id must begin with 1, since node is 0 is reserved as padding index. Other than that, there is no limitation for node id, the node id can be inconsecutive as well.
