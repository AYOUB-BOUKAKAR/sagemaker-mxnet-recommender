"""M4 - modele de factorisation matricielle (MXNet Gluon)."""
import numpy as np
import mxnet as mx
from mxnet import gluon, nd


class MF(gluon.HybridBlock):
    # note(u,i) = mean + b_u + b_i + p_u . q_i
    def __init__(self, n_users, n_items, k=32, mean=3.5, drop=0.0, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.mean = mean
        with self.name_scope():
            self.P = gluon.nn.Embedding(n_users, k)
            self.Q = gluon.nn.Embedding(n_items, k)
            self.bu = gluon.nn.Embedding(n_users, 1)
            self.bi = gluon.nn.Embedding(n_items, 1)
            self.drop = gluon.nn.Dropout(drop)

    def hybrid_forward(self, F, u, i):
        p = self.drop(self.P(u))
        q = self.drop(self.Q(i))
        dot = F.sum(p * q, axis=1)
        b = F.reshape(self.bu(u), (-1,)) + F.reshape(self.bi(i), (-1,))
        return dot + b + self.mean


def score_matrix(net, n_users, n_items):
    # toutes les notes predites (n_users x n_items), pour le classement
    P = net.P.weight.data().asnumpy()
    Q = net.Q.weight.data().asnumpy()
    bu = net.bu.weight.data().asnumpy()
    bi = net.bi.weight.data().asnumpy()
    return P @ Q.T + bu + bi.T + net.mean