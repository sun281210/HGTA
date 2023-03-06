import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import scipy
import dgl

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):#归一化
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
#    rowsum = np.array(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""#将scipy稀疏矩阵转换为torch稀疏张量
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_ACM(ratio, type_num):
    path = "../data/ACM/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "features_0.npz").astype(float)
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = np.load(path + "pap.npy")
    a = sparse.csr_matrix(pap)
    a = dgl.from_scipy(a)
    psp = np.load(path + "psp.npy")
    s = sparse.csr_matrix(psp)
    s = dgl.from_scipy(s)
    a = dgl.add_self_loop(a)
    s = dgl.add_self_loop(s)
    pos = sp.load_npz(path + "pos_acm2.npz")
    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))#
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    adjM = scipy.sparse.load_npz(path + '/adjM.npz').toarray()#这里加载的是原始
    features_0=np.eye(4019)
    features_1=np.eye(7167)
    features_2=np.eye(60)
    labels = np.load(path + '/labels.npy')
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test,adjM,[features_0, features_1, features_2],labels,[a, s]


def load_DBLP(ratio, type_num):
    path = "../data/DBLP/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "features_0.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = np.load(path + "apa.npy")
    p = sparse.csr_matrix(apa)
    p = dgl.from_scipy(p)
    apcpa = np.load(path + "apcpa.npy")
    c = sparse.csr_matrix(apcpa)
    c = dgl.from_scipy(c)
    aptpa = np.load(path + "aptpa.npy")
    t = sparse.csr_matrix(aptpa)
    t = dgl.from_scipy(t)
    p = dgl.add_self_loop(p)
    c= dgl.add_self_loop(c)
    t = dgl.add_self_loop(t)
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    adjM = scipy.sparse.load_npz(path + '/adjM.npz').toarray()#这里加载的是原始图
    features_a=np.eye(4057)
    features_p=np.eye(14328)
    features_s = np.eye(7723)
    features_t = np.eye(20)
    labels = np.load(path + '/labels.npy')
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test,adjM,[features_a, features_p,features_s,features_t], labels,[p, c, t]

def load_freebase(ratio, type_num):
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    mam = sp.load_npz(path + "mam.npz")
    mams = sparse.csr_matrix(mam)
    a = dgl.from_scipy(mams)
    mdm = sp.load_npz(path + "mdm.npz")
    mdms = sparse.csr_matrix(mdm)
    d = dgl.from_scipy(mdms)
    mwm = sp.load_npz(path + "mwm.npz")
    mwms = sparse.csr_matrix(mwm)
    w = dgl.from_scipy(mwms)
    a = dgl.add_self_loop(a)
    d = dgl.add_self_loop(d)
    w = dgl.add_self_loop(w)
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    features_0=np.eye(3492)
    features_1=np.eye(2502)
    features_2=np.eye(33401)
    features_3=np.eye(4459)
    labels = np.load(path + "labels.npy").astype('int32')
    adjM = scipy.sparse.load_npz(path + '/adjM.npz').toarray()#这里加载的是原始图

    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test,adjM,[features_0, features_1, features_2,features_3],labels[a,d,w]

def load_data(dataset, ratio,type_num):
    if dataset == "ACM":
        data = load_ACM(ratio, type_num)
    elif dataset == "DBLP":
        data = load_DBLP(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
    return data
