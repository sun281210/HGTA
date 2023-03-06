import numpy
import torch
from utils import load_data, set_params, evaluate
from module import HGTA
import warnings
import datetime
import pickle as pkl
import os
import random
import dgl
from scipy import sparse
import torch.nn.functional as F
import numpy as np



warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset
negative_slope=args.slope
residual=args.res
num_hidden=args.hidden_dim

num_classes=3

activation = F.elu
num_heads=8
out_dim = 3
svm_macro_avg = np.zeros((7,), dtype=np.float)
svm_micro_avg = np.zeros((7,), dtype=np.float)
nmi_avg = 0
ari_avg = 0


## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train():
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test, adjM,features,labels,gs = \
            load_data(args.dataset,args.ratio, args.type_num)

    features_list = [torch.FloatTensor(feature).to(device) for feature in features]# 遍历特征列表，就是3种节点的特征
    onehot_feature_list = [torch.FloatTensor(feature).to(device) for feature in features]

    in_dims_1 = [features.shape[0] for features in onehot_feature_list]
    print(in_dims_1)
    for i in range(0, len(onehot_feature_list)):
        dim = onehot_feature_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        onehot_feature_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    # 加载node_type_feature
    node_type_feature = [[0 for c in range(1)] for r in range(len(features_list))]
    node_type_feature_init = F.one_hot(torch.arange(0, len(features_list)), num_classes=len(features_list))
    for i in range(0, len(features_list)):
        node_type_feature[i] = node_type_feature_init[i].expand(features_list[i].shape[0],
                                                                    len(node_type_feature_init)).to(
                device).type(torch.FloatTensor)
    in_dims_2 = [features.shape[1] for features in node_type_feature]  # 这个是节点类型特征的输入维度



    adjm = sparse.csr_matrix(adjM)
    g = dgl.DGLGraph(adjm + (adjm.T))  # 增加双向边
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)  # 这里的g和HGAT中的有点小差距


    nb_classes = label.shape[-1]#

    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)



    heads = [num_heads] * args.num_layers + [1]
    model = HGTA(args.hidden_dim, feats_dim_list, args.Tfeat_drop, args.Tattn_drop, P, g,in_dims_1,in_dims_2,num_hidden,num_classes,args.num_layers,heads,activation,negative_slope,residual,
                      args.tau, args.lam,args.Sfeat_drop, args.Sattn_drop)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        gs = gs.cuda()
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, gs, onehot_feature_list, node_type_feature)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HGTA_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HGTA_'+own_str+'.pkl'))
    model.eval()
    os.remove('HGTA_'+own_str+'.pkl')
    embeds1 = model.get_embeds1(feats,onehot_feature_list,node_type_feature)
    embeds2 = model.get_embeds2(gs, feats)
    for i in range(len(idx_train)):
        evaluate(embeds1, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                     args.eva_lr, args.eva_wd)
    for i in range(len(idx_train)):
        evaluate(embeds2, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                     args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")

    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()

if __name__ == '__main__':
    train()
