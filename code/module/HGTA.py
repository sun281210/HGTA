import torch.nn as nn
import torch.nn.functional as F
from .sem_encoder import sem_encoder
from .topo_encoder import topo_encoder
from .contrast import Contrast


class HGTA(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, Tfeat_drop, Tattn_drop, P, g,in_dims_1,in_dims_2,num_hidden,num_classes,num_layers,heads,activation,negative_slope,residual,
                  tau, lam,Sfeat_drop,Sattn_drop):
        super(HGTA, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        if Tfeat_drop > 0:
            self.feat_drop = nn.Dropout(Tfeat_drop)
        else:
            self.feat_drop = lambda x: x
        self.sem = sem_encoder( P, hidden_dim, Sfeat_drop,Sattn_drop)
        self.topo = topo_encoder(g,in_dims_1,in_dims_2,num_hidden,num_classes,num_layers,heads,activation,Tfeat_drop,Tattn_drop,negative_slope,residual)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, feats, pos, gs, onehot_feature_list, node_type_feature):# p a s
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_sem = self.sem(gs, h_all[0])
        z_topo = self.topo(onehot_feature_list, node_type_feature)
        loss = self.contrast(z_sem, z_topo, pos)

        return loss

    def get_embeds1(self, feats,onehot_feature_list,node_type_feature):
         z_topo = []
         for i in range(len(feats)):
            z_topo.append(F.elu(self.fc_list[i](feats[i])))
         z_topo=self.topo(onehot_feature_list,node_type_feature)
         return z_topo.detach()


    def get_embeds2(self, gs, feats):
        z_sem = F.elu(self.fc_list[0](feats[0]))
        z_sem = self.sem(gs, z_sem)
        return z_sem.detach()