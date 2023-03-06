import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch


class topo_encoder(nn.Module):
    def __init__(self,g,in_dims,in_dims_2,num_hidden,num_classes,num_layers,heads,activation,Tfeat_drop,Tattn_drop,negative_slope,residual
    ):
        super(topo_encoder, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.hgat_layers = nn.ModuleList()
        self.activation = activation

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)



        #--------------------------------------这个是特征转换
        self.ntfc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims_2])
        for ntfc in self.ntfc_list:
            nn.init.xavier_normal_(ntfc.weight, gain=1.414)


        #--------------------------------------
        self.hgat_layers.append(GATConv(num_hidden*2, num_hidden, heads[0],Tfeat_drop, Tattn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            self.hgat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l],Tfeat_drop, Tattn_drop, negative_slope, residual, self.activation))
        # output projection
        self.hgat_layers.append(GATConv(num_hidden * heads[-2], num_hidden, heads[-1],Tfeat_drop, Tattn_drop, negative_slope, residual, None))
        self.lines=nn.Linear(num_hidden,num_classes,bias=True)
        nn.init.xavier_normal_(self.lines.weight, gain=1.414)

    def forward(self,onehot_feature_list, node_type_feature):
        h = []  # 节点特征

        h2=[]#节点类型特征
        for fc, feature in zip(self.fc_list, onehot_feature_list):
            h.append(fc(feature))



        h = torch.cat(h, 0)#逐行合并
        # #第三层长度是192
        # #-------------------------------
        for ntfc, feature in zip(self.ntfc_list, node_type_feature):
            h2.append(ntfc(feature))

        h2 = torch.cat(h2, 0)  # 逐行合并
        h = torch.cat((h, h2), 1)  # 将节点特征和节点类型特征进行合并
        for l in range(self.num_layers):

            h = self.hgat_layers[l](self.g, h).flatten(1)
        h = self.hgat_layers[-1](self.g, h).mean(1)

        z_topo=h[:4057,:]

        return z_topo