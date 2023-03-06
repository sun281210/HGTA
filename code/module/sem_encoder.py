import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv





class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]#Z=B*H
        return z_mp


class sem_encoder(nn.Module):
    def __init__(self, P, hidden_dim, Sfeat_drop,Sattn_drop):
        super(sem_encoder, self).__init__()
        self.P = P
        # self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.node_level = nn.ModuleList([GATConv(hidden_dim, hidden_dim,1,Sfeat_drop,Sattn_drop,activation=F.elu) for _ in range(P)])
        self.att = Attention(hidden_dim, Sattn_drop)
        self.linear1 = nn.Linear(hidden_dim*8, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim*8, hidden_dim, bias=True)
        self.linear3 = nn.Linear(hidden_dim * 8, hidden_dim, bias=True)

    #GAT
    def forward(self,gs, h):
        embed = []
        for i, g in enumerate(gs):
            embeds=self.node_level[i](g,h).flatten(1)
            embed.append(embeds)
        Z_sem = self.att(embed)
        return Z_sem
