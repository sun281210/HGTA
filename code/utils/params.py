import argparse
import sys


argv = sys.argv
dataset = 'DBLP'


def ACM_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--ratio', type=int, default=[20,40,60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=100000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--Tfeat_drop', type=float, default=0.5)
    parser.add_argument('--Tattn_drop', type=float, default=0.5)
    parser.add_argument('--Sfeat_drop', type=float, default=0.5)
    parser.add_argument('--Sattn_drop', type=float, default=0.5)
#    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6, 1])#A；S
    parser.add_argument('--lam', type=float, default=0.2)
    parser.add_argument('--slope', type=float, default=0.05, help='激活函数的负值倾斜度')
    parser.add_argument('--res', default=False, help='残差连接')
    parser.add_argument('--num_layers', type=int, default=5, help='神经网络层数')
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every n4ode type
    return args


def DBLP_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="DBLP")
    parser.add_argument('--ratio', type=int, default=[20,40,60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--Tfeat_drop', type=float, default=0.5)
    parser.add_argument('--Tattn_drop', type=float, default=0.5)
    parser.add_argument('--Sfeat_drop', type=float, default=0.5)
    parser.add_argument('--Sattn_drop', type=float, default=0.6)
    #    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6, 1])#A；S
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--slope', type=float, default=0.05, help='激活函数的负值倾斜度')
    parser.add_argument('--res', default=False, help='残差连接')
    parser.add_argument('--num_layers', type=int, default=1, help='神经网络层数')

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args




def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20,40,60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--Tfeat_drop', type=float, default=0.5)
    parser.add_argument('--Tattn_drop', type=float, default=0.5)
    # parser.add_argument('--sample_rate', nargs='+', type=int, default=[1, 18, 2])
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--Sfeat_drop', type=float, default=0.5)
    parser.add_argument('--Sattn_drop', type=float, default=0.5)
    #    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6, 1])#A；S
    parser.add_argument('--slope', type=float, default=0.05, help='激活函数的负值倾斜度')
    parser.add_argument('--res', default=False, help='残差连接')
    parser.add_argument('--num_layers', type=int, default=1, help='神经网络层数')

    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args

def set_params():
    if dataset == "ACM":
        args = ACM_params()
    elif dataset == "DBLP":
        args = DBLP_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args
