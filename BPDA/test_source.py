import argparse
import os, sys
import os.path as osp
import random, pdb, math, copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import numpy as np

from data_list import ImageList
import network, loss
from loss import CrossEntropyLabelSmooth

from utils import op_copy, lr_scheduler, cal_acc, image_train, image_test, print_args, cal_acc_aggre

import time
import datetime
 
def data_load(args):
    '''
        载入所需数据.

    returns:
        dset_loaders (dict of Dataloader): {'target_list': ...}
    '''
    # 准备数据
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    dset_loaders["target_list"] = []
    for tar_dset_path in args.tar_dset_path_list:
        target_txt = open(tar_dset_path).readlines()
        dsets["target"] = ImageList(target_txt, transform=image_test())
        dset_loaders["target_list"].append(DataLoader(dsets["target"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False))

    return dset_loaders

def test_target_aggre(args):
    # 载入目标域数据集
    dset_loaders = data_load(args)

    # 设置网络结构
    netF_list = []
    netB_list = []
    netC_list = []

    for output_dir_src in args.output_dir_src_list:
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net).cuda()
        elif args.net[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=args.net).cuda()  

        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
        netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
        
        # 载入模型权重
        modelpath = output_dir_src + '/source_F.pt'   
        netF.load_state_dict(torch.load(modelpath))
        modelpath = output_dir_src + '/source_B.pt'   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = output_dir_src + '/source_C.pt'   
        netC.load_state_dict(torch.load(modelpath))
        netF.eval()
        netB.eval()
        netC.eval()

        netF_list.append(netF)
        netB_list.append(netB)
        netC_list.append(netC)

    # 使用多模型聚合输出
    # 依次计算模型在各个域上的accuracy.
    start = time.time()
    print(f'Testing Aggregation Model of: {args.name_src_list} on domain: {args.name_tar_list}')

    target_num = len(args.t)
    for tar_idx in range(target_num):
        if args.dset == 'VISDA-C':
            acc, acc_list = cal_acc_aggre(dset_loaders['target_list'][tar_idx], netF_list, netB_list, netC_list, avg_class_acc=True)
            print(f"Task: {args.name_list[tar_idx]}: {acc:.2f}%")
            print(f"Acc List: {acc_list}")
            exit()
        else:
            acc, _ = cal_acc_aggre(dset_loaders['target_list'][tar_idx], netF_list, netB_list, netC_list)
        elapsed = datetime.timedelta(seconds=time.time() - start)
        remain = elapsed / (tar_idx + 1) * (target_num - (tar_idx + 1))

        print(f"Task: {args.name_list[tar_idx]}: {acc:.2f}%; elpased: {str(elapsed).split('.')[0]}, remain: {str(remain).split('.')[0]}")

if __name__ == "__main__":
    """
        描述:
            测试聚合模型或者单个模型在目标域列表中的每个目标域上的性能.
            仅输出到控制台, 不生成日志文件.
        
        输入:
            数据集:
                数据集一级目录下需要存储描述文件{domain}_list.txt
                目标域: args.tar_dset_path => folder + args.dset + args.t
            模型:
                源域模型输出目录: args.output_dir_src_list => args.output + args.dset + args.name_src

        输出:
            字符串 (控制台输出)
    """

    parser = argparse.ArgumentParser(description='BFDA-TEST-SOURCE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=[0], nargs='+', help="source")
    parser.add_argument('--t', type=int, default=[1], nargs='+', help="target")
    parser.add_argument('--max_epoch', type=int, default=2, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain-net', 'domain-net-20', 'domain-net-40', 'domain-net-80', 'domain-net-160'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ckps/source')
    parser.add_argument('--data_split', type=str, default='train_val', choices=['full', 'train_val'])
    args = parser.parse_args()

    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    elif args.dset == 'domain-net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345
    elif args.dset == 'domain-net-20':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 20
    elif args.dset == 'domain-net-40':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 40
    elif args.dset == 'domain-net-80':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 80
    elif args.dset == 'domain-net-160':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 160    
    else:
        print("在该数据集上的代码尚未实现.")
        exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # 目标域数据集注释文件路径的列表
    folder = './data/'
    args.tar_dset_path_list = [folder + args.dset + '/' + names[t] + '_list.txt' for t in args.t]

    # 源域模型输出目录列表
    args.name_src_list = [names[s].upper() for s in args.s]
    args.output_dir_src_list = [osp.join(args.output, args.dset, name_src).replace('\\', '/') for name_src in args.name_src_list]

    # 测试指定的源域的单一模型或者聚合模型在指定目标域上的性能.
    args.name_tar_list = [names[t].upper() for t in args.t]
    args.name_list = ['_'.join(names[s].upper() for s in args.s) + '_to_' + names[t].upper() for t in args.t]   # 进行测试的names列表: [domainA_to_domainB, ...]

    test_target_aggre(args)