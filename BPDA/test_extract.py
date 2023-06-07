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

from utils import op_copy, lr_scheduler, cal_acc, image_train, image_test, print_args
from utils import cal_acc, print_log

import time
import datetime

def data_load(args):
    '''
        载入所需数据.

    returns:
        dset_loaders (dict of Dataloader): {'target': ...}
    '''
    # 准备数据
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    target_txt = open(args.tar_dset_path).readlines()
    dsets["target"] = ImageList(target_txt, transform=image_test())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def test_extract(args):
    # 载入目标域数据集
    dset_loaders = data_load(args)

    # 设置网络结构
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    # 载入模型权重
    modelpath = args.output_dir_extract + f'/extract_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_extract + f'/extract_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_extract + f'/extract_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.dset == 'VISDA-C':
        acc, acc_list = cal_acc(dset_loaders['target'], netF, netB, netC, avg_class_acc=True)
        print(f"Task: {args.name}: {acc:.2f}%")
        print(f"Acc List: {acc_list}")
        exit()
    
    # 仅对单一模型测试性能.
    acc, _ = cal_acc(dset_loaders['target'], netF, netB, netC)
    log_str = f'Testing: Task: {args.name}, Accuracy = {acc:4.2f}%\n'

    print_log(log_str, args)

if __name__ == "__main__":
    """
        描述:
            测试窃取出来的模型在各个域上的性能.
        
        输入:
            数据集:
                数据集一级目录下需要存储描述文件{domain}_list.txt
                目标域: args.tar_dset_path => folder + args.dset + args.t
            模型:
                窃取模型输出目录: args.output_dir_extract => args.output_extract + args.dset + args.extract_name

        输出:
            字符串 (控制台输出)
    """

    parser = argparse.ArgumentParser(description='BFDA-TEST-EXTRACT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=[0], nargs='+', help="source")      # 指定所需要使用的窃取模型是窃取了s指定的聚合模型
    parser.add_argument('--t', type=int, default=[1], nargs='+', help="target")   # 指定在t指定的目标域上测试窃取模型的性能
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain-net', 'domain-net-20', 'domain-net-40', 'domain-net-80', 'domain-net-160'])
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output_extract', type=str, default='ckps/target_more')     # 目标域模型输出目录.

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
        print("该数据集上的代码尚未实现.")
        exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # 窃取模型的输出目录
    args.extract_name = '_'.join(names[s].upper() for s in args.s)
    args.output_dir_extract = osp.join(args.output_extract, args.dset, args.extract_name).replace('\\', '/')

    # 日志文件
    args.log_file = open(osp.join(args.output_dir_extract, 'log_test.txt').replace('\\', '/'), 'w')
    current_time = datetime.datetime.now()
    log_str = f"Start Time: {str(current_time).split('.')[0]}; Output Dir: {args.output_dir_extract}"
    print_log(log_str, args)
    args.log_file.write(print_args(args)+'\n')
    args.log_file.flush()

    # 目标域数据集注释文件的路径.
    for t in args.t:
        folder = './data/'
        args.tar_dset_path = folder + args.dset + '/' + names[t] + '_list.txt'

        # 测试指定的源域的单一模型或者聚合模型在指定目标域上的性能.
        args.name = args.extract_name + '_to_' + names[t].upper()

        test_extract(args)