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
from utils import cal_acc

def data_load(args):
    '''
        载入所需数据.

    returns:
        dset_loaders (dict of Dataloader): {'src_train': list_of_src_train_loaders, 'src_test': list_of_src_test_loaders, 'target': ...}
    '''
    # 准备数据
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    target_txt = open(args.tar_dset_path).readlines()
    dsets["target"] = ImageList(target_txt, transform=image_test())
    dset_loaders["target_test"] = DataLoader(dsets["target"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def test_target(args):
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
    modelpath = args.output_dir_tar + f'/target_F_par_{str(args.cls_par)}.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_tar + f'/target_B_par_{str(args.cls_par)}.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_tar + f'/target_C_par_{str(args.cls_par)}.pt'
    netC.load_state_dict(torch.load(modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.dset == 'VISDA-C':
        acc, acc_list = cal_acc(dset_loaders['target_test'], netF, netB, netC, avg_class_acc=True)
        print(f"Task: {args.name}: {acc:.2f}%")
        print(f"Acc List: {acc_list}")
        exit()
    
    # 仅对单一模型测试性能.
    acc, _ = cal_acc(dset_loaders['target_test'], netF, netB, netC)
    log_str = f'Testing: Task: {args.name}, Accuracy = {acc:.2f}%\n'

    print(log_str)

if __name__ == "__main__":
    """
        描述:
            测试最后的目标域模型的性能.
        
        输入:
            数据集:
                数据集一级目录下需要存储描述文件{domain}_list.txt
                目标域: args.tar_dset_path => folder + args.dset + args.t
            模型:
                目标域模型输出目录: args.output_dir_tar => args.output_target + args.dset + args.name

        输出:
            字符串 (控制台输出)
    """

    parser = argparse.ArgumentParser(description='BFDA-TEST-SOURCE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=[0], nargs='+', help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=2, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain-net', 'domain-net-20', 'domain-net-40', 'domain-net-80', 'domain-net-160'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output_target', type=str, default='ckps/target_more_finetune')     # 目标域模型输出目录.
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

    # 源域以及目标域数据集注释文件的路径.
    folder = './data/'
    args.tar_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # 测试指定的源域的单一模型或者聚合模型在指定目标域上的性能.
    args.name = '_'.join(names[s].upper() for s in args.s) + '_to_' + names[args.t].upper()    # 进行测试的names列表: [domainA_to_domainB, ...]

    # 目标域模型输出目录列表
    args.output_dir_tar = osp.join(args.output_target, args.dset, args.name).replace('\\', '/')

    test_target(args)