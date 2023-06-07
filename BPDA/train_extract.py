import argparse
import os, sys
import os.path as osp
import random, pdb, math, copy
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
from tensorboardX import SummaryWriter

import network, loss
from data_list import ImageList, ImageList_idx
from utils import op_copy, lr_scheduler, image_train, image_test, cal_acc, print_args
from utils import obtain_label_without_refinement_aggre, obtain_label_with_DEPICT_aggre, obtain_softlabel_without_refinement_aggre     # 源域模型上打标签的方法.
from utils import obtain_label_without_refinement_aggre_hard_label
from utils import get_lr
from utils import print_log, label_distribution_vis
from loss import CrossEntropySoftLabel

def data_load(args):
    '''
    准备数据集.
    模型窃取过程需要的数据集包括: 用于窃取模型的第三方数据集, 测试目标域性能的目标域数据集列表.

    returns:
        dset_loaders (dict): {"third_train": ..., "third_test": ..., "target_test_list": ...}
    '''
    # 准备数据.
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    
    # 第三方数据集需要用于train, 并且查询构造伪标签时, 仅需在train之前构造1次即可. 故也需要不shuffle的test.
    third_txt = open(args.third_dset_path).readlines()
    dsets["third_train"] = ImageList_idx(third_txt, transform=image_train())
    dset_loaders["third_train"] = DataLoader(dsets["third_train"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)
    dsets["third_test"] = ImageList_idx(third_txt, transform=image_test())
    dset_loaders["third_test"] = DataLoader(dsets["third_test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False, pin_memory=True)

    # 用于测试性能的目标域数据集: 实际上在模型窃取的过程中没有用到目标域信息
    dset_loaders["target_test_list"] = []
    for target_dset_path in args.target_dset_path_list:
        target_txt = open(target_dset_path).readlines()
        dsets["target"] = ImageList_idx(target_txt, transform=image_test())
        dset_loaders["target_test_list"].append(DataLoader(dsets["target"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False, pin_memory=True))

    return dset_loaders

def extract_source(args):
    # 构建tensorboard日志
    tensorboard_log = osp.join(args.output_dir_extract, 'tensorboard_log').replace('\\', '/')
    if os.path.exists(tensorboard_log):
        shutil.rmtree(tensorboard_log)
    writer = SummaryWriter(logdir=tensorboard_log)

    # 数据集载入
    dset_loaders = data_load(args)
    
    # 载入各源域预训练模型 (用于给第三方数据打伪标签)
    netF_source_list = []
    netB_source_list = []
    netC_source_list = []
    for output_dir_src in args.output_dir_src_list:
        # 网络结构
        if args.net[0:3] == 'res':
            netF_source = network.ResBase(res_name=args.net).cuda()
        elif args.net[0:3] == 'vgg':
            netF_source = network.VGGBase(vgg_name=args.net).cuda()  

        netB_source = network.feat_bootleneck(type=args.classifier, feature_dim=netF_source.in_features, bottleneck_dim=args.bottleneck).cuda()
        netC_source = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

        # 模型参数
        print_log(f"Load source model from: {output_dir_src}", args)
        modelpath = output_dir_src + '/source_F.pt'
        netF_source.load_state_dict(torch.load(modelpath))
        modelpath = output_dir_src + '/source_B.pt'
        netB_source.load_state_dict(torch.load(modelpath))
        modelpath = output_dir_src + '/source_C.pt'
        netC_source.load_state_dict(torch.load(modelpath))

        # 源域模型参数固定, 不参与训练.
        netF_source.eval()
        for k, v in netF_source.named_parameters():
            v.requires_grad = False
        netB_source.eval()
        for k, v in netB_source.named_parameters():
            v.requires_grad = False
        netC_source.eval()
        for k, v in netC_source.named_parameters():
            v.requires_grad = False
        
        netF_source_list.append(netF_source)
        netB_source_list.append(netB_source)
        netC_source_list.append(netC_source)
    
    # 窃取模型网络结构
    # 网络结构
    if args.net[0:3] == 'res':
        netF_extract = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF_extract = network.VGGBase(vgg_name=args.net).cuda()  

    netB_extract = network.feat_bootleneck(type=args.classifier, feature_dim=netF_extract.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC_extract = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # 设置优化器参数.
    # 其中, 对netF学习率的衰减速度lr_decay1是netB和netC学习率的衰减速度lr_decay2的10倍.
    param_group_extract = []
    for k, v in netF_extract.named_parameters():
        if args.lr_decay1 > 0:
            param_group_extract += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB_extract.named_parameters():
        if args.lr_decay2 > 0:
            param_group_extract += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC_extract.named_parameters():
        if args.lr_decay2 > 0:
            param_group_extract += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer_extract = optim.SGD(param_group_extract)
    optimizer_extract = op_copy(optimizer_extract)

    # 仅在窃取开始前为第三方数据集中的所有数据打一次伪标签
    log_str = "Querying presudo labels..."
    print_log(log_str, args)

    start = time.time()
    for src_num in range(len(args.s)):
        netF_source_list[src_num].eval()
        netB_source_list[src_num].eval()
        netC_source_list[src_num].eval()
    
    # 直接使用聚合的源域模型打标签 -> 源域聚合模型打上标签之后进行精炼
    if args.use_softlabel:
        # mem_label: [num_samples, num_classes]
        mem_label = obtain_softlabel_without_refinement_aggre(dset_loaders['third_test'], netF_source_list, netB_source_list, netC_source_list, args)
        print_log("<<< Use Soft Label >>>", args)
    else:
        if args.hardlabel_aggre:
            ''' 若进行预测结果聚合时使用hard-label聚合, 而不是使用logits聚合. '''
            args.use_softlabel = 1  # 当使用多个硬标签计算软标签时, 下面的计算要使用软标签来进行.
            mem_label = obtain_label_without_refinement_aggre_hard_label(dset_loaders['third_test'], netF_source_list, netB_source_list, netC_source_list, args)
            print_log("<<< Use Hard Label >>>", args)
        else:
            # mem_label: [num_samples]
            if args.no_DEPICT:
                mem_label = obtain_label_without_refinement_aggre(dset_loaders['third_test'], netF_source_list, netB_source_list, netC_source_list, args)
                print_log("<<< Without DEPICT Algorithm >>>", args)
            else:
                mem_label = obtain_label_with_DEPICT_aggre(dset_loaders['third_test'], netF_source_list, netB_source_list, netC_source_list, args)
                print_log("<<< Using DEPICT Algorithm to refine the labels >>>", args)
    
    mem_label = torch.from_numpy(mem_label).cuda()

    # 统计标签分布.
    label_distribution = {}
    for label_idx in range(args.class_num):
        label_distribution[f'{label_idx}'] = torch.sum(mem_label == label_idx).detach().cpu().item()
    print_log(f"label_distribution: {label_distribution}", args)
    if args.label_distribution_vis:
        label_distribution_vis(mem_label.detach().cpu().numpy(), osp.join(args.output_dir_extract, 'label_distribution_refinement.jpg').replace('\\', '/'), args.class_num)
    elapsed =  datetime.timedelta(seconds=time.time() - start)
    print_log(f"presudo label query takes: {str(elapsed).split('.')[0]}", args)

    # 窃取(训练)模型
    print_log("Extracting source models...", args)
    start = time.time()
    acc_target_avg_best = 0.0
    acc_target_dict_best = {}
    max_iter = args.max_epoch * len(dset_loaders["third_train"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        # 取训练数据
        try:
            inputs_train, _, train_idx = iter_train.next()     # [image, label, index], 第三方数据集本身的label是没有意义的.
        except:
            iter_train = iter(dset_loaders["third_train"])
            inputs_train, _, train_idx = iter_train.next()

        if inputs_train.size(0) == 1:
            continue

        inputs_train = inputs_train.cuda()

        # TODO: 可视化取出来的数据.

        iter_num += 1
        lr_scheduler(optimizer_extract, iter_num=iter_num, max_iter=max_iter)   # 手动调整学习率

        # 获取模型输出.
        features_extract = netB_extract(netF_extract(inputs_train))
        outputs_extract = netC_extract(features_extract)

        # 总损失 = 分类损失
        # 分类损失
        if args.cls_par > 0:
            pseudo_label = mem_label[train_idx]
            ''' 是否使用softlabel来计算loss. '''
            if args.use_softlabel:
                classifier_loss = CrossEntropySoftLabel()(outputs_extract, pseudo_label)
            else:
                classifier_loss = nn.CrossEntropyLoss()(outputs_extract, pseudo_label.long())
            classifier_loss *= args.cls_par

        optimizer_extract.zero_grad()
        classifier_loss.backward()
        optimizer_extract.step()

        # 每隔interval_iter, 测试模型性能
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF_extract.eval()
            netB_extract.eval()
            netC_extract.eval()

            # 实际上这里应该在除了args.s中存在的域上的其他域的平均acc来衡量该模型的性能.

            acc_target_avg_test = 0.0
            acc_target_dict = {}
            num_of_target = len(dset_loaders['target_test_list'])
            for t in range(num_of_target):
                if args.dset == "VISDA-C":
                    ''' 对于VISDA-C数据集, 其评价指标是average class accuracy. '''
                    acc_target_test, _ = cal_acc(dset_loaders["target_test_list"][t], netF_extract, netB_extract, netC_extract, avg_class_acc=True)
                else:
                    acc_target_test, _ = cal_acc(dset_loaders['target_test_list'][t], netF_extract, netB_extract, netC_extract)
                acc_target_dict[args.name_target_list[t]] = acc_target_test
                acc_target_avg_test += acc_target_test
            
            acc_target_avg_test /= num_of_target

            # 记录日志信息
            # 学习率
            current_lr = get_lr(optimizer_extract)['lr_0']
            writer.add_scalars('learning_rate', {'lr': current_lr}, global_step=iter_num)
            # 时间信息
            elapsed =  datetime.timedelta(seconds=time.time() - start)
            remain = elapsed / (iter_num + 1e-8) * (max_iter - iter_num)
            # 损失值
            writer.add_scalars('classifier_loss', {'classifier_loss': classifier_loss}, global_step=iter_num)
            # 性能
            acc_target_dict["Average"] = acc_target_avg_test
            writer.add_scalars('target_test_accuracy', acc_target_dict, global_step=iter_num)

            # 总日志
            log_str = f"Task: Extract {args.name}, Iter:{iter_num}/{max_iter}; elapsed: {str(elapsed).split('.')[0]}, remain: {str(remain).split('.')[0]};To save: {acc_target_avg_test > acc_target_avg_best}\n"
            log_str += f"Average target test accuracy = {acc_target_avg_test:4.2f}%\n"
            log_str += f"Target accuracy dict: {acc_target_dict}\n"
            log_str += f"classifier loss: {classifier_loss}, learning rate: {current_lr}\n"
            print_log(log_str, args)

            if acc_target_avg_test > acc_target_avg_best:
                acc_target_avg_best = acc_target_avg_test
                acc_target_dict_best = acc_target_dict

                # 下面这3个是dict, 而不是model.
                best_netF_extract = netF_extract.state_dict()
                best_netB_extract = netB_extract.state_dict()
                best_netC_extract = netC_extract.state_dict()

                # 随时保存训练过程中找到的最优性能.
                if args.issave:
                    torch.save(best_netF_extract, osp.join(args.output_dir_extract, "extract_F.pt").replace('\\', '/'))
                    torch.save(best_netB_extract, osp.join(args.output_dir_extract, "extract_B.pt").replace('\\', '/'))
                    torch.save(best_netC_extract, osp.join(args.output_dir_extract, "extract_C.pt").replace('\\', '/'))

            netF_extract.train()
            netB_extract.train()
            netC_extract.train()
    
    netF_extract.load_state_dict(best_netF_extract)
    netB_extract.load_state_dict(best_netB_extract)
    netC_extract.load_state_dict(best_netC_extract)

    log_str = f"The extracted best aggregation model of source domains: {args.name_src_list}; Test accuracy on target domains: {args.name_target_list}\n"
    log_str += f"Best Average Target Acc: {acc_target_avg_best}\n"
    log_str += f"Target Accuracy Dict: {acc_target_dict_best}\n"
    log_str += f"Finish Time: {str(datetime.datetime.now()).split('.')[0]}"
    print_log(log_str, args)

    return netF_extract, netB_extract, netC_extract

if __name__ == "__main__":
    """
        描述:
            模型窃取的代码.
            根据Model Extraction Attack的思想, 使用第三方公开数据集探查黑盒模型的输入-输出关系, 从而窃取模型.
            该代码仅是将模型窃取出来, 而不做任何与目标域相关的事, 因而窃取出来的模型可以放到多个目标域上进行调优.
            该代码保存到各个目标域平均性能最优的模型, 但实际场景下, 窃取过程中无法利用目标域数据来测试.
        
        输入:
            数据集:
                第三方数据集:
                    args.target_dset_path => folder + args.dset + names[args.t]
                    args.test_dset_path => folder + args.dset + names[args.t]
            源域模型保存目录:
                args.output_dir_src_list => args.output_src + args.dset + names[args.s]
        
        输出:
            窃取模型保存目录:
                args.output_dir_extract => args.output_target + args.dset + names[args.s] + names[args.t]
    """
    
    parser = argparse.ArgumentParser(description='BFDA-EXTRACT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=[0], nargs='+', help="source")     # 多源域聚合模型
    parser.add_argument('--t', type=int, default=[1], nargs='+', help="target")
    parser.add_argument('--max_epoch', type=int, default=1, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-caltech', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain-net', 'domain-net-centroids', 'domain-net-20', 'domain-net-40', 'domain-net-80', 'domain-net-160'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--cls_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)     # 对特征提取器netF的学习率衰减
    parser.add_argument('--lr_decay2', type=float, default=1.0)     # 对BN层netB的学习率衰减

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output_src', type=str, default='ckps/source')    # 预训练源域模型存储路径
    parser.add_argument('--output_extract', type=str, default='ckps/extract')   # 窃取模型输出路径
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--use_softlabel', type=int, default=0)     # NOTE: 是否使用软标签进行训练.
    parser.add_argument('--hardlabel_aggre', type=int, default=1)   # NOTE: 模型进行聚合时是否使用硬标签进行聚合.
    parser.add_argument('--no_DEPICT', action="store_true")    # FIXME: 这里默认为FALSE, 是否在窃取过程中使用DEPICT算法进行标签精炼.

    parser.add_argument('--label_distribution_vis', type=bool, default=False)
    # parser.add_argument('--third_dset_path', type=str, default='data/ImageNet/ILSVRC2012_img_val_list.txt')     # 第三方数据集的注释文件.
    parser.add_argument('--third_dset_path', type=str, default='data/ImageNet/imagenet_500_5_list.txt')     # 使用ImageNet中500类图像, 每类5张作为第三方数据集.
    # parser.add_argument('--third_dset_path', type=str, default='data/COCO/train2017_list.txt')     # 使用VOC2012作为第三方数据集.

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
    elif args.dset == 'domain-net-centroids':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345
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

    # 数据集路径: 第三方数据集窃取 + 目标域数据集性能测试.
    folder = './data/'
    args.target_dset_path_list = [folder + args.dset + '/' + names[t] + '_list.txt' for t in args.t]
    args.name_target_list = [names[t].upper() for t in args.t]
    args.name_src_list = [names[s].upper() for s in args.s]

    # 模型路径: 
    args.name = '_'.join([names[s].upper() for s in args.s])
    args.output_dir_src_list = [osp.join(args.output_src, args.dset, names[s].upper()).replace('\\', '/') for s in args.s]
    args.output_dir_extract = osp.join(args.output_extract, args.dset, args.name).replace('\\', '/')
    
    if not osp.exists(args.output_dir_extract):
        os.makedirs(args.output_dir_extract)

    # 日志文件
    args.log_file = open(osp.join(args.output_dir_extract, 'log_extract.txt').replace('\\', '/'), 'w')
    current_time = datetime.datetime.now()
    log_str = f"Start Time: {str(current_time).split('.')[0]}; Output Dir: {args.output_dir_extract}"
    print_log(log_str, args)
    args.log_file.write(print_args(args)+'\n')
    args.log_file.flush()

    # 窃取源域模型
    extract_source(args)