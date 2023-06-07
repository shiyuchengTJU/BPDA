import argparse
import os, sys
import os.path as osp
import random, pdb, math, copy
from tqdm import tqdm
import time
import datetime
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import numpy as np
from tensorboardX import SummaryWriter

import network, loss
from data_list import ImageList, ImageList_idx
from utils import op_copy, lr_scheduler, image_train, image_test, cal_acc, print_args, obtain_label
from utils import print_log, get_lr

def data_load(args):
    '''
    准备数据集. target与test的区别在于是否进行shuffle以及batch_size.
    在UDA的设置下, 可以使用目标域的所有数据进行fine-tune, 并在整个目标域上计算测试性能.

    returns:
        dset_loaders (dict): {"target_train": ..., "target_test": ...}
    '''
    # 准备数据.
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    target_txt = open(args.target_dset_path).readlines()

    dsets["target_train"] = ImageList_idx(target_txt, transform=image_train())
    dset_loaders["target_train"] = DataLoader(dsets["target_train"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)
    dsets["target_test"] = ImageList_idx(target_txt, transform=image_test())
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False, pin_memory=True)

    return dset_loaders

def train_target(args):
    # tensorboard日志目录
    tensorboard_log = osp.join(args.output_dir_target, 'tensorboard_log').replace('\\', '/')
    if os.path.exists(tensorboard_log):
        shutil.rmtree(tensorboard_log)
    writer = SummaryWriter(logdir=tensorboard_log)

    # 数据集载入
    dset_loaders = data_load(args)

    # 设置主干网络
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    print_log(f"Load extracted model from {args.output_dir_extract}", args)
    modelpath = args.output_dir_extract + '/extract_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_extract + '/extract_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_extract + '/extract_C.pt'
    netC.load_state_dict(torch.load(modelpath))


    # 固定分类器 netC (hypothesis)
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    # 训练特征提取器: netF, netB.
    # 其中, 对netF学习率的衰减速度lr_decay1是netB学习率的衰减速度lr_decay2的10倍.
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    start = time.time()
    acc_target_best = 0.0
    max_iter = args.max_epoch * len(dset_loaders["target_train"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    while iter_num < max_iter:
        try:
            inputs_target, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target_train"])
            inputs_target, _, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue
        
        # 每隔interval_iter就使用当前模型重新打一次伪标签.
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            refine_start = time.time()
            print_log(f"Iter: {iter_num}; Refining the presudo labels...", args)

            netF.eval()
            netB.eval()
            ''' utils.obtain_label已经内置了除了聚类外, 是否使用DEPICT算法来进行调优的选项. '''
            mem_label, refined_target_accuracy = obtain_label(dset_loaders['target_test'], netF, netB, netC, args, get_refined_accuracy=True, cal_class_acc=True)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

            refine_elapsed = datetime.timedelta(seconds=time.time() - refine_start)
            print_log(f"Refining the presudo labels taken {str(refine_elapsed).split('.')[0]}", args)

        inputs_target = inputs_target.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)   # 手动调整学习率

        features_target = netB(netF(inputs_target))
        outputs_target = netC(features_target)

        # 计算损失: 总损失 = 分类损失 + 互信息损失
        # 分类损失
        if args.cls_par > 0:
            pred = mem_label[tar_idx].long()
            classifier_loss = nn.CrossEntropyLoss()(outputs_target, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        # 互信息损失
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # 每隔interval_iter, 测试模型性能
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()

            # 计算 target accuracy
            if args.dset == "VISDA-C":
                ''' 对于VISDA-C数据集, 其评价指标是average class accuracy. '''
                acc_target_test, acc_list = cal_acc(dset_loaders['target_test'], netF, netB, netC, avg_class_acc=True)
                print_log(f"Acc List: {acc_list}")
            else:
                acc_target_test, _, tgt_class_acc = cal_acc(dset_loaders['target_test'], netF, netB, netC, cal_class_acc=True)

            # 记录日志
            # 时间信息
            elapsed = datetime.timedelta(seconds=time.time() - start)
            remain = elapsed / (iter_num + 1e-8) * (max_iter - iter_num)
            # 学习率
            current_lr = get_lr(optimizer)['lr_0']
            writer.add_scalars('learning_rate', {'lr': current_lr}, global_step=iter_num)
            # 损失
            writer.add_scalars('classifier_loss', {'classifier_loss': classifier_loss}, global_step=iter_num)
            # 性能信息
            writer.add_scalars('accuracy', {'target_test_accuracy': acc_target_test, 'refined_target_accuracy': refined_target_accuracy * 100}, global_step=iter_num)
            # 总日志
            log_str = f"Task: {args.name}, Iter:{iter_num}/{max_iter}; elapsed: {str(elapsed).split('.')[0]}, remain: {str(remain).split('.')[0]}\n"
            log_str += f"Target accuracy = {acc_target_test:.2f}%; Refined target accuracy: {refined_target_accuracy * 100:.2f}%\n"
            log_str += f"Target class acc: {tgt_class_acc}\n"
            log_str += f"loss: {classifier_loss}  learning rate: {current_lr}\n"
            print_log(log_str, args)

            if acc_target_test > acc_target_best:
                acc_target_best = acc_target_test

                # 下面这3个是dict, 而不是model.
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

                # 保存训练过程中找到的最优模型
                if args.issave:
                    torch.save(best_netF, osp.join(args.output_dir_target, "target_F_" + args.savename + ".pt").replace('\\', '/'))
                    torch.save(best_netB, osp.join(args.output_dir_target, "target_B_" + args.savename + ".pt").replace('\\', '/'))
                    torch.save(best_netC, osp.join(args.output_dir_target, "target_C_" + args.savename + ".pt").replace('\\', '/'))
                
            netF.train()
            netB.train()

    writer.close()
    
    log_str = f"Best target accuracy: {acc_target_best:.2f}%\n"
    log_str += f"Finish Time: {str(datetime.datetime.now()).split('.')[0]}"
    print_log(log_str, args)

    netF.load_state_dict(best_netF)
    netB.load_state_dict(best_netB)
    netC.load_state_dict(best_netC)

    return netF, netB, netC


if __name__ == "__main__":
    """
        描述:
            在目标域进行Fine-Tune的代码, 将预训练的源域模型窃取出来后进行进一步的调整.
            在完成窃取过程后, 仅需要调整该代码中的模型载入路径.
        
        输入:
            数据集:
                目标域数据集:
                    args.target_dset_path => folder + args.dset + names[args.t]
            窃取模型保存目录:
                args.output_dir_extract => args.output_extract + args.dset + names[args.s]
        
        输出:
            目标域模型保存目录: 
                args.output_dir_target => args.output_target + args.dset + names[args.s] + names[args.t]
    """
    
    parser = argparse.ArgumentParser(description='BFDA-TRAIN-TARGET')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=[0], nargs='+', help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=2, help="max iterations")
    parser.add_argument('--interval', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-caltech', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain-net', 'domain-net-20', 'domain-net-40', 'domain-net-80', 'domain-net-160'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)     # 对特征提取器netF的学习率衰减
    parser.add_argument('--lr_decay2', type=float, default=1.0)     # 对BN层netB的学习率衰减

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output_extract', type=str, default='ckps/extract')
    parser.add_argument('--output_target', type=str, default='ckps/target')
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--no_DEPICT', type=int, default=1)    # 在目标域模型调优过程中, 除了使用加权聚类外, 是否还使用DEPICT算法进行标签精炼.
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

    # 数据集路径
    folder = './data/'
    args.target_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # 已窃取模型路径以及目标域模型保存路径
    args.src_names = '_'.join(names[s].upper() for s in args.s)
    args.tar_name = names[args.t].upper()
    args.output_dir_extract = osp.join(args.output_extract, args.dset, args.src_names).replace('\\', '/')
    args.output_dir_target = osp.join(args.output_target, args.dset, args.src_names + '_to_' + args.tar_name).replace('\\', '/')
    args.name = args.src_names + ' -> ' + args.tar_name

    if not osp.exists(args.output_dir_target):
        os.makedirs(args.output_dir_target)

    # 日志文件
    args.savename = 'par_' + str(args.cls_par)
    args.log_file = open(osp.join(args.output_dir_target, 'log_target_' + args.savename + '.txt').replace('\\', '/'), 'w')
    log_str = f"Start at: {str(datetime.datetime.now()).split('.')[0]}; Output Dir: {args.output_dir_target}"
    print_log(log_str, args)
    args.log_file.write(print_args(args)+'\n')
    args.log_file.flush()


    # 训练目标域模型
    train_target(args)