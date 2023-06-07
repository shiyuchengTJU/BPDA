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

from data_list import ImageList
import network, loss
from loss import CrossEntropyLabelSmooth

from utils import op_copy, lr_scheduler, cal_acc, image_train, image_test, print_args
from utils import get_lr, print_log

import time
import datetime
from tensorboardX import SummaryWriter

def data_load(args):
    '''
        载入所需数据.

    returns:
        dset_loaders (dict of Dataloader): {'src_train': ..., 'src_test': ..., 'target_list': [list of dataloader]}
    '''
    # 准备数据
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    # 1. 源域数据集
    src_txt = open(args.src_dset_path).readlines()
    if args.data_split == 'train_val':
        # 数据划分: 训练集:验证集 = 9:1
        data_size = len(src_txt)
        train_size = int(0.9 * data_size)
        src_train_txt, src_test_txt = torch.utils.data.random_split(src_txt, [train_size, data_size - train_size])
    else:
        # 所有数据作为训练集
        src_train_txt = src_txt

    dsets["src_train"] = ImageList(src_train_txt, transform=image_train())
    dset_loaders["src_train"] = DataLoader(dsets["src_train"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)
    if args.data_split == 'train_val':
        dsets["src_test"] = ImageList(src_test_txt, transform=image_test())
        dset_loaders["src_test"] = DataLoader(dsets["src_test"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False, pin_memory=True)
    
    # 2. 目标域数据集列表.
    dset_loaders["target_list"] = []
    for test_dset_path in args.test_dset_path_list:
        target_txt = open(test_dset_path).readlines()
        dsets["target"] = ImageList(target_txt, transform=image_test())
        dset_loaders["target_list"].append(DataLoader(dsets["target"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False, pin_memory=True))


    return dset_loaders

def train_source(args):
    '''
        训练源域数据
    '''
    # tensorboard日志目录
    tensorboard_log = osp.join(args.output_dir_src, 'tensorboard_log').replace('\\', '/')
    if os.path.exists(tensorboard_log):
        shutil.rmtree(tensorboard_log)
    writer = SummaryWriter(logdir=tensorboard_log)

    # 载入数据.
    dset_loaders = data_load(args)  # dset_loaders (dict of dataset): {"src_train": src_train_dset, "src_test": src_test_dset, "test": test_dset}
    
    # 设置主干网络
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # 载入在公共数据集上的预训练权重
    # if not args.pretrained_model is None:
    #     model_path = args.pretrained_model + '/source_F.pt'
    #     netF.load_state_dict(torch.load(model_path))
    #     model_path = args.pretrained_model + '/source_B.pt'
    #     netB.load_state_dict(torch.load(model_path))

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]     # 经过预训练的主干网ResNet的learning rate设为原本的1/10.
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_src_best = 0.0  # 目前模型在源域上最高的accuracy
    acc_target_avg_best = 0.0   # 目前在各个目标域上平均迁移性能最高的accuracy
    max_iter = args.max_epoch * len(dset_loaders["src_train"])
    interval_iter = max_iter // args.interval  # 每隔interval_iter就进行一次测试, 并保存最优的模型.
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    print_log(f"Training source model of {args.name_src}...", args)

    start = time.time()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["src_train"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)   # 调整学习率

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        # 采用平滑标签损失, 提高域泛化能力.
        # classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            # 源域性能
            if args.dset == "VISDA-C":
                ''' 对于VISDA-C数据集, 其评价指标是average class accuracy. '''
                acc_src_test, acc_list = cal_acc(dset_loaders["src_test"], netF, netB, netC, avg_class_acc=True)
                print(acc_list)
            else:
                acc_src_test, _, src_class_acc = cal_acc(dset_loaders["src_test"], netF, netB, netC, cal_class_acc=True)

            # 目标域平均迁移性能
            acc_target_avg_test = 0.0
            acc_target_dict = {}
            num_of_target = len(dset_loaders['target_list'])
            for t in range(num_of_target):
                if args.dset == "VISDA-C":
                    ''' 对于VISDA-C数据集, 其评价指标是average class accuracy. '''
                    acc_target_test, acc_target_class_list = cal_acc(dset_loaders["target_list"][t], netF, netB, netC, avg_class_acc=True)  # 这里只用了最后的平均准确率, 实际上还有各个类别的准确率.
                else:
                    acc_target_test, _, tgt_class_acc = cal_acc(dset_loaders["target_list"][t], netF, netB, netC, cal_class_acc=True)
                acc_target_dict[args.name_target_list[t]] = acc_target_test
                acc_target_avg_test += acc_target_test
            
            acc_target_avg_test /= num_of_target

            # 记录日志信息.
            # 学习率
            current_lr = get_lr(optimizer)['lr_0']
            writer.add_scalars('learning_rate', {'lr': current_lr}, global_step=iter_num)
            # 耗时
            elapsed = datetime.timedelta(seconds=time.time() - start)
            remain = elapsed / (iter_num + 1e-8) * (max_iter - iter_num)
            # 损失
            writer.add_scalars('classifier_loss', {'classifier_loss': classifier_loss}, global_step=iter_num)
            # 性能信息
            acc_target_dict["Average"] = acc_target_avg_test
            writer.add_scalars('target_test_accuracy', acc_target_dict, global_step=iter_num)

            # 总日志.
            log_str = f"Task: {args.name_src}, Iter:{iter_num}/{max_iter}, elapsed:{str(elapsed).split('.')[0]}, remain: {str(remain).split('.')[0]};\n"
            log_str += f'Average Target Accuracy = {acc_target_avg_test:4.2f}%, Source Accuracy = {acc_src_test:4.2f}%\n'
            log_str += f"acc_target_dict: {acc_target_dict}\n"
            log_str += f"classifier_loss: {classifier_loss}, learning_rate: {current_lr}\n"
            log_str += f"Source Class ACC: {src_class_acc}\n"
            log_str += f"Target Class ACC: {tgt_class_acc}\n"
            # if acc_target_class_list:
            #     log_str += f"acc_target_class_list: {acc_target_class_list}"    # NOTE: 只有VisDA-C才会用到这个.
            print_log(log_str, args)


            if acc_target_avg_test >= acc_target_avg_best:
                acc_target_avg_best = acc_target_avg_test
                acc_target_dict_best = acc_target_dict

                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

                # 训练过程中随时保存当前找到的最优的模型.
                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt").replace('\\', '/'))
                torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt").replace('\\', '/'))
                torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt").replace('\\', '/'))

            netF.train()
            netB.train()
            netC.train()

    writer.close()

    log_str = f"Best Model on {args.name_src}:\nAverage Target Acc: {acc_target_avg_best}, Source acc: {acc_src_test}\n"
    log_str += f"Target Accuracy Dict: {acc_target_dict_best}\n\n"
    log_str += f"Finish Time: {str(datetime.datetime.now()).split('.')[0]}"
    print_log(log_str, args)
    # 返回当前找到的最好的模型, 而不是最后的模型.
    netF.load_state_dict(best_netF)
    netB.load_state_dict(best_netB)
    netC.load_state_dict(best_netC)

    return netF, netB, netC

if __name__ == "__main__":
    """
        描述:
            训练单个源域的模型并在训练后测试其在其他域上的性能.
        
        数据集:
            数据集一级目录下需要存储描述文件{domain}_list.txt
            源域: args.src_dset_path => folder + args.dset + args.s
            目标域: args.test_dset_path => folder + args.dset + args.t

        输出:
            源域模型输出目录: args.output_dir_src => args.output + args.dset + args.name_src
            训练/测试日志输出目录 (args.log_file): args.output_dir_src => args.output + args.dset + args.name_src
    """

    parser = argparse.ArgumentParser(description='BFDA-TRAIN-SOURCE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=[1], nargs='+', help="target")
    parser.add_argument('--max_epoch', type=int, default=1, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-caltech', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain-net', 'domain-net-20', 'domain-net-40', 'domain-net-80', 'domain-net-160'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ckps/source')    # 模型保存以及日志输出的根目录.
    parser.add_argument('--data_split', type=str, default='train_val', choices=['full', 'train_val'])   # 对训练数据的数据划分方式.
    parser.add_argument('--interval', type=int, default=10)   # 训练过程中一共进行interval次测试.
    # parser.add_argument('--pretrained_model', type=str, default='ckps/pretrained_model/OID/VALIDATION')   # 所使用的预训练模型的路径

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
        print("尚未实现该数据集的代码.")
        exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # 源域数据集注释文件的路径.
    folder = './data/'  # 数据集存放的根目录.
    args.src_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'

    # 目标域数据集注释文件路径列表
    args.test_dset_path_list = [folder + args.dset + '/' + names[t] + '_list.txt' for t in args.t]

    # 训练后的源域模型以及训练日志的输出目录.
    args.name_src = names[args.s].upper()
    args.output_dir_src = osp.join(args.output, args.dset, args.name_src).replace('\\', '/')

    # 目标域名称
    args.name_target_list = [names[t].upper() for t in args.t]
    
    if not osp.exists(args.output_dir_src):
        os.makedirs(args.output_dir_src)

    # 日志文件.
    args.log_file = open(osp.join(args.output_dir_src, 'log_train.txt').replace('\\', '/'), 'w')
    current_time = datetime.datetime.now()
    log_str = f"Start Time: {str(current_time).split('.')[0]}; Output Dir: {args.output_dir_src}"
    print_log(log_str, args)
    args.log_file.write(print_args(args)+'\n')
    args.log_file.flush()

    # 训练源域模型
    train_source(args)