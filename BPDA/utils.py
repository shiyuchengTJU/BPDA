import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from data_list import ImageList
import loss
from sklearn.metrics import confusion_matrix


def op_copy(optimizer):
    ''' 将优化器的当前学习率保存到param_group['lr0']中. 其中, param_group['lr0']在SGD这个优化器中不存在, 是自定义的变量, 便于自行调整学习率. '''
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def get_lr(optimizer):
    ''' 返回当前学习率 '''
    lr_dict = {}
    lr_cnt = 0
    for param_group in optimizer.param_groups:
        lr_dict[f'lr_{lr_cnt}'] = param_group['lr']
        lr_cnt += 1

    return lr_dict

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    ''' 自定义的学习率调整策略. 衰减率越来越大. '''
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224):
    ''' 训练图像的transform '''
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
    ''' 测试图像的transform '''
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def cal_acc(loader, netF, netB, netC, avg_class_acc=False, cal_class_acc=False):
    ''' 计算单个网络模型在loader指定的数据集上的accuracy和entropy. '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_entropy = torch.mean(loss.Entropy(all_output)).cpu().data.item()   # 熵

    ''' 对于VisDA-C数据集, 其评价指标不是accuracy, 而是各个类别accuracy的平均值. '''
    if avg_class_acc:
        print("Computing Confusion Matrix......")
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc

    if cal_class_acc:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        # acc = ' '.join(aa)
        
        return accuracy * 100, mean_entropy, aa
    
    return accuracy * 100, mean_entropy

def print_args(args):
    ''' 打印args中的参数. '''
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s

def get_aggre_output(inputs, netF_list, netB_list, netC_list):
    ''' 获取多个网络模型对inputs的聚合输出 '''
    num_of_nets = len(netF_list)

    logits_list = []
    for i in range(num_of_nets):
        logits = netF_list[i](inputs)
        logits = netB_list[i](logits)
        logits = netC_list[i](logits)

        logits_list.append(logits)
    
    logits = torch.stack(logits_list, dim=0)    # [[B, C]. [B, C], ...] => [N, B, C]

    logits_aggre = torch.mean(logits, dim=0)    # [N, B, C] => [B, C]

    return logits_aggre

def get_aggre_output_hardlabel(inputs, netF_list, netB_list, netC_list, args):
    '''
        根据多个网络模型的logits, 求每个模型预测的hard label.
        然后利用hard label计算logits, 最后使用soft label来计算.
    '''
    num_of_nets = len(netF_list)

    logits_list = []
    for i in range(num_of_nets):
        logits = netF_list[i](inputs)
        logits = netB_list[i](logits)
        logits = netC_list[i](logits)
        ''' FIXME: 这里求label而不是求logits '''
        _, label = torch.max(logits, dim=1)     # [batch_size, num_classes] -> [batch_size]
        label = F.one_hot(label, args.class_num)    # one-hot的硬标签.
        logits_list.append(label)
    
    logits = torch.stack(logits_list, dim=0)    # [[B, C]. [B, C], ...] => [N, B, C]
    # logits = logits.transpose(0, 1)     # [N, B] -> [B, N]
    logits = logits.sum(dim=0)  # 将硬标签聚合. 相当于变成了多标签分类问题.

    ''' 使用多个模型的hard label来构成soft label. '''
    logits[logits == 0] = -10
    logits = F.softmax(logits.float(), dim=1)
    

    return logits

    ''' 使用出现次数最多的label作为label. '''
    # maxlabels_aggre = []
    # logits_list = logits.detach().cpu().numpy().tolist()
    # for logit in logits_list:
    #     max_label = max(logit, key=logit.count)     # 将出现次数最多的标签作为最终的标签.
    #     maxlabels_aggre.append(max_label)
    # # logits_aggre = torch.mean(logits, dim=0)    # [N, B, C] => [B, C]
    # maxlabels_aggre = torch.tensor(maxlabels_aggre).long()

    # return maxlabels_aggre

def get_aggre_feature(inputs, netF_list, netB_list):
    ''' 获取多个网络模型对inputs的聚合输出 '''
    num_of_nets = len(netF_list)

    logits_list = []
    for i in range(num_of_nets):
        logits = netF_list[i](inputs)
        logits = netB_list[i](logits)
        # logits = netC_list[i](logits)

        logits_list.append(logits)
    
    logits = torch.stack(logits_list, dim=0)    # [[B, C]. [B, C], ...] => [N, B, C]

    logits_aggre = torch.mean(logits, dim=0)    # [N, B, C] => [B, C]

    return logits_aggre

def cal_acc_aggre(loader, netF_list, netB_list, netC_list, avg_class_acc=False):
    ''' 计算网络模型的聚合模型在loader指定的数据集上的accuracy和entropy. '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            # 这里使用聚合模型的输出
            outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)     # 多模型聚合输出

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_entropy = torch.mean(loss.Entropy(all_output)).cpu().data.item()   # 熵

    ''' 对于VisDA-C数据集, 其评价指标不是accuracy, 而是各个类别accuracy的平均值. '''
    if avg_class_acc:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
   
    return accuracy * 100, mean_entropy

def obtain_label(loader, netF, netB, netC, args, get_refined_accuracy=False, cal_class_acc=False):
    '''
    求netF, netB, netC构成的网络对loader中的数据集的预测标签,
    并使用加权聚类以及DEPICT算法(可选)来对标签进行精炼.

    returns:
        pred_label (np.ndarray): 使用加权聚类假设以及DEPICT算法(可选)精炼后的标签.
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features = netB(netF(inputs))
            outputs = netC(features)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    entropy = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - entropy / np.log(args.class_num)     # 这个没有实际用途
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cal_class_acc:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc_before = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc_before = acc_before.mean()
        aa_before = [str(np.round(i, 2)) for i in acc_before]

    ''' 使用DEPICT Algorithm进行标签调整(可选) '''
    if not args.no_DEPICT:
        print_log("<<< 使用DEPICT Algorithm进行标签精炼 >>>", args)
        all_output = DEPICT_Algorithm(all_output)

    ''' 使用加权聚类进行标签调整 '''
    if args.distance == 'cosine':
        all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()

    all_features = all_features.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()  # probability: [N, num_classes]
    initc = aff.transpose().dot(all_features)   # 
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])    # initc: 聚类中心, [num_classes, feature_dim]
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_features, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_features, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_features)
    if cal_class_acc:
        matrix = confusion_matrix(all_label, torch.squeeze(torch.from_numpy(pred_label)).float())
        acc_after = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc_after = acc_after.mean()
        aa_after = [str(np.round(i, 2)) for i in acc_after]
    
    log_str = f"Class Accuracy Before: {aa_before}\n"
    log_str += f"Class Accuracy After : {aa_after}\n"
    log_str += 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)


    args.log_file.write(log_str + '\n')
    args.log_file.flush()
    print(log_str)

    # 返回精炼后的accuracy, 用于日志信息
    if get_refined_accuracy:
        return pred_label, acc
    
    return pred_label.astype('int')

def obtain_label_without_refinement(loader, netF, netB, netC, args):
    '''
    求netF, netB, netC构成的网络对loader中的数据集的预测标签.

    returns:
        pred_label (np.ndarray):
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()     # data: [image, label, index]
            inputs = data[0].cuda()
            features = netB(netF(inputs))
            outputs = netC(features)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    entropy = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - entropy / np.log(args.class_num)     # 这个没有实际用途
    _, predict = torch.max(all_output, 1)

    predict = predict.detach().cpu().numpy()

    return predict.astype('int')

def obtain_label_without_refinement_aggre(loader, netF_list, netB_list, netC_list, args):
    '''
    求netF, netB, netC构成的网络对loader中的数据集的预测标签.

    returns:
        pred_label (np.ndarray):
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        
        for _ in range(len(loader)):
            data = iter_test.next()     # data: [image, label, index]
            inputs = data[0].cuda()

            # 模型聚合输出
            outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)

            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)  # [num_data, num_classes]
    entropy = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    _, predict = torch.max(all_output, 1)   # [num_data]

    predict = predict.detach().cpu().numpy()

    return predict.astype('int')

def obtain_label_without_refinement_aggre_hard_label(loader, netF_list, netB_list, netC_list, args):
    '''
    求netF, netB, netC构成的网络对loader中的数据集的预测标签.
    NOTE: 聚合过程不是使用各个模型输出的logits进行聚合, 而是使用各个模型输出的hard label的一致性进行聚合.

    returns:
        pred_label (np.ndarray):
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        
        for _ in range(len(loader)):
            data = iter_test.next()     # data: [image, label, index]
            inputs = data[0].cuda()

            ''' NOTE: 模型聚合输出的硬标签. '''
            outputs = get_aggre_output_hardlabel(inputs, netF_list, netB_list, netC_list, args)     # outputs: [batch_size, num_classes]

            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    if not args.no_DEPICT:
        ''' 使用DEPICT进行标签精炼 '''
        print_log("<<< Using DEPICT Algorithm to refine the HARD labels >>>", args)
        all_output = DEPICT_Algorithm(all_output)

    predict = all_output

    predict = predict.detach().cpu().numpy()

    return predict.astype('float')

def obtain_softlabel_without_refinement_aggre(loader, netF_list, netB_list, netC_list, args):
    '''
    求netF, netB, netC构成的网络对loader中的数据集的预测"软"标签.   NOTE: 软标签

    returns:
        pred_label (np.ndarray): 各个源域模型聚合的软标签. 本质上就是概率的均值.
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        
        for _ in range(len(loader)):
            data = iter_test.next()     # data: [image, label, index]
            inputs = data[0].cuda()

            # 模型聚合输出
            outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)

            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    entropy = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    ''' FIXME: 这里不求标签, 直接返回预测的概率值. '''
    # _, predict = torch.max(all_output, 1)
    predict = all_output

    predict = predict.detach().cpu().numpy()
    if args.use_softlabel:
        return predict.astype('float')

    return predict.astype('int')

def DEPICT_Algorithm(prob):
    '''
    使用DEPICT Algorithm对整个数据集上的标签进行精炼. https://arxiv.org/abs/1704.06327
    q_{i k}=\frac{p_{i k} /\left(\sum_{i^{\prime}} p_{i^{\prime} k}\right)^{\frac{1}{2}}}{\sum_{k^{\prime}} p_{i k^{\prime}} /\left(\sum_{i^{\prime}} p_{i^{\prime} k^{\prime}}\right)^{\frac{1}{2}}}

    Args:
        prob (Tensor): 整个数据集上, 神经网络对每一个类预测的概率值, 即logits经过Softmax后的值. [N, num_classes]
    
    Returns:
        prob_refinement (Tensor): 经过精炼后的概率值.
    '''
    
    child = prob / torch.sqrt(prob.sum(dim=0))  # 计算分子
    parent = child.sum(dim=1)   # 计算分母

    # 精炼后的标签
    prob_refinement = child / parent.unsqueeze(1)

    return prob_refinement

def obtain_label_with_DEPICT_aggre(loader, netF_list, netB_list, netC_list, args):
    '''
    求netF_list, netB_list, netC_list构成的网络对loader中的数据集的预测标签, 并将此标签经过DEPICT Algorithm进行精炼.

    returns:
        pred_label (np.ndarray): 经过DEPICT精炼后的标签.
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        
        for _ in range(len(loader)):
            data = iter_test.next()     # data: [image, label, index]
            inputs = data[0].cuda()

            # 模型聚合输出
            outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)

            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)  # logits -> probability
    
    ''' 使用DEPICT进行标签精炼 '''
    all_output = DEPICT_Algorithm(all_output)
    
    _, predict = torch.max(all_output, 1)

    predict = predict.detach().cpu().numpy()

    return predict.astype('int')

def imshow_and_save(img_tensor, mean=0, std=1, title=None, save_path=None):
    """
    对单张PyTorch Tensor图像进行可视化. 可视化的步骤可以分为如下3步:
    (1) 将Tensor图像转换为Numpy图像: [C, H, W] => [H, W, C]
    (2) 解除归一化(Denormalize)
    (3) 使用matplotlib.pyplot显示图像

    注意: 
    (1) 需要在转换为Numpy图像后再进行归一化, 这涉及到Numpy的broadcasting机制.
    (2) 该imshow()函数与plt.imshow()一样只负责绘图, 而不负责显示. 需要显示则自行调用plt.show()

    @parameters:
        img_tensor (Tensor): 单张PyTorch图像, 类型为Tensor, 尺寸为: [3, H, W]
        title (str): 图像的标题
        mean (int or list or numpy.ndarray): 进行归一化操作时使用的均值. 当是list或者numpy.ndarray时, len或者shape为[3]
        std (int or list or numpy.ndarray): 进行归一化操作时使用的标准差. 当是list或者numpy.ndarray时, len或者shape为[3]
    
    @returns:
        None
    """
    # plt.ioff()
    # 1. 将Tensor图像转换为Numpy图像.
    img_numpy = img_tensor.numpy().transpose((1, 2, 0))

    # 2. 解除归一化(denormalize)
    img_numpy = img_numpy * std + mean
    img_numpy = img_numpy.clip(min=0, max=1)

    # 3. 使用matplotlib.pyplot绘制numpy图像
    
    if title is not None:
        plt.title(title)
        plt.imshow(img_numpy)
        plt.show()
    
    if save_path is not None:
        plt.imsave(save_path, img_numpy)

def print_log(log_str, args):
    ''' 将log_str写入日志文件log_file, 并输出到控制台. '''
    args.log_file.write(log_str + '\n')
    args.log_file.flush()
    print(log_str)

def label_distribution_vis(labels, image_path, class_num):
    ''' 将标签分布使用pyplot进行可视化并将图片保存到指定路径. '''
    plt.hist(labels, np.arange(class_num + 1) - 0.5, histtype='bar', rwidth=0.8)
    plt.title("label distribution")
    plt.xlabel("label")
    plt.ylabel("num of label")
    
    plt.savefig(fname=image_path)

def get_cluster_center(loader, netF, netB, netC, args):
    '''
    求netF, netB, netC构成的网络对loader中的数据集的预测标签.

    returns:
        initc (np.array): 聚类中心. 
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features = netB(netF(inputs))
            outputs = netC(features)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    entropy = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - entropy / np.log(args.class_num)     # 这个没有实际用途
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()

    all_features = all_features.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_features)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])    # initc: 聚类中心, [num_classes, feature_dim]
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_features, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_features, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_features)
    
    return initc

def ifgsm(netF, netB, netC, adv_image, source_label, args, adv_step=5, adv_lr=80):
    '''
    iFGSM算法: 进行针对性错分 -- 使用目标域模型打出来的标签以及源域模型打出来的标签作为指标, 优化第三方数据, 使其分布向目标域数据靠近.

    Args:
        netF, netB, netC (nn.Module): 网络模型.
        adv_image (Tensor): 需要优化的图像. [B, C, H, W]
        source_label (Tensor): 源域模型打的伪标签.
        adv_step (int): 进行对抗攻击的迭代步数.
        adv_lr (float): 对抗训练的学习率.
    
    Returns:
        adv_image (Tensor): [B, C, H, W]. 返回的对抗样本.
    '''
    # FIXME: random source label
    # source_label = torch.tensor([4., 1, 8, 7]).cuda()

    _min = -3
    _max = 3
    noise_min = -3
    noise_max = 3

    batch_size = len(adv_image)
    source_image = adv_image   # 原始图像
    
    ''' 将模型转为eval()模式 '''
    netF.eval()
    netB.eval()

    # 将source_label转为one-hot编码
    # _, target_label = torch.max(nn.Softmax(dim=1)(netC(netB(netF(adv_image)))), dim=1)
    # if (target_label != source_label).sum().item() > 0:
    #     print()
    # print(f"source_label: {source_label}")
    source_label = onehot_encode(source_label, num_classes=args.class_num).cuda()
    ''' 开始进行adv_step次迭代攻击 '''
    for adv_counter in range(adv_step):
        ''' adv_image始终维持无梯度的状态. '''
        # 每次都用temp_adv_image查询
        temp_adv_image = adv_image.clone()

        # 对抗样本优化器
        # adv_opt = optim.Adam([{'params': temp_adv_image, 'lr': adv_lr}])
        temp_adv_image.requires_grad = True
        
        # 计算当前图像在目标域上进行查询得到的标签
        target_logits = netC(netB(netF(temp_adv_image)))
        target_logits = nn.Softmax(dim=1)(target_logits)

        #计算 KL Divergence loss, 并对原始图像进行优化.
        kl_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(target_logits), source_label)
        # loss = nn.CrossEntropyLoss()
        # cls_loss = loss(target_logits, source_label.long())
        # print(kl_loss)
        
        kl_loss.backward()
        # print(cls_loss)
        # cls_loss.backward()

        grad_l2norm = torch.sqrt((temp_adv_image.grad ** 2).view(batch_size, -1).sum(dim=1))
        grad = temp_adv_image.grad / grad_l2norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        temp_adv_image.data = temp_adv_image.data - adv_lr * grad

        # grad_l1norm = torch.abs(temp_adv_image.grad).view(batch_size, -1).sum(dim=1)
        # print(adv_lr * (temp_adv_image.grad / grad_l1norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)))
        # temp_adv_image.data = temp_adv_image.data - adv_lr * (temp_adv_image.grad / grad_l1norm.unsqueeze(1).unsqueeze(2).unsqueeze(3))


        temp_adv_image.grad.zero_()     # 清空图像梯度
        # adv_opt.step()


        # 根据对抗样本的优化结果, 向原图添加噪声.
        temp_adv_image = temp_adv_image.detach()
        temp_adv_image.requires_grad = False
        noise = torch.clamp(temp_adv_image - adv_image, noise_min, noise_max)
        adv_image = torch.clamp(noise + adv_image, _min, _max)

        # print(adv_image)

    ''' 恢复模型状态 '''
    netF.train()
    netB.train()

    # 最终的噪声.
    noise = adv_image - source_image
    # noise_norm = torch.norm(noise, p=2)
    # print(f"noise_norm: {noise_norm}")
    # target_logits = netC(netB(netF(adv_image)))
    # target_pred = F.softmax(target_logits, dim=1)
    # values, ind = torch.max(target_pred, dim=1)
    # print(f"values: {values}    ind: {ind}")
    # print(f"target label: {ind}")

    for i in range(batch_size):
        imshow_and_save(noise.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'noise_{i}.jpg')
        imshow_and_save(adv_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'adv_image_{i}.jpg')
    # print(f'{noise_norm}')

    return adv_image

def ifgsm_feature(netF_target, netB_target, adv_image, target_image, adv_step=5, adv_lr=80, visualization=False):
    '''
    iFGSM算法: 通过计算第三方数据和目标域数据经过目标域模型提取的特征在特征空间上的KL散度, 优化第三方数据, 使其分布与目标域数据分布接近.

    Args:
        netF_target, netB_target, netC_target (nn.Module): 目标域网络模型.
        adv_image (Tensor): 需要优化的第三方数据的图像. [B, C, H, W]
        target_image (Tensor): 目标域数据的图像. [B, C, H, W]
        adv_step (int): 进行对抗攻击的迭代步数.
        adv_lr (float): 对抗训练的学习率.
    
    Returns:
        adv_image (Tensor): [B, C, H, W]. 返回的对抗样本.
    '''
    _min = -3
    _max = 3
    noise_min = -3
    noise_max = 3

    batch_size = len(adv_image)
    source_image = adv_image   # 原始图像

    ''' 在提取特征之前模型先转为eval()模式 '''
    netF_target.eval()
    netB_target.eval()
    # 计算目标域模型提取的特征
    target_feature = netB_target(netF_target(target_image)).detach()
    
    ''' 开始进行adv_step次迭代攻击 '''
    for adv_counter in range(adv_step):
        ''' adv_image始终维持无梯度的状态. '''
        # 每次都用temp_adv_image查询
        temp_adv_image = adv_image.clone()

        temp_adv_image.requires_grad = True
        
        # 计算对抗样本在当前目标域模型提取的特征.
        adv_feature = netB_target(netF_target(temp_adv_image))

        #计算 KL Divergence loss, 并对原始图像进行优化.
        kl_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(adv_feature), target_feature)
        
        kl_loss.backward()

        grad_l2norm = torch.sqrt((temp_adv_image.grad ** 2).view(batch_size, -1).sum(dim=1))
        grad = temp_adv_image.grad / grad_l2norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        temp_adv_image.data = temp_adv_image.data - adv_lr * grad

        temp_adv_image.grad.zero_()     # 清空图像梯度

        # 根据对抗样本的优化结果, 向原图添加噪声.
        temp_adv_image = temp_adv_image.detach()
        temp_adv_image.requires_grad = False
        noise = torch.clamp(temp_adv_image - adv_image, noise_min, noise_max)
        adv_image = torch.clamp(noise + adv_image, _min, _max)
    
    ''' 恢复网络的状态 '''
    netF_target.train()
    netB_target.train()

    # 最终的噪声.
    noise = adv_image - source_image
    # noise_norm = torch.norm(noise, p=2)
    # print(f"noise_norm: {noise_norm}")
    # target_logits = netC(netB(netF(adv_image)))
    # target_pred = F.softmax(target_logits, dim=1)
    # values, ind = torch.max(target_pred, dim=1)
    # print(f"values: {values}    ind: {ind}")
    # print(f"target label: {ind}")

    if visualization:
        for i in range(batch_size):
            imshow_and_save(noise.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'noise_{i}.jpg')
            imshow_and_save(adv_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'adv_image_{i}.jpg')
            imshow_and_save(source_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'source_image_{i}.jpg')
            imshow_and_save(target_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'target_image_{i}.jpg')
    # print(f'{noise_norm}')

    return adv_image

def ifgsm_feature_prototype(netF_target, netB_target, adv_image, label, prototypes, adv_step=5, adv_lr=80, visualization=False):
    '''
    iFGSM算法: 通过计算第三方数据和目标域数据经过目标域模型提取的特征在特征空间上的KL散度, 优化第三方数据, 使其分布与目标域数据分布接近.

    Args:
        netF_target, netB_target, netC_target (nn.Module): 目标域网络模型.
        adv_image (Tensor): 需要优化的第三方数据的图像. [B, C, H, W]
        label (Tensor): 第三方图像的伪标签. [B]
        prototypes (Tensor): 所有类别的原型. [n_classes, n_feature]
        adv_step (int): 进行对抗攻击的迭代步数.
        adv_lr (float): 对抗训练的学习率.
    
    Returns:
        adv_image (Tensor): [B, C, H, W]. 返回的对抗样本.
    '''
    _min = -3
    _max = 3
    noise_min = -3
    noise_max = 3

    batch_size = len(adv_image)
    source_image = adv_image   # 原始图像

    ''' 在提取特征之前模型先转为eval()模式 '''
    netF_target.eval()
    netB_target.eval()
    ''' 需要优化的方向 '''
    target_feature = prototypes[label]
    
    ''' 开始进行adv_step次迭代攻击 '''
    for adv_counter in range(adv_step):
        ''' adv_image始终维持无梯度的状态. '''
        # 每次都用temp_adv_image查询
        temp_adv_image = adv_image.clone()

        temp_adv_image.requires_grad = True
        
        # 计算对抗样本在当前目标域模型提取的特征.
        adv_feature = netB_target(netF_target(temp_adv_image))

        #计算 KL Divergence loss, 并对原始图像进行优化.
        kl_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(adv_feature), target_feature)
        
        kl_loss.backward()

        grad_l2norm = torch.sqrt((temp_adv_image.grad ** 2).view(batch_size, -1).sum(dim=1))
        grad = temp_adv_image.grad / grad_l2norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        temp_adv_image.data = temp_adv_image.data - adv_lr * grad

        temp_adv_image.grad.zero_()     # 清空图像梯度

        # 根据对抗样本的优化结果, 向原图添加噪声.
        temp_adv_image = temp_adv_image.detach()
        temp_adv_image.requires_grad = False
        noise = torch.clamp(temp_adv_image - adv_image, noise_min, noise_max)
        adv_image = torch.clamp(noise + adv_image, _min, _max)
    
    ''' 恢复网络的状态 '''
    netF_target.train()
    netB_target.train()

    # 最终的噪声.
    noise = adv_image - source_image
    # noise_norm = torch.norm(noise, p=2)
    # print(f"noise_norm: {noise_norm}")
    # target_logits = netC(netB(netF(adv_image)))
    # target_pred = F.softmax(target_logits, dim=1)
    # values, ind = torch.max(target_pred, dim=1)
    # print(f"values: {values}    ind: {ind}")
    # print(f"target label: {ind}")

    # if visualization:
    #     for i in range(batch_size):
    #         imshow_and_save(noise.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'noise_{i}.jpg')
    #         imshow_and_save(adv_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'adv_image_{i}.jpg')
    #         imshow_and_save(source_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'source_image_{i}.jpg')
    #         # imshow_and_save(target_image.detach().cpu()[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], title=None, save_path=f'target_image_{i}.jpg')
    # print(f'{noise_norm}')

    return adv_image

def obtain_batch_label_aggre(images, netF_list, netB_list, netC_list):
    '''
    求netF, netB, netC构成的网络对一个batch的图像的预测标签.

    returns:
        pred_label (np.ndarray):
    '''
    inputs = images.cuda()

    # 模型聚合输出
    outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)
    outputs = nn.Softmax(dim=1)(outputs)
    _, predict = torch.max(outputs, 1)

    return predict

def onehot_encode(label, num_classes):
    '''
    将输入的label编码为one-hot向量

    Args:
        label (Tensor): 标签. [B]
        num_classes (int): 总类别数量
    
    Returns:
        onehot (Tensor): label对应的onehot编码. [B, num_classes]
    '''
    batch_size = label.size(0)
    onehot = torch.zeros(batch_size, num_classes)
    onehot[torch.arange(batch_size), label.long()] = 1

    return onehot

def valid_cluster_performance(loader, netF, netB, netC, args):
    '''
    求聚类中心打出来的标签与均匀分布的KL散度与其真实标签的关系.

    returns:
        initc (np.array): 聚类中心. 
    '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features = netB(netF(inputs))
            outputs = netC(features)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    entropy = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - entropy / np.log(args.class_num)     # 这个没有实际用途
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()

    all_features = all_features.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_features)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])    # initc: 聚类中心, [num_classes, feature_dim]
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_features, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        # dd: 每个数据样本的特征到聚类中心的距离, [N, n_features]
        dd = cdist(all_features, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]   # pred_label: 使用聚类打出来的标签
    
    ''' 计算每个样本到聚类中心的距离与均匀分布的KL散度 '''
    dd = torch.from_numpy(dd)
    uniform = dd.sum(dim=1).unsqueeze(dim=1).expand_as(dd) / K

    KL_div_loss = []
    for i in range(dd.size(0)):
        KL_div_loss.append(nn.KLDivLoss(reduction="batchmean")(torch.log(dd[i]), uniform[i]).item())
    
    index = [i for i in range(dd.size(0))]
    
    is_true_label = (torch.from_numpy(pred_label) == all_label.long())
    
    temp = [(kl, true_label.item(), ind) for kl, true_label, ind in zip(KL_div_loss, is_true_label, index)]
    temp.sort()

    half = len(temp) // 2
    low_kl = [t[1] for t in temp[:half]]
    high_kl = [t[1] for t in temp[half:]]

    low_kl = torch.tensor(low_kl).sum()
    high_kl = torch.tensor(high_kl).sum()

    low_index = [t[2] for t in temp[:half]]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_features)
    
    return low_index

def cal_probability_aggre(loader, netF_list, netB_list, netC_list, get_all_labels=False):
    ''' 计算网络模型的聚合模型在loader指定的数据集上的分类概率 '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            # 这里使用聚合模型的输出
            outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)     # 多模型聚合输出

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_entropy = torch.mean(loss.Entropy(all_output)).cpu().data.item()   # 熵
    
    if get_all_labels:
        return accuracy * 100, all_label, all_output
    
    return accuracy * 100, mean_entropy, all_output

def cal_logits_aggre(loader, netF_list, netB_list, netC_list):
    ''' 计算网络模型的聚合模型在loader指定的数据集上的分类概率 '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            # 这里使用聚合模型的输出
            outputs = get_aggre_output(inputs, netF_list, netB_list, netC_list)     # 多模型聚合输出

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # all_output = nn.Softmax(dim=1)(all_output)    # 返回的不是 probablity, 而是 logits.
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_entropy = torch.mean(loss.Entropy(all_output)).cpu().data.item()   # 熵
   
    return accuracy * 100, mean_entropy, all_output

def get_prototypes(loader, netF, netB, netC, args, avg_class_acc=False):
    ''' 计算每个类别的原型 '''
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features = netB(netF(inputs))    # 特征层: 256
            outputs = netC(features)

            if start_test:
                all_features = features
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    probs, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_entropy = torch.mean(loss.Entropy(all_output)).cpu().data.item()   # 熵

    ''' 计算各个类别的原型 '''
    prototypes = []
    for c in range(args.class_num):
        c_ind = (predict == c)
        c_feature = all_features[c_ind]
        c_prob = probs[c_ind].cuda()
        prototype = (c_feature * c_prob[:, None]).sum(dim=0) / c_feature.size(0)  # 特征均值作为原型中心.
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes, dim=0)
    
    ''' 对于VisDA-C数据集, 其评价指标不是accuracy, 而是各个类别accuracy的平均值. '''
    if avg_class_acc:
        print("Computing Confusion Matrix......")
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc

    ''' 利用Prototypes来计算伪标签的准确率 '''
    # 根据特征和原型之间的余弦相似度计算准确率.
    # cal_acc_by_prototypes(prototypes, all_features, all_label)
    
    return prototypes, accuracy * 100

def cal_acc_by_prototypes(prototypes, all_features, all_labels):
    '''
        使用原型和所有数据的特征计算准确率.
        prototypes (Tensor): 原型. [B, n_features]
        all_features (Tensor): 所有的特征. [num_of_data, n_features]
    '''
    # 归一化
    prototypes = F.normalize(prototypes, p=2, dim=1)
    all_features = F.normalize(all_features, p=2, dim=1)

    # 计算余弦相似度
    cos_distance = all_features.mm(prototypes.t())

    # 直接使用L2距离来算准确率


    # 根据余弦相似度分配伪标签
    _, pseudo_labels = torch.max(cos_distance, dim=1)

    acc = (pseudo_labels == all_labels.cuda()).sum() / len(pseudo_labels)

    print(f"Acc by Prototype: {acc}")


