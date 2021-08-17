import argparse
from PIL import Image
import os
import sys
import cv2
import json
import time
import shutil
import logging
import numpy as np

import matplotlib.cm as cm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from dataset_thyroid import Thyroid
# from model import get_model

from model import model_new, model3, model5, model_new, depthwise_model,depthwise_model2,depthwise_model1
# from model import model3_new as model3
from image_loader import TripletImageLoader, MetaLoader
from torch import nn
import torch
from torch.nn import functional as F

# Command Line Argument Parser
parser = argparse.ArgumentParser(description='Attribute-Specific Embedding Network')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.000001, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='disable CUDA training')  # 表示若命令行触发 --cuda => cuda是true 反之cuda是false
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='unet', type=str,
                    help='name of experiment')
parser.add_argument('--num_triplets', type=int, default=2000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--test_num_triplets', type=int, default=300, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--dim_embed', type=int, default=1024, metavar='N',
                    help='dimensions of embedding (default: 1024)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--visdom_port', type=int, default=4655, metavar='N',
                    help='visdom port')
parser.add_argument('--data_path', default="./data_new", type=str,
                    help='path to data directory')
parser.add_argument('--dataset', default="thyroid", type=str,
                    help='name of dataset')
parser.add_argument('--model', default="res", type=str,
                    help='model to load')
parser.add_argument('--step_size', type=int, default=1, metavar='N',
                    help='learning rate decay step size')
parser.add_argument('--decay_rate', type=float, default=0.985, metavar='N',
                    help='learning rate decay rate')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.set_defaults(test=False)
parser.set_defaults(visdom=False)


# parser.set_defaults(resume='./runs/res2/model_best_loss.pth')
def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    arr = torch.cat((arr1, arr2), dim=0)

    return arr


def new_loss(out, lab, loss_func):
    if len(lab) == 0:
        return 0
    else:
        a = loss_func(out, lab)
        print("a ", a)
        return a


def cross_entropy(input_, target, reduction='elementwise_mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    res = -target * logsoftmax(input_)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    # margin = 30 2 4(放大倍数)
    def __init__(self, margin=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        print('output1', output1.shape)
        # print('output2', output2)
        # print(torch.sum(torch.abs(output1 - output2),dim=-1))
        # s1 = F.cosine_similarity(output1, output2)
        # print(s1)
        # s1 =torch.mean(s1)
        # s1 = torch.mean(torch.sum(torch.abs(output1 - output2), dim=-1))
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # l1_distance =
        print("euclidean_distance:", euclidean_distance)
        # loss_contrastive = torch.mean((1-label) * euclidean_distance+
        #                               (label) * torch.clamp(self.margin - euclidean_distance, min=0.0))
        loss_contrastive = torch.mean((1 - label) * euclidean_distance +
                                      (label) * torch.clamp(self.margin - euclidean_distance, min=0.0))

        # loss_contrastive = torch.max((1-label) * torch.pow(euclidean_distance, 2)) +\
        #                               torch.min((label) * torch.clamp(self.margin - euclidean_distance, min=0.0))
        print("loss_contrastive:", loss_contrastive)
        # print("loss_contrastive:",loss_contrastive.size(0))
        return loss_contrastive


class focal_loss(nn.Module):
    def __init__(self, gamma=1.5, alpha=1, size_average=True):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        # print(logits.shape[0])
        # print(logits.shape[1])
        # print(logits.shape[2])
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """

        batch_size = logits.size(0)
        labels_length = logits.size(1)
        # seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        # print(new_label)

        label_onehot = torch.zeros([batch_size, labels_length]).cuda().scatter_(1, new_label, 1)
        # print(label_onehot)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * pt
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


def train(train_loader, model, loss_func, criterion, optimizer, epoch):
    total = 0
    acc1 = 0
    train_loss, train_loss2, train_loss3 = 0, 0, 0

    # train_loss1,train_loss2,train_loss3 = 0,0,0
    # print("len(train_loader.dataset):", len(train_loader.dataset))
    model.train()

    for batch_idx, (data1, label1, data2, label2) in enumerate(train_loader):

        if args.cuda:
            # data1,label1 = data1.cuda(),label1.cuda()
            data1, data2 = data1.cuda(), data2.cuda(),
            label1, label2 = label1.cuda(), label2.cuda()
        # print('{}:{}'.format(batch_idx, data1.size(0)))
        # print(label1)

        optimizer.zero_grad()

        # new
        # inputs, targets_a, targets_b, lam = mixup_data(data1, label1, data2,label2,args.alpha,  args.cuda)
        # inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        # res1, res2, a, b = model(inputs, inputs)
        #
        # loss_func1 = mixup_criterion(targets_a, targets_b, lam)
        # losst = loss_func1(loss_func, res1)
        # _, predicted = torch.max(res1.data, 1)
        #
        # acc1 += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()
        res1, res2, a, b = model(data1, data2)

        label = torch.cat((label1, label2), 0)
        out = torch.cat((res1, res2), 0)
        _, pred = torch.max(out, 1)
        acc1 += torch.sum(pred == label).item()
        losst = loss_func(out, label)

        # print("losst",losst)

        # a1 = label.unsqueeze(dim=1).cpu().data
        # one_hot = torch.zeros(data1.size(0)*2, 5).scatter_(1, a1, 1)
        # print(one_hot)
        # s = torch.tensor([[1,1.2,1,1,1]])
        # t = torch.mul(one_hot, s).cuda()
        # t.requires_grad = True
        # losst = cross_entropy(out, t)

        # print(out.requires_grad)
        # print(out_t)
        # label = torch.cat((label1, label2, label3), 0)

        # loss1 = loss_func(out, label)

        # print("out", out)
        # print("label", label)
        # print("pred", pred)
        # print("loss1", losst)5
        # total = total + data1.size(0)

        train_loss += losst.item() * data1.size(0)

        # loss1 = loss_func(out1,label1)
        # _,pred = torch.max(out1,1)
        # print('label1',label1)
        # print("pred",pred)
        total = total + data1.size(0)
        # acc1 += torch.sum(pred == label1).item( )
        # train_loss1 += loss1.item()*data1.size(0)
        # print('loss1',loss1)

        # print("vector1:",vector1.size(0))
        # print("vector1:",vector1.size(1))

        # _, _, out2 = model(data2)
        # loss2 = loss_func(out2, label2)
        #
        # _, pred = torch.max(out2, 1)
        # acc1 += torch.sum(pred == label2).item()
        # train_loss2 += loss2.item() * data2.size(0)

        # loss3 = (loss1+loss2)/2

        # optimizer.zero_grad()
        # compute similarity
        # sim_a, sim_b = tnet(data1, data2)

        loss_double1 = criterion(a, b, 1)  # 不同类
        train_loss2 += loss_double1 * data1.size(0)
        # loss_double2 = criterion(a, b, 0)  # 不同类
        # train_loss3 += loss_double2 * data1.size(0)

        # #loss = loss1+loss2
        # print('loss1',loss1)
        # # print('loss2', loss2)
        # print('loss_double', loss_double)
        # loss = 0.5*(loss1 + loss2) + loss_double*2

        # target = torch.FloatTensor(fea1.size()).fill_(-1)
        # if args.cuda:
        #     target = target.cuda()
        # loss_triplet1 = criterion(vector1, vector2, 1)
        # loss_triplet2 = criterion(vector1, vector3, 0)
        # loss = loss_triplet1 +loss_triplet2

        # loss_triplet = criterion(fea1, fea2, target)
        # train_loss3 += loss_triplet * data1.size(0)
        # loss = loss_triplet
        loss = losst + loss_double1
        # loss = losst
        # print(loss.requires_grad)
        # train_loss += loss*data1.size(0)

        loss.backward()
        optimizer.step()

        # compute gradient and do optimizer step

    accs = (float(acc1)) / (total * 2)
    # losses = (train_loss) / total
    losses = (train_loss + train_loss2) / total
    # accs =acc1/total
    # losses = (train_loss1)/total

    logger.info('Train Epoch: {} \t'
                'TrainLoss: {:.4f} \t'
                'TrainAcc: {:.2f}% '.format(
        epoch, losses, 100. * accs))
    # torch.cuda.empty_cache()
    return format(losses, '.4f'), format(accs, '.2f')


def test(test_loader, test_model, criterion, loss_func):
    test_model.eval()
    # tnet.eval()
    total = 0
    acc0, acc1, acc2, acc3, acc4 = 0, 0, 0, 0, 0
    test_loss = 0
    test_correct = 0
    acck3 = 0
    acck2 = 0

    # switch to train mode
    for i, data in enumerate(test_loader):

        test_model.eval()
        with torch.no_grad():
            # img, label = data[0], data[1]
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            # _, temp, _, _,= test_model(img, img,label,label)
            _, temp, _, _ = test_model(img, img)
            loss1 = loss_func(temp, label)
            test_loss += loss1.item() * batch_size

            _, pred = torch.max(temp, 1)
            print('pred', pred)
            print('label', label)

            total += batch_size
            for i in range(batch_size):
                if pred[i] == label[i] and label[i] == 0:
                    acc0 = acc0 + 1
                if pred[i] == label[i] and label[i] == 1:
                    acc1 = acc1 + 1
                if pred[i] == label[i] and label[i] == 2:
                    acc2 = acc2 + 1
                if pred[i] == label[i] and label[i] == 3:
                    acc3 = acc3 + 1
                if pred[i] == label[i] and label[i] == 4:
                    acc4 = acc4 + 1
            # test_correct += torch.sum(concat_predict == label).item()
            test_correct += torch.sum(pred == label).item()
            _, id = torch.topk(temp, 2)
            label = label.view(-1, 1)
            acck2 += (label == id).sum().item()
            _, id = torch.topk(temp, 3)
            # label = label.view(-1, 1)

            acck3 += (label == id).sum().item()

    # losses = loss_double/total
    # print("total", total)
    accs = test_correct / (total)
    acck2 = acck2 / total
    acck3 = acck3 / total
    losses = (test_loss) / total
    # accs =float(acc1)/total
    # losses = test_loss1/total

    logger.info('TestLoss: {:.4f} \t'
                'TestAcc: {:.2f}% \t' 'acc0:{:d} \t''acc1:{:d} \t''acc2:{:d} \t''acc3:{:d} \t''acc4:{:d}\t''acck2:{:.2f}%\t''acck3:{:.2f}%\t'
                .format(losses, 100. * accs, acc0, acc1, acc2, acc3, acc4, 100. * acck2, 100. * acck3))
    return format(losses, '.4f'), format(accs, '.4f')


def save_checkpoint(state, is_best, is_acc, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best_loss.pth')
    if is_acc:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best_acc.pth')


def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置输出格式 时间 logger名字(main) 日志级别名称(info)

    logfile = args.model + '.log' if not args.test else args.model + '_test.log'
    file_handler = logging.FileHandler(logfile, 'w')  # 用于写入文件的handler
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()  # 用于输出到控制台的handler
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main():
    global args
    args = parser.parse_args()
    print(args.cuda)

    global logger
    logger = set_logger()

    args.cuda = args.cuda and torch.cuda.is_available()

    print(args.cuda)

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.visdom:
        global plotter
        # plotter = VisdomLinePlotter(env_name=args.name)

    global meta
    meta = MetaLoader(args.data_path, args.dataset)

    global attributes
    attributes = [i for i in range(len(meta.data['ATTRIBUTES']))]



    # backbone = model_fpn.model_fpn()
    # net = model_fpn.DoubleNet(backbone)
    # net = model5.DoubleNet(backbone)
    pre_backbone = depthwise_model.Res()
    pre_net = depthwise_model.DoubleNet(pre_backbone)
    backbone = depthwise_model1.Res()
    net = depthwise_model1.DoubleNet(backbone)
    checkpoint = torch.load('./runs/depthwise/2to03/model_best_acc.pth', map_location = 'cuda:0' )
    pre_net.load_state_dict(checkpoint['state_dict'])


    pre_dict = {k: v for k, v in pre_net.named_parameters() if k in net.state_dict()}
    model_dict = net.state_dict()
    model_dict.update(pre_dict)
    net.load_state_dict(model_dict)
    # # print(net)
    #
    #     #print(param)
    #
    #
    # # print('b',net.load_state_dict)
    # for k, v in net.named_parameters():
    # #         print(k)
    #         if k!= 'embeddingnet.fc1.weight' and k!='embeddingnet.fc1.bias' and k!= 'embeddingnet.fc2.weight' and k!='embeddingnet.fc2.bias':
    #       #     and k!='embeddingnet.conv1.1.weight' and  k!='embeddingnet.conv1.1.bias' and  k!='embeddingnet.conv1.0.weight' and  k!='embeddingnet.conv1.0.bias':
    # # #        if  k!= 'fc3.weight' and k!='fc3.bias':
    #              v.requires_grad = False  # 固定参数
    # # # # # #         # print('k,v:',k,v)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    # enet = get_model(args.model)(backbone, n_attributes=len(attributes), embedding_size=args.dim_embed)
    # tnet = get_model('Tripletnet')(enet)
    if args.cuda:
        backbone.cuda()
        net.cuda()
        # net.cuda()
    # criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    criterion = ContrastiveLoss()
    loss_func = nn.CrossEntropyLoss()
    #loss_func1 = focal_loss()
    n_parameters = sum([p.data.nelement() for p in backbone.parameters()])
    logger.info('  + Number of params: {}'.format(n_parameters))

    # optionally resume from a checkpoint
    if args.resume:
        print('resume')
        print(args.resume)
        if os.path.isfile(args.resume):
            print('find')
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['prec']
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {} loss on validation set {})"
                        .format(args.resume, checkpoint['epoch'], loss))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    print(args.start_epoch)
    cudnn.benchmark = True

    kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            TripletImageLoader(args.data_path, args.dataset, args.test_num_triplets, 'filenames_test2.txt',

                               transform=transforms.Compose([
                                   transforms.Resize(224, interpolation=Image.BICUBIC),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize,
                               ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loss, test_acc = test(test_loader, net, criterion, loss_func)
        sys.exit()

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=0.01)
    # optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=0.01)
    # optimizer = optim.SGD(parameters, lr=args.lr,weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_rate)

    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader(args.data_path, args.dataset, args.num_triplets, 'filenames_train0.txt',
                           transform=transforms.Compose([
                               # transforms.Resize(224,interpolation=Image.BICUBIC),
                               transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                               # transforms.CenterCrop(224),
                               # transforms.RandomHorizontalFlip(),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5),
                               # transforms.ColorJitter(brightness=(1,1.5),contrast=(1,1.5)),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print('load traindata successful!')
    testset = Thyroid(root='./data_new', name1='filenames_test0.txt', is_train=False, data_len=None)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16,
                                              shuffle=False, num_workers=0, drop_last=False)
    # test_loader = torch.utils.data.DataLoader(
    #     TripletImageLoader(args.data_path, args.dataset, args.test_num_triplets, 'filenames_test0.txt',
    #                        transform=transforms.Compose([
    #                            transforms.Resize(224,interpolation=Image.BICUBIC),
    #                            transforms.CenterCrop(224),
    #                            transforms.ToTensor(),
    #                            normalize,
    #                        ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    print('load testdata successful!')
    logger.info("Begin training on {} dataset.".format(args.dataset))

    best_loss = float("inf")
    best_acc = 0
    start = time.time()
    list = []

    for epoch in range(args.start_epoch, args.epochs + 1):
        # print('epoch',epoch)
        # train for one epoch
        train_loss, train_acc = train(train_loader, net, loss_func, criterion, optimizer, epoch)
        train_loader.dataset.refresh()
        # evaluate on validation set
        test_loss, test_acc = test(test_loader, net, criterion, loss_func)
        # test_loader.dataset.refresh()
        #
        # # remember best meanAP and save checkpoint
        is_best = float(test_loss) < best_loss
        is_acc = float(test_acc) > best_acc
        best_loss = min(float(test_loss), best_loss)
        best_acc = max(float(test_acc), best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'prec': best_loss,
        }, is_best, is_acc, filename='checkpoint' + str(epoch) + '.pth')

        # update learning rate
        scheduler.step()
        for param in optimizer.param_groups:
            logger.info('lr:{}'.format(param['lr']))
            break
        list.append([epoch, train_loss, test_loss, train_acc, test_acc])

    end = time.time()
    duration = int(end - start)
    minutes = (duration // 60) % 60
    hours = duration // 3600
    logger.info('training time {}h {}min'.format(hours, minutes))
    name = ['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc']

    temp = pd.DataFrame(columns=name, data=list)
    temp.to_csv("./runs/unet/" + "loss.csv")


if __name__ == '__main__':
    main()
