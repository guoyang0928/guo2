import torch
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
# from utils.transform import get_transform_for_test
# from senet.se_resnet import FineTuneSEResnet50
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from dataset_thyroid import Thyroid
from model import model_new,model3
from image_loader import TripletImageLoader, MetaLoader
from torchvision import transforms

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

testset = Thyroid(root='/home/stellla/PycharmProjects/guo/data_new', name1='filenames_train0.txt', is_train=False, data_len=None)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0, drop_last=False)
train_loader = torch.utils.data.DataLoader(
    TripletImageLoader("/home/stellla/PycharmProjects/guo/data_new", "thyroid", 1, 'draw.txt',
                       transform=transforms.Compose([
                           # transforms.Resize(224,interpolation=Image.BICUBIC),
                           transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                           # transforms.CenterCrop(224),
                           # transforms.RandomHorizontalFlip(),
                           # transforms.RandomHorizontalFlip(p=0.5),
                           # transforms.RandomVerticalFlip(p=0.5),
                           # transforms.ColorJitter(brightness=(1,1.5),contrast=(1,1.5)),
                           transforms.ToTensor(),
                           normalize,
                       ])),
    batch_size=1, shuffle=True)

num_class = 5  # 类别数量

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 12}
#plt.rc('font', **font)


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test(model):
    # 加载测试集和预训练模型参数

    model.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签

    for i, data in enumerate(test_loader):
        inputs, labels = data[0].cuda(), data[1].cuda()

        _,outputs,_ = model(inputs,inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    #
    # # macro
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_class):
    #     mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # # Finally average it and compute AUC
    # mean_tpr /= num_class
    # fpr_dict["macro"] = all_fpr
    # tpr_dict["macro"] = mean_tpr
    # roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线

    plt.figure(figsize=(5, 4), dpi=100)
    #plt.style.use('seaborn-darkgrid')

    palette = plt.get_cmap('Set1')

    #plt.figure()
    lw = 1.8
    # plt.plot(fpr_dict["micro"], tpr_dict["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc_dict["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr_dict["macro"], tpr_dict["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc_dict["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['black', 'darkorange', 'cornflowerblue','forestgreen','deeppink'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=palette(i), lw=lw,linestyle='-',
                 label='TR{0} (AUC = {1:0.2f})' ''.format(i, roc_auc_dict[i]))
    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font)
    plt.ylabel('True Positive Rate', fontdict=font)
    plt.title('ROC Curves of GLF-Net', fontdict=font)
    #plt.rcParams.update(font)
    plt.legend(prop=font,loc="lower right")

    plt.savefig('set113_roc.jpg',dpi =300)
    plt.show()
def test1(model):
    # 加载测试集和预训练模型参数

    model.eval()

    for i, data in enumerate(test_loader):
        inputs, labels = data[0].cuda(), data[1].cuda()

        _,outputs,_ = model(inputs,inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
def test2(model):
    model.eval()
    for batch_idx, (data1, label1, data2, label2) in enumerate(train_loader):

        data1, data2 = data1.cuda(), data2.cuda()
        label1, label2 = label1.cuda(), label2.cuda()
        model(data1, data2)


if __name__ == '__main__':
    # 加载模型
    backbone = model3.Res()
    net = model3.DoubleNet(backbone)
    checkpoint = torch.load('/home/stellla/PycharmProjects/guo/runs/new1/model_best_acc.pth')
    backbone.cuda()
    net.cuda()
    net.load_state_dict(checkpoint['state_dict'])

    test2(net)