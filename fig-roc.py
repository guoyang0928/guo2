import torch
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder

from scipy import interpolate as interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from dataset_thyroid import Thyroid
from model import model_new,model3,model5,depthwise_model,depthwise_model2
from image_loader import TripletImageLoader, MetaLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
# import seaborn as sns

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

testset = Thyroid(root='./data_new', name1='filenames_test0.txt', is_train=False, data_len=None)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0, drop_last=False)
# train_loader = torch.utils.data.DataLoader(
#     TripletImageLoader("/home/stellla/PycharmProjects/guo/data_new", "thyroid", 1, 'draw.txt',
#                        transform=transforms.Compose([
#                            # transforms.Resize(224,interpolation=Image.BICUBIC),
#                            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
#                            # transforms.CenterCrop(224),
#                            # transforms.RandomHorizontalFlip(),
#                            # transforms.RandomHorizontalFlip(p=0.5),
#                            # transforms.RandomVerticalFlip(p=0.5),
#                            # transforms.ColorJitter(brightness=(1,1.5),contrast=(1,1.5)),
#                            transforms.ToTensor(),
#                            normalize,
#                        ])),
#     batch_size=1, shuffle=True)

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

        _,outputs,_ = model(inputs)
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
    d = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}
    for i, color in zip(range(num_class-1,-1,-1), colors):
        print(i)
        j = d[i]

        plt.plot(fpr_dict[i], tpr_dict[i], color=palette(i), lw=lw,linestyle='-',
                 label='TR{0} (AUC = {1:0.2f})' ''.format(j, roc_auc_dict[i]))
    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font)
    plt.ylabel('True Positive Rate', fontdict=font)
    # plt.title('ROC Curves of GLF-Net', fontdict=font)
    #plt.rcParams.update(font)
    plt.legend(prop=font,loc="lower right")

    # plt.savefig('set113_roc.jpg',dpi =300)
    plt.show()
def test1(model): # 画 混淆矩阵
    # 加载测试集和预训练模型参数

    model.eval()
    predict_list = []  # 存储预测得分
    label_list = []  # 存储真实标签

    for i, data in enumerate(test_loader):
        inputs, labels = data[0].cuda(), data[1].cuda()

        _,outputs,_= model(inputs)
        _, pred =  torch.max(outputs, 1)
        predict_list.extend(pred.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
    # labels = ['TR5','TR4','TR3','TR2','TR1']
    labels = ['TR1', 'TR2', 'TR3', 'TR4', 'TR5']
    cm = confusion_matrix(label_list, predict_list)
    new_cm = np.zeros((5,5))
    # print("mmm", cm)

    for j in range(5):
        new_cm[j] = cm[4-j][::-1]
    print("mmm", new_cm)



    cm_normalized = new_cm.astype('float') / new_cm.sum(axis=1)[:, np.newaxis]
    return cm_normalized
    #print(cm_normalized)
    # plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Oranges)
    # xlocations = np.array(range(len(labels)))
    # # plt.title("Confusion Matrix")
    # # plt.colorbar()
    # plt.xticks(xlocations, labels, rotation=90)
    # plt.yticks(xlocations, labels)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    #
    # iters = np.reshape([[[i, j] for j in range(5)] for i in range(5)], (cm_normalized.size, 2))
    # for i, j in iters:
    #     plt.text(j, i, "%0.2f" %((cm_normalized[i, j])),va = 'center', ha = 'center')  # 显示对应的数字
    # return
    # plt.show()

def test2(model):
    model.eval()
    d ={0:5,1:4,2:3,3:2,4:1}
    f = open("./2class_result.txt","a+")
    for i, data in enumerate(test_loader):
        inputs= data[0].cuda()
        label = data[1]
        name = data[2]
        print(name[0])

        _,outputs,_= model(inputs)
        _, pred =  torch.max(outputs, 1)
        label = label.numpy()
        pred = pred.cpu().data.numpy()
        f.write(name[0]+" " +str(d[label[0]])+" "+str(d[int(pred[0])])+"\n")




if __name__ == '__main__':
    # 加载模型
    backbone = depthwise_model2.Res()
    net = depthwise_model2.DoubleNet(backbone)
    checkpoint = torch.load('./runs/depthwise/model3_2/model_best_acc.pth')
    backbone.cuda()
    net.cuda()
    net.load_state_dict(checkpoint['state_dict'])
    test(backbone)
    '''
    labels = ['TR1', 'TR2', 'TR3', 'TR4', 'TR5']

    checkpoint = torch.load('./runs/depthwise/2to03/model_best_acc.pth')
    backbone.cuda()
    net.cuda()
    net.load_state_dict(checkpoint['state_dict'])
    cm2 = test1(backbone)




    fig = plt.figure(figsize=(16, 9))
    plt.title("Confusion Matrix")


    plt.subplot(121)
    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.Oranges)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('(a)', y=-0.2)

    iters = np.reshape([[[i, j] for j in range(5)] for i in range(5)], (cm1.size, 2))
    for i, j in iters:
        plt.text(j, i, "%0.2f" %((cm1[i, j])),va = 'center', ha = 'center')  # 显示对应的数字


    plt.subplot(122)
    h3 = plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Oranges)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('(b)', y=-0.2)

    iters = np.reshape([[[i, j] for j in range(5)] for i in range(5)], (cm2.size, 2))
    for i, j in iters:
        plt.text(j, i, "%0.2f" % ((cm2[i, j])), va='center', ha='center')  # 显示对应的数字

    l = 0.92
    b = 0.12
    w = 0.015
    h = 1 - 2 * b

    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h3, cax=cbar_ax)

    # 设置colorbar标签字体等
    cb.ax.tick_params(labelsize=12)  # 设置色标刻度字体大小。

    plt.show()
    '''

