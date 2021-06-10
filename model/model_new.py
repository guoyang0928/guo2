import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from draw_features import draw_features,apply_heatmap
from torch.nn import Parameter
from model import res


def norm(X):
    """L2-normalize columns of X
    """
    # norm = X.sum(dim=1, keepdim=True)
    min,_=torch.min(X,dim=2, keepdim=True)
    max,_=torch.max(X,dim=2, keepdim=True)
    # print(min)
    # print(max)
    X = (X - min)/(max-min)

    #X = torch.div(X, norm)
    return X
def process(fea,v):
    fea = fea.unsqueeze(dim=2)
    # print(fea1)
    v1= v[:, 5:, :]

    fea1 = fea[:, 5:, :]

    t1 = torch.mul(fea1,v1)
    #print(t1.shape)
    t1 = torch.reshape(t1,(t1.size(0), -1))

    #print(t1)


    # fea = fea.unsqueeze(dim=2)
    #
    # v = v[:, 5:, :]  # 16*5*10
    #
    # fea = fea[:, 5:, :]  # 16*5*1
    # # print(fea1.shape)
    # t2 = torch.cat((fea, v), dim=2)
    # t2 = t2.view(t2.size(0), 1, -1)
    # t2 = t2.squeeze(dim=1)
    return t1

def norm1(X):
    """L2-normalize columns of X
    """
    # norm = X.sum(dim=1, keepdim=True)
    min,_=torch.min(X,dim=1, keepdim=True)
    max,_=torch.max(X,dim=1, keepdim=True)
    # print(min)
    # print(max)
    X = (X - min)/(max-min)

    #X = torch.div(X, norm)
    return X
class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        # kernel_up = [[[-1],[1]] for _ in range(10)]
        # #print(kernel_up.shape)
        # kernel_bottom = [[[1],[-1]] for _ in range(10)]
        # kernel_left = [[[1,-1]] for _ in range(10)]
        #
        #
        # kernel_right = [[[-1, 1]] for _ in range(10)]
        kernel_up = [[-1],[1]]
        #print(kernel_up.shape)
        kernel_bottom = [[1],[-1]]
        kernel_left = [[1,-1]]


        kernel_right = [[-1, 1]]


        kernel_up = torch.FloatTensor(kernel_up).unsqueeze(0).unsqueeze(0)
        #print(kernel_up.shape)
        #print(kernel_up)
        kernel_bottom = torch.FloatTensor(kernel_bottom).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor(kernel_left).unsqueeze(0).unsqueeze(0)
        #print(kernel_left.shape)
        kernel_right = torch.FloatTensor(kernel_right).unsqueeze(0).unsqueeze(0)

        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_bottom = nn.Parameter(data=kernel_bottom, requires_grad=False)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)


    def forward(self, x):
        x = x[:, 0]  #只取了第一个通道做梯度 写的有问题  但先融合各通道效果一般
      


        x=x.unsqueeze(1)
        #print("x", x.shape)

        x1 = F.conv2d(x, self.weight_up, padding=0)
        pad = nn.ZeroPad2d(padding=(0,0, 0, 1))
        x1 = pad(x1)
        x1 = x1.squeeze(dim=1)


        x2 = F.conv2d(x, self.weight_bottom, padding=0)
        pad = nn.ZeroPad2d(padding=(0, 0, 1, 0))
        x2 = pad(x2)
        x2 = x2.squeeze(dim=1)


        x3 = F.conv2d(x, self.weight_left, padding=0)
        pad = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        x3 = pad(x3)
        x3 = x3.squeeze(dim=1)


        x4 = F.conv2d(x, self.weight_right, padding=0)
        pad = nn.ZeroPad2d(padding=(0, 1, 0, 0))
        x4 = pad(x4)
        x4 = x4.squeeze(dim=1)
        res = torch.sqrt(x1**2+x2**2+x3**2+x4**2)
        #print(res.shape)



        return res


class DoubleNet(nn.Module):
    def __init__(self, embeddingnet):
        super(DoubleNet, self).__init__()
        self.embeddingnet = embeddingnet

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(10, 10, kernel_size=2, stride=2),
        #     # nn.Conv2d(32, 10, kernel_size=4, stride=2,padding= 1),
        #     nn.BatchNorm2d(10),
        #     # nn.ReLU(inplace=True)
        # )
        # self.conv2 = nn.Sequential(
        #     #nn.Conv2d(512, 10, kernel_size=3, stride=1),
        #     nn.Conv2d(10, 10, kernel_size=2,stride = 2),
        #     nn.BatchNorm2d(10),
        #     #nn.ReLU(inplace=True)
        # )
        #self.pool = torch.nn.AvgPool2d(2)
        #self.out1 = nn.Conv2d(64, 1, kernel_size=1)  #1*1卷积
        # output = (1,1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x, y):
        """ x: Anchor image,
            y: Distant (negative) image,
            fea t
        """

        x1, fea1, t1 = self.embeddingnet(x)
        y1, fea2, t2 = self.embeddingnet(y)
        # #z1, fea3, t3 = self.embeddingnet(z, lb2)
        # print(t1.shape)
        # print("x1",x1.shape)
        # print("y1",x1.shape)

        x2 = x1.view(x1.size(0), x1.size(1), -1)
        #print(x2.shape)
        y2 = y1.view(y1.size(0), y1.size(1), -1)
        wxy = torch.abs(t1 - t2)
        #print(wxy)


        # z2 = z1.view(z1.size(0), z1.size(1), -1)
        # wyx = torch.abs(t1 - t3)

        wxy = wxy.view(wxy.size(0),wxy.size(1),-1)


        #print(wxy.shape)
        res1 = x2.mul(wxy)
        res2 = y2.mul(wxy)
        # res3 = z2.mul(wyx)
        # res4 = x2.mul(wyx)
        #print(res1.shape)
        res1 = res1.view(res1.size(0), res1.size(1), 7, 7)
        res2 = res2.view(res2.size(0), res2.size(1), 7, 7)
        # res3 = res3.view(res1.size(0), res1.size(1), 7, 7)
        # res4 = res4.view(res2.size(0), res2.size(1), 7, 7)

        #print(res1.shape)
        res1 = self.pool(res1)
        res1 = res1.view(res1.size(0), res1.size(1))
        #print(res1.shape)
        res2 = self.pool(res2)
        res2 = res2.view(res2.size(0), res2.size(1))

        # res3 = self.pool(res3)
        # res3 = res3.view(res2.size(0), res2.size(1))
        # res4 = self.pool(res4)
        # res4 = res4.view(res2.size(0), res2.size(1))







        # Z1 = torch.nn.functional.adaptive_avg_pool2d(Z1, (1, 1))
        # Z1 = Z1.view(Z1.size(0), Z1.size(1))
        # # print("Z1:",Z1.shape)
        # Z2 = torch.nn.functional.adaptive_avg_pool2d(Z2, (1, 1))
        # #print('Z2:', Z2.shape)
        # Z2 = Z2.view(Z2.size(0), Z2.size(1))
        #
        # Z1 = l2norm(Z1)
        # Z2 = l2norm(Z2)
        # Z1 = Z1.view(Z1.size(0),-1)
        # Z1 = Z1.view(Z1.size(0),-1)
        #print(Z1.shape)
        # Z2 = Z2.view(Z1.size(0), -1)
        # Z2 = Z2.view(Z2.size(0),-1)

        return fea1,fea2,res1, res2

def l2norm(X):
    """L2-normalize columns of X
    """
    # norm = X.sum(dim=1, keepdim=True)
    norm = torch.pow(X, 2).sum(dim=2, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
def l2norm1(X):
    """L2-normalize columns of X
    """
    mean = torch.mean(X, dim=1)
    std = torch.std(X, dim=1)
    mean = mean.unsqueeze(dim=1)
    std = std.unsqueeze(dim=1)
    # print(std)
    # print(mean)
    X = (X - mean) / std
    return X

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=5,
                 m=0.1,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        # self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        # print(x.size()[0])
        # print(x.size()[1])
        # print(self.in_feats)
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12) #按行求1范数
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12) #按列求2范数
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        #loss = self.ce(costh_m_s, lb)
        return costh_m_s
def Conv1(in_planes, places, size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=size,stride=stride,padding = padding,bias=False),
        nn.BatchNorm2d(places),
        #nn.ReLU(places)
    )
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.features0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)
        self.features1 = model.layer1
        self.features2 = model.layer2
        self.features3 = model.layer3
        self.features4 = model.layer4
        # self.features = nn.Sequential(
        #     model.conv1,
        #     model.bn1,
        #     model.relu,
        #     model.maxpool,
        #
        #     # model.pre,
        #
        #     model.layer1,
        #
        #     model.layer2,
        #
        #     model.layer3,
        #     model.layer4
        #
        #     # model.layer3#new
        # )
        self.conv = Conv1(in_planes=512, places=10, size=3,stride=1,padding=1)
        self.conv_s = Conv1(in_planes=10, places=10, size=3, stride=1, padding=1)
        self.conv_1 = Conv1(in_planes=10, places=10, size=2, stride=2, padding=0)
        self.conv_2 = Conv1(in_planes=10, places=10, size=2, stride=2, padding=0)
        self.conv_s1 = Conv1(in_planes=64, places=10, size=3, stride=1, padding=1)
        self.conv_s2 = Conv1(in_planes=128, places=10, size=3, stride=1, padding=1)
        self.conv_s3 = Conv1(in_planes=256, places=10, size=3, stride=1, padding=1)

        #self.conv_2_1 = Conv1(in_planes=10, places=1, size=1)


        '''
        self.conv = nn.Sequential(
            # nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            # nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            #nn.LeakyReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=2, stride=2),
            # nn.Conv2d(32, 10, kernel_size=4, stride=2,padding= 1),
            nn.BatchNorm2d(10),
            # nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            #nn.Conv2d(512, 10, kernel_size=3, stride=1),
            nn.Conv2d(10, 10, kernel_size=2,stride = 2),
            nn.BatchNorm2d(10),
            #nn.ReLU(inplace=True)
        )
        '''
        #
        # self.conv3 = nn.Sequential(
        #     #nn.Conv2d(512, 10, kernel_size=3, stride=1),
        #     nn.Conv2d(10, 10, kernel_size=3,stride = 1),
        #     nn.BatchNorm2d(10),
        #     #nn.ReLU(inplace=True)
        # )
        # new
        # self.add_conv = nn.Sequential(
        #     # nn.Conv2d(512, 10, kernel_size=3, stride=1),
        #     nn.Conv2d(10, 10, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.addconv1 = nn.Sequential(
        #     # nn.Conv2d(512, 10, kernel_size=3, stride=1),
        #     nn.Conv2d(10, 3, kernel_size=1),
        #     nn.BatchNorm2d(3),
        #     # nn.ReLU(inplace=True)
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(10,10, kernel_size=1),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(inplace=True)
        # )

        # self.fc1 = nn.Linear(4*4*10,128)
        # self.fc2 = nn.Linear(128,64)
        # self.fc3 = nn.Linear(64,5)



        self.softmax = nn.Softmax(dim=2)
        self.gas = GaussianBlur()
        # self.w1 = Parameter(torch.tensor(1).float())
        # self.w2 = Parameter(torch.tensor(1).float())
        # self.w3 = Parameter(torch.tensor(1).float())
        #self.s=nn.Sigmoid()

        self.fc1 = nn.Linear(70,128)
        self.fc2 = nn.Linear(128, 5)
        # self.fc_l2 = nn.Linear(28*28, 10)
        # self.fc_l3 = nn.Linear(14*14, 10)
        self.fc_l = nn.Linear(7 * 7, 10)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        #self.soft = AMSoftmax(64,5)
        self.soft = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def CIM(self, x):


        # print("x_global",x_global.cpu().data.numpy())
        t = x.view(x.size(0), x.size(1), -1)  # 16个通道
        t =l2norm(t)
        x1=t


        #print("x1",x1.cpu().data.numpy())
        x1_T = torch.transpose(x1, 2, 1)
        x_global = torch.bmm(x1, x1_T)

        return x_global

    def forward(self, x):
        x = self.features0(x)   # 16 64 56 56

        x = self.features1(x)   # 16 64 56 56
        #x_s1 = self.conv_s1(x)

        x = self.features2(x)   # 16 128 28 28
        #print(x.shape)
        x_s2 = self.conv_s2(x)
        #x_s2_1 = self.conv_2_1(x_s2)
        #print("w",x_s2_1.shape)

        x_s2_1 = self.relu(x_s2)
        # for i in range(10):
        #     a = x_s2_1[0, i, :, :]
        #     a = a.squeeze()
        #     draw_features(a.cpu().data.numpy(), origi_image)
        x_l2 = self.gas(x_s2_1)
        #x_l2 =x_l2.view(x_l2.size(0),x_l2.size(2)*x_l2.size(3))
        x_l2 = self.pool1(x_l2)
        # for i in range(1):
        #     a = x_l2[0, i, :, :]
        #     a = a.squeeze()
        #
        #     #apply_heatmap(a.cpu().data.numpy(), origi_image)
        #     draw_features(a.cpu().data.numpy(), origi_image)

        #x_l2 = self.pool(x_l2)
        #print(x_l2.shape)

        #x_l2 = self.fc_l2(x_l2)



        x = self.features3(x)   # 16 256 14 14
        #print(x.shape)
        x_s3 = self.conv_s3(x)

        #x_s3_1 = self.conv_2_1(x_s3)

        x_s3_1 = self.relu(x_s3)
        # for i in range(10):
        #     a = x_s3_1[0, i, :, :]
        #     a = a.squeeze()
        #     draw_features(a.cpu().data.numpy(), origi_image)
        x_l3 = self.gas(x_s3_1)
        x_l3 = self.pool2(x_l3)
        # for i in range(1):
        #     a = x_l3[0, i, :, :]
        #     a = a.squeeze()
        #     draw_features(a.cpu().data.numpy(), origi_image)
            #apply_heatmap(a.cpu().data.numpy(), origi_image)


        # x_l3 =x_l3.view(x_l3.size(0),x_l3.size(2)*x_l3.size(3))
        # x_l3 = self.fc_l3(x_l3)

        x = self.features4(x)   # 16 512 7 7



        x = self.conv(x)
        x_s4 = self.conv_s(x)

        #x_s4_1 = self.conv_2_1(x_s4)
        x_s4_1 = self.relu(x_s4)

        # for i in range(10):
        #     a = x_s4_1[0, i, :, :]
        #     a = a.squeeze()
        #     draw_features(a.cpu().data.numpy(), origi_image)
        x_l4 = self.gas(x_s4_1)

        # for i in range(1):
        #     a = x_l4[0, i, :, :]
        #     a = a.squeeze()
        #     draw_features(a.cpu().data.numpy(), origi_image)
            #apply_heatmap(a.cpu().data.numpy(), origi_image)



        # x_l4 =x_l4.view(x_l4.size(0),x_l4.size(2)*x_l4.size(3))
        # x_l4 = self.fc_l4(x_l4)

        x_l = x_l2+x_l3+x_l4
        #x_l = x_l2+x_l3
        print("e",x_l.shape)
        x_l = x_l.view(x_l.size(0), x_l.size(1) * x_l.size(2))
        x_l = self.fc_l(x_l)
        #print(x_l.shape)
        x_l = l2norm1(x_l)



        copy_x = x
        copy_x = self.conv_1(copy_x)
        # print("w",copy_x.shape)
        # print(copy_x)
        #
        copy_x = self.conv_2(copy_x)
        # print("w", copy_x.shape)
        # #print(copy_x)
        copy_x=copy_x.view(copy_x.size(0),copy_x.size(1))
        #print(copy_x.shape)
        # #print(copy_x)
        copy_x = l2norm1(copy_x)


        x_global2 = self.CIM(x_s2)
        x_global3 = self.CIM(x_s3)
        x_global4 = self.CIM(x_s4)
        #
        #
        (fea2, v2) = x_global2.symeig(eigenvectors=True)
        (fea3, v3) = x_global3.symeig(eigenvectors=True)
        (fea4, v4) = x_global4.symeig(eigenvectors=True)
        #print(fea2.shape)
        t2 = process(fea2,v2)
        t3 = process(fea3, v3)
        t4 = process(fea4, v4)
        #
        fea = t2+t3+t4


        fea = l2norm1(fea)

        fea = torch.cat((copy_x,fea,x_l),dim=1)
        #fea = x_l
        #fea = l2norm1(fea)
        #print(fea.shape)


        # new
        # x = x.view(x.size(0), -1)
        # fea = self.fc1(x)

        res = self.fc1(fea)

        res = self.fc2(res)

        #res = self.soft(res)

        #res = self.soft(res,lb)
        #print(res)
        # print(fea.shape)

        return x,res,copy_x


def Res(pretrained=True):
    model = torchvision.models.resnet34(pretrained=True)
    #model = torchvision.models.resnet18(pretrained=True)
    #model = res.ResNet50()
    print(model)
    return Net(model)


if __name__ == "__main__":
    model = Res()
    # model1 = torchvision.models.resnet50(pretrained=True)
    # print(model1)
    # print(model)
    # unet = Net(in_channels=3)
    x = torch.zeros(16, 3, 224, 224)
    #model(x)
    # unet(x)
    lb = torch.randint(0, 5, (16,), dtype=torch.long)
    DoubleNet = DoubleNet(embeddingnet=model)
    y= torch.zeros(16, 3, 224, 224)
    model(x)