import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
#from draw_features import draw_features
from torch.nn import Parameter
from model import res,FPN




# class Tripletnet(nn.Module):
#     def __init__(self, embeddingnet):
#         super(Tripletnet, self).__init__()
#         self.embeddingnet = embeddingnet
#
#     def forward(self, x, y, z):
#         """ x: Anchor image,
#             y: Distant (negative) image,
#             z: Close (positive) image,
#             c: Integer indicating according to which attribute images are compared"""
#         embedded_x,t1 = self.embeddingnet(x)
#         embedded_y,t2 = self.embeddingnet(y)
#         embedded_z,t3 = self.embeddingnet(z)
#         sim_a = torch.sum(embedded_x * embedded_y, dim=1)
#         #print(sim_a)
#         sim_b = torch.sum(embedded_x * embedded_z, dim=1)
#
#         return embedded_x, embedded_y, embedded_z

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
        #z1, fea3, t3 = self.embeddingnet(z, lb2)

        # print(t1.shape)
        # print("x1",x1.shape)
        # print("y1",x1.shape)

        x2 = x1.view(x1.size(0), x1.size(1), -1)
        #print(x2.shape)
        y2 = y1.view(y1.size(0), y1.size(1), -1)
        wxy = torch.abs(t1 - t2)
        # z2 = z1.view(z1.size(0), z1.size(1), -1)
        # wyx = torch.abs(t1 - t3)

        wxy = wxy.view(wxy.size(0),wxy.size(1),-1)
        #wyx = wyx.view(wxy.size(0), wxy.size(1), -1)
        #print(wxy)

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
    # norm = X.sum(dim=1, keepdim=True)
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.features = model
        self.conv = nn.Sequential(
            # nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.Conv2d(64, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            #nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=2, stride=2),
            # nn.Conv2d(32, 10, kernel_size=4, stride=2,padding= 1),
            nn.BatchNorm2d(10),
            #nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            #nn.Conv2d(512, 10, kernel_size=3, stride=1),
            nn.Conv2d(10, 10, kernel_size=2,stride = 2),
            nn.BatchNorm2d(10),
            #nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            # nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.Conv2d(64, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            # nn.ReLU(inplace=True)
        )
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
        self.w1 = Parameter(torch.tensor(1).float())
        self.w2 = Parameter(torch.tensor(1).float())

        #self.s=nn.Sigmoid()

        self.fc1 = nn.Linear(110,128)
        self.fc2 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(p=0.2)
        #self.soft = AMSoftmax(5,5)
        self.soft = nn.Softmax(dim=1)

    def CIM(self, x):
        # print("x_global",x_global.cpu().data.numpy())
        t = x.view(x.size(0), x.size(1), -1)  # 16个通道
        print(t.shape)
        t = l2norm(t)
        x1 = t


        x1_T = torch.transpose(x1, 2, 1)
        x_global = torch.bmm(x1, x1_T)
        # x_global = torch.matmul(x1, x1_T)  # b*16*16
        # print("x_global", x_global)
        # x_global = norm(x_global)

        #print("x_global", x_global.cpu().data.numpy())

        # new
        # w_global = self.softmax(x_global)
        # x_new = torch.matmul(w_global,x1)  #10,28*28
        #
        # x_new = x_new.view(x_new.size(0),x_new.size(1),x_new.size(2),-1)
        #
        # x_new = self.pool(x_new)
        #
        # fea1 = x_new.view(x_new.size(0),x_new.size(1))
        # new

        # w_local = self.softmax(-x_global)
        w_local = 1 - x_global
        # print("w_local",w_local.cpu().data.numpy())
        # print(w_local.shape)
        # print(w_local.cpu().data.numpy())
        x_local_temp = torch.matmul(w_local, x1)  # 64 64 * 64 224*224 = 64 224*224
        # x_local_temp_T = torch.transpose(x_local_temp, 2, 1)
        # x_local =  torch.matmul(x_local_temp, x_local_temp_T)
        # new
        # x_local_temp = x_local_temp.view(x_local_temp.size(0),x_local_temp.size(1),x_local_temp.size(2),-1)
        #
        #
        # x_local_temp =self.pool(x_local_temp)
        #
        # fea2 =x_local_temp.view(x_local_temp.size(0),x_local_temp.size(1))
        # new

        x_local_temp = l2norm(x_local_temp)
        x_local_T = torch.transpose(x_local_temp, 2, 1)
        x_local = torch.bmm(x_local_temp, x_local_T)
        #x_local = torch.matmul(x_local_temp, x_local_T)
        #x_local = norm(x_local)
        # for i in range(x_local.shape[0]):
        #     for j in range(x_local.shape[1]):
        #         a = x_local[i, j, :]
        #         a = a.cpu().data.numpy()
        #         x_local[i, j, :] = norm(a)*10
        # print("x_local", x_local.cpu().data.numpy())
        # x_global = x_global*10
        # print(x_global)
        #print("x_local", x_local.cpu().data.numpy())
        # x_local = x_local*10
        return x_global, x_local


    def forward(self, x):
        x1,x2 = self.features(x)

        # print("x1",x1.shape)
        # print("x2",x2.shape)


        x1 = self.conv(x1)
        copy_x = x1
        #print(copy_x.shape)
        copy_x = self.conv_1(copy_x)
        #print("w",copy_x.shape)

        copy_x = self.conv_2(copy_x)
        #print("w", copy_x.shape)
        # #print(copy_x)
        copy_x=copy_x.view(copy_x.size(0),copy_x.size(1))

        x2 = self.conv2(x2)
        x_global, x_local = self.CIM(x2)

        (fea1, v1) = x_global.symeig(eigenvectors=True)
        (fea2, v2) = x_local.symeig(eigenvectors=True)
        # # print("fea", fea1.shape)
        #print("fea1", fea1)
        # # print("fea2", fea2)
        #
        # fea1 = l2norm(fea1)
        # # print("fea1",fea1)
        # fea2 = l2norm(fea2)
        #fea = self.w1 * fea2 + self.w2 * copy_x

        #fea = torch.cat((fea1, fea2, copy_x), dim=1)


        #v1 = v1[:, 5:, :]
        # #v2 = v2[:, 5:, :]
        #fea1 = fea1[:, 5:]
        # #fea2 = fea2[:, 5:]
        #fea1 = fea1.unsqueeze(dim=1)
        # #fea2 = fea2.unsqueeze(dim=1)
        #t1 = torch.matmul(fea1, v1)
        # #t2 = torch.matmul(fea2, v2)
        #t1 = t1.squeeze()
        #t2 = t2.squeeze()

        fea1 = fea1.unsqueeze(dim=1)
        t1 = torch.cat((fea1, v1), dim=1)
        t1 = t1.view(t1.size(0), 1, -1)
        t1 = t1.squeeze(dim=1)
        #print(t1.shape)
        #
        # fea2 = fea2.unsqueeze(dim=1)
        # t2 = torch.cat((fea2, v2), dim=1)
        # t2 = t2.view(t2.size(0), 1, -1)
        # t2 = t2.squeeze(dim=1)
        # print(t2.shape)

        # new
        # x = x.view(x.size(0), -1)
        #fea = torch.cat((copy_x,t1),dim=1)
        #fea = self.w1*t1+self.w2*t2
        fea = t1
        fea = l2norm1(fea)
        #print(fea)

        res = self.fc1(fea)
        #res = self.dropout(res)
        res = self.fc2(res)
        #print(res.shape)
        res = self.soft(res)
        #res = self.soft(res,lb)
        #print(res)
        # print(fea.shape)

        return x1,res,copy_x


def model_fpn( ):
    #model = torchvision.models.resnet34(pretrained=True)
    model = FPN.FPN101()
    print(model)
    return Net(model)


if __name__ == "__main__":
    model =model_fpn()
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
    DoubleNet(x, y)