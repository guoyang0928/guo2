import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
#from draw_features import draw_features
from torch.nn import Parameter
from model import res
from draw_features import draw_features
import cv2
import csv
import codecs

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


        x1, fea1, t1,origi_x = self.embeddingnet(x)
        # for i in range(10):
        #     a = x1[0, i, :, :]
        #     a = a.squeeze()
        #     ori,_,_ = draw_features(a.cpu().data.numpy(), origi_x,i)

        y1, fea2, t2,origi_y = self.embeddingnet(y)

        # #z1, fea3, t3 = self.embeddingnet(z, lb2)
        # print(t1.shape)
        # print("x1",x1.shape)
        # print("y1",x1.shape)

        x2 = x1.view(x1.size(0), x1.size(1), -1)

        y2 = y1.view(y1.size(0), y1.size(1), -1)
        wxy = torch.abs(t1 - t2)
        # z2 = z1.view(z1.size(0), z1.size(1), -1)
        # wyx = torch.abs(t1 - t3)

        wxy = wxy.view(wxy.size(0),wxy.size(1),-1)



        av = (torch.sum(wxy,dim=1))/10
        av = torch.repeat_interleave(av, 10, dim=1)

        av = av.unsqueeze(dim=2)
        #print(av.shape)
        ones = torch.ones_like(wxy)
        wxy = torch.where(wxy < av, ones, wxy*10)
        #wyx = wyx.view(wxy.size(0), wxy.size(1), -1)

        # print(wxy.shape)
        # print(x2.shape)
        #print(x2[0,1,:])
        res1 = x2.mul(wxy)
        #print(res1[0,1,:])

        res2 = y2.mul(wxy)
        # res3 = z2.mul(wyx)
        # res4 = x2.mul(wyx)
        #print(res1.shape)
        res1 = res1.view(res1.size(0), res1.size(1), 7, 7)
        # for i in range(10):
        #     a = res1[0, i, :, :]
        #     a = a.squeeze()
        #     ori, _, _ = draw_features(a.cpu().data.numpy(), origi_x, i)

        res2 = res2.view(res2.size(0), res2.size(1), 7, 7)
        '''
        画图   
        conv = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1, stride=1, padding=0)
        conv = conv.cuda()
        res1 = conv(res1)
        res2 = conv(res2)
        print(res1.shape)

        for i in range(1):
            a = res1[0, i, :, :]
            a = a.squeeze()

            #apply_heatmap(a.cpu().data.numpy(), origi_image)
            t,d,f = draw_features(a.cpu().data.numpy(), x)
            cv2.imwrite('/home/stellla/Desktop/test/' + 'a' + "4_1"+ ".png", t, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite('/home/stellla/Desktop/test/' + 'b' + "4_1" + ".png", d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite('/home/stellla/Desktop/test/' + 'c' +"4_1" + ".png", f, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        for i in range(1):
            a = res2[0, i, :, :]
            a = a.squeeze()

            #apply_heatmap(a.cpu().data.numpy(), origi_image)
            t,d,f = draw_features(a.cpu().data.numpy(), y)
            cv2.imwrite('/home/stellla/Desktop/test/' + 'a'+"1" + ".png", t, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite('/home/stellla/Desktop/test/' + 'b' + "1" + ".png", d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite('/home/stellla/Desktop/test/' + 'c' + "1" + ".png", f, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # res3 = res3.view(res1.size(0), res1.size(1), 7, 7)
        # res4 = res4.view(res2.size(0), res2.size(1), 7, 7)
        '''

        #print(res1.shape)
        res1 = self.pool(res1)
        res1 = res1.view(res1.size(0), res1.size(1))
        print("res1",res1)
        res2 = self.pool(res2)

        res2 = res2.view(res2.size(0), res2.size(1))
        print("res2", res2)

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
    # norm = X.sum(dim=1, keepdim=True)
    # norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    # X = torch.div(X, norm)
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
def Conv1(in_planes, places, size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=size,stride=stride,padding = padding,bias=False),
        nn.BatchNorm2d(places),
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

        self.conv = Conv1(in_planes=512, places=10, size=3,stride=1,padding=1)

        self.conv_1 = Conv1(in_planes=10, places=10, size=2, stride=2, padding=0)
        self.conv_2 = Conv1(in_planes=10, places=10, size=2, stride=2, padding=0)

        self.softmax = nn.Softmax(dim=2)

        #self.s=nn.Sigmoid()

        self.fc1 = nn.Linear(10,64)
        self.fc2 = nn.Linear(64, 5)
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
        #x_global = torch.matmul(x1, x1_T)  # b*16*16
        #print("x_global", x_global)
        #x_global = norm(x_global)

        #print("x_global",x_global.cpu().data.numpy())

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

        #w_local = self.softmax(-x_global)
        #w_local =1-x_global
        # print("w_local",w_local.cpu().data.numpy())
        # print(w_local.shape)
        #print(w_local.cpu().data.numpy())
        #x_local_temp = torch.matmul(w_local, x1)  # 64 64 * 64 224*224 = 64 224*224
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

        # x_local_temp = norm(x_local_temp)
        # x_local_T = torch.transpose(x_local_temp, 2, 1)
        # #x_local = torch.bmm(x_local_temp, x_local_T)
        # x_local = torch.matmul(x_local_temp, x_local_T)
        # x_local = norm(x_local)
        # for i in range(x_local.shape[0]):
        #     for j in range(x_local.shape[1]):
        #         a = x_local[i, j, :]
        #         a = a.cpu().data.numpy()
        #         x_local[i, j, :] = norm(a)*10
        # print("x_local", x_local.cpu().data.numpy())
        # x_global = x_global*10
        # print(x_global)
        #print("x_local",x_local.cpu().data.numpy())
        # x_local = x_local*10
        return x_global

    def forward(self, x):
        origi = x
        x = self.features0(x)   # 16 64 56 56

        x = self.features1(x)   # 16 64 56 56

        x = self.features2(x)   # 16 128 28 28

        x = self.features3(x)   # 16 256 14 14

        x = self.features4(x)   # 16 512 7 7

        x = self.conv(x)   # * 10 7 7





        copy_x = x
        copy_x = self.conv_1(copy_x)

        copy_x = self.conv_2(copy_x)
        # print("w", copy_x.shape)
        # #print(copy_x)
        copy_x=copy_x.view(copy_x.size(0),copy_x.size(1))
        print("copy_x",copy_x)

        # t = copy_x.squeeze()
        # print(t.cpu().data.numpy())
        # filename = "C:\\Users\\guoyang\\Desktop\\label_4.txt"
        # file_csv = codecs.open(filename, 'a+', 'utf-8')  # 追加
        # writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(t.cpu().data.numpy())
        # print("保存文件成功，处理结束")

        # copy_x = l2norm(copy_x)
        # print(copy_x)

        # for i in range(10):
        #     a = x[0, i, :, :]
        #     a = a.squeeze()
        #
        #     draw_features(a.cpu().data.numpy(), i)
        # for i in range(10):
        #     a = x[0, i, :, :]
        #     a = a.squeeze()
        #
        #     draw_features(a.cpu().data.numpy(), i)
        # print(x.shape)


        # new
        # x_local = self.addconv1(x_local)
        # fea = self.pool(x_local)
        #
        # fea =fea.view(fea.size(0),fea.size(1))
        # t=  nn.Softmax(dim=1)
        # fea = t(fea)
        # print(fea.shape)

        #fea=self.CIM1(x)
        # for i in range(x_global.size(0)):
        #     t = x_global[i, :, :, :]
        #     print(t.shape)
        #     (fe, _) = t.symeig(eigenvectors=True)
        #     f = fe.view(-1)
        #     f = f.unsqueeze(dim=0)
        #     # print(f)
        #     # print(f.shape)
        #     if i == 0:
        #         fea1 = f
        #     else:
        #         fea1 = torch.cat((fea1, f), dim=0)
        # for i in range(x_local.size(0)):
        #     t = x_local[i, :, :, :]
        #     # print(t.shape)
        #     (fe, _) = t.symeig(eigenvectors=True)
        #     f = fe.view(-1)
        #     f = f.unsqueeze(dim=0)
        #     if i == 0:
        #         fea2 = f
        #     else:
        #         fea2 = torch.cat((fea2, f), dim=0)

        fea = copy_x
        #print("t2",t2)
        #print("copy_x",copy_x)
        fea = l2norm1(fea)
        #print(fea)
        #print("fea1",fea1)
        #fea2 = l2norm(fea2)
        #print(self.w1)
        #print("fea1", self.w1*fea1)
        #fea_t = self.w1*t1 +self.w2 *t2
        #fea = torch.cat((copy_x,fea_t),dim=1)
        #print(fea)


        # new
        # x = x.view(x.size(0), -1)
        # fea = self.fc1(x)

        res = self.fc1(fea)
        #res = self.dropout(res)
        res = self.fc2(res)

        #res = self.soft(res)
        #res = self.soft(res,lb)
        #print(res)
        # print(fea.shape)

        return x,res,copy_x,origi


def Res(pretrained=True):
    model = torchvision.models.resnet34(pretrained=True)
    #model = res.ResNet50()
    print(model)
    return Net(model)


if __name__ == "__main__":
    model = Res()
    # model1 = torchvision.models.resnet50(pretrained=True)
    # print(model1)
    # print(model)
    # unet = Net(in_channels=3)
    x = torch.zeros(5, 3, 224, 224)
    #model(x)
    # unet(x)
    lb = torch.randint(0, 5, (16,), dtype=torch.long)
    DoubleNet = DoubleNet(embeddingnet=model)
    y= torch.zeros(5, 3, 224, 224)
    DoubleNet.eval()
    DoubleNet(x, y)