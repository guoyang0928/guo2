# coding = UTF-8

import numpy as np
import os

from PIL import Image
from torchvision import transforms
import torch
import shutil

# INPUT_SIZE = 224
class Thyroid():

    def __init__(self, root, name1,is_train,  data_len=None):
        # root文件夹下必须有annotations.txt,train_test_split.txt
        # annotation里边写了image_name，label;     train_test_split.txt写了image_name,is_train;
        # feature_num是第几个feature,和文件中的feature在字符串中所在的位置一样
        self.root = root
        self.is_train = is_train

        img_file = open(os.path.join(self.root, name1))
        list = []

        label_list = []

        for line in img_file:
            # if line[:-1].split(' ')[0] == '000624.jpg':
            #     print(line)
            # temp = int(line[:-1].split(' ')[1])
            # if temp == 4:
            #     list.append(line[:-1].split(' ')[0])
            #     label_list.append(int(line[:-1].split(' ')[1]))
            #list.append(line[:-1].split(' ')[0])  # line[:-1]去除文本最后一个字符（换行符）后剩下的部分 以空格为分隔符保留最后一段
            # if label == 1 or label == 2 or label == 3 :
            #     label_list.append(0)  #恶性
            #
            # if label == 4 or label == 5 or label == 6 or label == 7:
            #     label_list.append(1)  #良性

            list.append(line[:-1].split(' ')[0])
            label_list.append(int(line[:-1].split(' ')[1]))
            #label_list.append(int(line[:-1].split('.')[0].split('_')[1])-4)

        print(len(list))
        print(len(label_list))
        print(list)

        if self.is_train:
            self.list=list
            self.train_img = [Image.open(os.path.join(self.root, 'thyroid', train_file)).convert('RGB') for train_file in
                              list[:data_len]]
            #print(data_len)
            self.train_label = label_list[:data_len]
        #测试集  读取图片与对应类别
        if not self.is_train:
            self.list=list
            self.test_img = [Image.open(os.path.join(self.root, 'thyroid', test_file)).convert('RGB') for test_file in
                             list[:data_len]] #读出来为numpy类型
            self.test_label = label_list[:data_len]


    def __getitem__(self, index): #迭代读取
        if self.is_train:
            img_name=self.list[index]
            # print('----------------------',img_name)
            img, target = self.train_img[index], self.train_label[index]
            #print(img.shape)
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)   #
            #img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((224,224), Image.BICUBIC)(img)
            #img = transforms.Resize((224), Image.BICUBIC)(img)
            #img = transforms.CenterCrop(224)(img)
            # img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
            img = transforms.RandomVerticalFlip(p=0.5)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:

            img_name=self.list[index]


            img, target = self.test_img[index], self.test_label[index]
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            #img = Image.fromarray(img, mode='RGB')
            # img =transforms.Resize(224, interpolation=Image.BICUBIC)(img)
            # img = transforms.CenterCrop(224)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            # img = transforms.Resize((224), Image.BICUBIC)(img)   # 重置图像分辨率
            # img = transforms.CenterCrop(224)(img)
            # img = transforms.CenterCrop(INPUT_SIZE)(img)                #功能：依据给定的size从中心裁剪
            img = transforms.ToTensor()(img)                            # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)   #对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc

        return img, target,img_name

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':

    dataset = Thyroid(root='./data',name1 = 'filename_train0.txt',is_train=True,data_len=1)
    print(dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                   shuffle=True, num_workers=0, drop_last=False)
    # dataset = (root='../data/new_thyroid_data_20190923/thyroid_data_20191031/all/',is_train=true,feature_num=f_num)
    print(len(dataset.train_img))
    # print(dataset.train_img)
    print(len(dataset.train_label))
    print(dataset.train_label)
    for data in train_dataloader: #调取get_item
        img, label = data[0].cuda(), data[1].cuda()
        print(data[0].size(), data[1]) #返回图片大小(batch * chan * h* w)和标记





