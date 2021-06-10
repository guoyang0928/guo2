import os
import csv
import json
import random
import pickle
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable


class TripletGenerator(object):
    def __init__(self, root, base_path, filenames, meta):

        fnamelistfile = os.path.join(root,filenames)

        self.fnamelist = []
        self.labels = []
        with open(fnamelistfile, 'r') as f:
            for fname in f:
                self.fnamelist.append(fname.strip().split(' ')[0])
                self.labels.append(fname.strip().split(' ')[1])
        label = self.labels
        fname = self.fnamelist
        # print(label)
        # print(fname)





        # 结果是[[0, 3, 35, 1, 37, 0, 55, 0, 59, 0, 128, 2, 156], [1, 0, 55, 3, 93]...]

        self.category = meta['ATTRIBUTES']
        self.category_num = meta['ATTRIBUTES_NUM']
        # ['texture-related', 'fabric-related', 'shape-related', 'part-related', 'style-related']
        # {'texture-related': 156, 'fabric-related': 218, 'shape-related': 180, 'part-related': 216, 'style-related': 230}
        self.category_dict = []

        # for c in self.category:
        #     self.category_dict[c] = []

        for i in range(len(self.labels)):

            self.category_dict.append([fname[i], label[i]])
        #print("d",self.category_dict)
                # 获取每个属性对应的图片name以及子标签标记

    def get_triplet(self, num_triplets):
        triplets = []

        for i in range(num_triplets):


            # cate_sub = random.randint(1, self.category_num[self.category[cate_r]])
            cate_sub = random.choice([0,1,2,3,4])
            # if epoch<40:
            #      cate_sub = random.choice([0])
            # elif epoch < 80 and epoch>=40:
            #     cate_sub = random.choice([1])
            # elif epoch < 120 and epoch>=80:
            #     cate_sub = random.choice([2])
            # elif epoch < 160 and epoch>=120:
            #     cate_sub = random.choice([3])
            # elif epoch < 200 and epoch>=160 :
            #     cate_sub = random.choice([4])
            #print("dd",self.category_dict[0][1])

            while True:


                a = random.randint(0, len(self.category_dict) - 1)
                #print(a)
                if int(self.category_dict[a][1]) == cate_sub:
                     break
            #print(a)

            # while True:
            #     c = random.randint(0, len(self.category_dict) - 1)
            #
            #     if int(self.category_dict[c][1]) == cate_sub and c != a:
            #         break
            #print(c)

            while True:
            #     if cate_sub == 1:
            #         cate_sub1=random.choice([0,2])
            #     elif cate_sub == 0:
            #         cate_sub1=1
            #     elif cate_sub == 2:
            #         cate_sub1 = random.choice([1,3])
            #     elif cate_sub == 3:
            #         cate_sub1 = random.choice([2,4])
            #     elif cate_sub == 4:
            #         cate_sub1 = 3
                # else:
                cate_sub1 = random.choice([0,1,2,3,4])
                # print("cate_sub1:",cate_sub1)
                if cate_sub1 != cate_sub:
                    break
            while True:
                b = random.randint(0, len(self.category_dict) - 1)
                if int(self.category_dict[b][1]) == cate_sub1:
                    break

            triplets.append([self.category_dict[a],
                             cate_sub,
                             self.category_dict[b],
                             cate_sub1,
                             # self.category_dict[c],
                             # cate_sub

                             ])
        print(triplets)

        return triplets


class MetaLoader(object):
    def __init__(self, root, dataset):
        self.data = json.load(open(os.path.join(root, 'meta.json')))[dataset]

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)

        return cls.__instance


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, base_path, num_triplets, filenames, transform=None,
                 loader=default_image_loader):
        self.root = root
        self.base_path = base_path
        self.num_triplets = num_triplets

        self.meta = MetaLoader(self.root, self.base_path)
        self.filenames = filenames


        self.triplet_generator = TripletGenerator(self.root, self.base_path, self.filenames, self.meta.data)
        self.triplets = self.triplet_generator.get_triplet(self.num_triplets)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):

        path1 = self.triplets[index][0][0]
        path2 = self.triplets[index][2][0]
        #path3 = self.triplets[index][4][0]

        label1 = self.triplets[index][1]
        label2 = self.triplets[index][3]
        #label3 = self.triplets[index][5]

        # print('1:',path1,label1)
        # print('2:',path2, label2)
        # print('3',path3,label3)

        if os.path.exists(os.path.join(self.root, self.base_path, path1)):
            img1 = self.loader(os.path.join(self.root, self.base_path, path1))
        else:
            print('not found', path1)
            return None

        if os.path.exists(os.path.join(self.root, self.base_path, path2)):
            img2 = self.loader(os.path.join(self.root, self.base_path, path2))
        else:
            print('not found', path2)
            return None

        # if os.path.exists(os.path.join(self.root, self.base_path, path3)):
        #     img3 = self.loader(os.path.join(self.root, self.base_path, path3))
        # else:
        #     print('not found', path3)
        #     return None

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            #img3 = self.transform(img3)

        # print("label:",label1,label2)
        # 2是不一样的
        #print(img1.size)
        return img1, label1, img2, label2

    def __len__(self):
        return len(self.triplets)

    def refresh(self):
        self.triplets = self.triplet_generator.get_triplet(self.num_triplets)
        #print("refresh")



if __name__ == '__main__':
    loader = TripletImageLoader('./data/', "thyroid", 2, "filenames_train0.txt")
    for batch_idx, (data1, label1, data2, label2, data3, label3) in enumerate(loader):
        print(1)