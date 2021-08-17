import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from pyheatmap.heatmap import HeatMap

import torch
def transform(array):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224), Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
    new = transform(array)
    return new


def apply_heatmap(data,image):
    n = []
    data = cv2.resize(data, (224, 224), interpolation=cv2.INTER_CUBIC)
    norm_img = np.zeros(data.shape)
    data = data[:, :]
    data = cv2.normalize(data, norm_img, 0, 255, cv2.NORM_MINMAX)
    data = np.asarray(data, dtype=np.uint8)
    print(data.shape[0])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            tmp = [i, j, data[i][j]]
            n.append(tmp)

    print(n)
    image = image.cpu().data.numpy()

    image = image[0, 0, :, :]
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = np.asarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB)
    print(image)
    '''image是原图，data是坐标'''

    '''创建一个新的与原图大小一致的图像，color为0背景为黑色。这里这样做是因为在绘制热力图的时候如果不选择背景图，画出来的图与原图大小不一致（根据点的坐标来的），导致无法对热力图和原图进行加权叠加，因此，这里我新建了一张背景图。'''
    #background = Image.new("RGB", (image.shape[1], image.shape[0]), color=0)
    #print(background)
  # 开始绘制热度图
    hm = HeatMap(n)
    im = hm.heatmap()  # 热图
    # plt.imshow(im)
    # plt.show()
    #print(hm)
    #hit_img = hm.heatmap(base=background, r = 100) # background为背景图片，r是半径，默认为10
    #print(hit_img)
    # plt.figure()
    # plt.imshow(hit_img)
    # plt.show()
  #hit_img.save('out_' + image_name + '.jpeg')
    hit_img = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)#Image格式转换成cv2格式
    overlay = image.copy()
    alpha = 0.5 # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1) # 设置蓝色为热度图基本色蓝色
    #image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0) # 将背景热度图覆盖到原图
    image = cv2.addWeighted(hit_img, alpha, image, 1-alpha, 0) # 将热度图覆盖到原图
    plt.imshow(image)
    # plt.imshow(img,cmap='gray')
    # plt.savefig('/home/stellla/Desktop/3760_2pic/'+str(i)+".png")
    plt.show()
def draw_features(x,origi_img,i):

    origi_img = origi_img.cpu().data.numpy()
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = x[:, :]


    origi_img = origi_img[0, 0, :, :]
    # plt.imshow(origi_img, cmap='gray')
    # plt.show()
    norm_img = np.zeros(img.shape)
    ori_img = cv2.normalize(origi_img, norm_img, 0, 255, cv2.NORM_MINMAX)
    # ori_img = ori_img.astype('uint8')

    ori_img = np.asarray(ori_img, dtype=np.uint8)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
    a = ori_img.copy()
    # cv2.imwrite('/home/stellla/Desktop/test/' + 'o' + ".png", ori_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # print(img.shape)
    # pmin = np.min(img)
    # print(pmin)
    #
    # pmax = np.max(img)
    # img = (img - pmin) / (pmax - pmin + 0.000001)
    # print('img',img)

    # superimposed_img = img  * origi_img
    # cv2.addWeighted(img, 0.9, origi_img, 0.1, 0,img,cv2.CV_32F)
    # img = np.uint8(255 * img)  # 将热力图转换为RGB格式
    # print('img', img)
    norm_img = np.zeros(img.shape)
    norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    overlay = ori_img.copy()

    alpha = 0.25
    alpha1 = 0.05
    cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色蓝色
    cv2.addWeighted(overlay, alpha1, ori_img, 1 - alpha1, 0, ori_img)  # 将背景热度图覆盖到原图
    # cv2.imshow('or_image', ori_img)  # BGR moshi
    cv2.addWeighted(img, alpha, ori_img, 1 - alpha, 0, ori_img)

    # img_add = cv2.addWeighted(ori_img, 0.9, img, 0.1,0)

    # plt.savefig('/home/stellla/Desktop/3760_2pic/' + str(i) + ".png")
    cv2.imshow('ori_image', ori_img)
    # plt.show('ori_image', ori_img)
    #path = "C:\\Users\\guoyang\\Desktop\\label_0_pic\\" +str(i)
    cv2.imwrite("C:\\Users\\guoyang\\Desktop\\fig\\" + str(i) + ".png", ori_img,
                 [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    #cv2.imshow('ori_image', ori_img)  # BGR moshi
    #cv2.imwrite('/home/stellla/Desktop/test/' + 'g_1' + ".png", ori_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    cv2.waitKey(20)
    return ori_img,a,img