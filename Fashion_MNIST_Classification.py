# 训练模型
import os

import numpy as np
import torchvision
from torchvision import transforms

from Classifier import classify_data

'''
该函数是用于处理fashion mnist数据的函数，因为原始是用ubyte格式保存，通过这个函数获得的训练数据形状为(60000,784)：
- 60000行，每一行代表一个图片数据，
- 784列，每一列代表一个像素，因为图片的大小是28x28=784

这也是机器学习方法的惯用套路:
- 数据统一格式：(样本数，每个样本特征数)。这样，每次输入算法的就为一行数据，也就是一个样本。
- 输出（也即标签）为(60000,)也就是60000维的列向量，每一个数代表该样本的类型
'''


# 读取数据，返回数据和标签
def load_data_fashion_mnist(path, kind='train'):
    # 下载数据集
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0～1之间
    trans = transforms.ToTensor()
    torchvision.datasets.FashionMNIST(
        root="./fashionmnist_data", train=True, transform=trans, download=True)
    torchvision.datasets.FashionMNIST(
        root="./fashionmnist_data", train=False, transform=trans, download=True)
    
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    # 打开二进制文件并读取
    # lbpath.read()函数读取整个文件的内容，并返回一个字节串（bytes）
    # np.frombuffer()函数将这个字节串转换为一个NumPy数组
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


# 主函数
def main():
    # 这里的路径改需要该为自己的数据所在路径
    X_train, Y_train = load_data_fashion_mnist('fashionmnist_data/FashionMNIST/raw', kind='train')  # 处理/读取训练数据
    X_test, Y_test = load_data_fashion_mnist('fashionmnist_data/FashionMNIST/raw', kind='t10k')  # 处理/读取测试数据
    # print(X_train.shape,Y_train.shape)#可以打印训练数据的形状(60000, 784) (60000,)

    # 使用机器学习算法来分类
    classify_data(X_train, Y_train, X_test, Y_test, classifier_type="KNN")


if __name__ == '__main__':
    main()
