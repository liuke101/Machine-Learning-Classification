# 训练模型
import os
import numpy as np
import torchvision
from torchvision import transforms

from Classifier import classify_data

'''
该函数是用于处理 CIFAR-10 数据的函数，通过这个函数获得的训练数据形状为(50000,3072)：
- 50000行，对应50000张图片，每一行代表一个图片数据，
- 3072 列，每一列代表一个像素，因为图片的大小是3 * 32 * 32=3072

5个训练batch：50000张图片
1个测试batch：10000张图片
测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。
'''


# 数据集官网上提供了python3读取CIFAR-10的方式，以下函数可以将数据集转化为字典类型：
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


# 创建训练样本和测试样本
def load_data_cifar10():
    # 下载数据集
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0～1之间
    trans = transforms.ToTensor()
    torchvision.datasets.CIFAR10(
        root="./cifar10_data", train=True, transform=trans, download=True)
    torchvision.datasets.CIFAR10(
        root="./cifar10_data", train=False, transform=trans, download=True)

    # 创建训练样本
    # 依次加载batch_data_i,并合并到x,y
    x = []
    y = []
    for i in range(1, 6):
        batch_path = 'cifar10_data/cifar-10-batches-py/data_batch_%d' % (i)
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b'data'].astype('float')
        train_label = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_label)
    # 将5个训练样本batch合并为50000x3072，标签合并为50000x1
    # np.concatenate默认axis=0，为纵向连接
    train_data = np.concatenate(x)
    train_label = np.concatenate(y)

    # 创建测试样本
    # 直接写cifar-10-batches-py\test_batch会报错，因此把/t当作制表符了，应用\\;
    # test_dict = unpickle("cifar10_data/cifar-10-batches-py/test_batch")

    # 建议使用os.path.join()函数
    testpath = os.path.join('cifar10_data/cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(testpath)
    test_data = test_dict[b'data'].astype('float')
    test_label = np.array(test_dict[b'labels'])

    return train_data, train_label, test_data, test_label


# 主函数
def main():
    # 这里的路径改需要该为自己的数据所在路径
    X_train, Y_train, X_test, Y_test = load_data_cifar10()

    print(X_train.shape, Y_train.shape)  # 可以打印训练数据的形状(50000, 3072) (50000,)

    # 使用机器学习算法来分类
    classify_data(X_train, Y_train, X_test, Y_test, classifier_type="KNN")


if __name__ == '__main__':
    main()
