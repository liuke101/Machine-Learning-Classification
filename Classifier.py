# 构建分类器,括号里面那些都是超参数，可以自己调节，俗称【调参】
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# 构建分类器,括号里面那些都是超参数，可以自己调节，俗称【调参】
def get_classifier(classifier_type):
    if classifier_type == "RandomForest":  # 随机森林分类器
        return RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
    elif classifier_type == "KNN":  # KNN分类器
        return KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                                    p=2, metric='minkowski', metric_params=None, n_jobs=None)
    else:
        raise ValueError("Invalid classifier type")


# 使用机器学习算法分类
def classify_data(X_train, Y_train, X_test, Y_test, classifier_type="RandomForest"):
    # 使用机器学习算法来分类
    # 选择随机森林分类器或者是KNN分类器
    Classifier = get_classifier(classifier_type)  # RandomForest or KNN
    Classifier.fit(X_train, Y_train)  # 使用训练集进行训练
    print('.................训练完成.............')

    # 使用测试集进行测试
    
    # 根据结果绘制折线图
    # 假设Classifier是你的分类器，X_test和Y_test是测试集的特征和标签
    Y_pred = Classifier.predict(X_test)

    # 获取classification_report, 输出字典
    report = classification_report(Y_test, Y_pred, output_dict=True)
    # print('.................打印分类结果的信息.............')
    # print(classification_report(Y_test, Classifier.predict(X_test))) 

    # 从报告中提取指标值
    classes = [str(i) for i in range(10)] 
    precision = [report[class_label]['precision'] for class_label in classes]
    recall = [report[class_label]['recall'] for class_label in classes]
    f1_score = [report[class_label]['f1-score'] for class_label in classes]

    # 绘制曲线图
    plt.figure(figsize=(10, 6))

    plt.plot(classes, precision, marker='o', label='Precision')
    plt.plot(classes, recall, marker='o', label='Recall')
    plt.plot(classes, f1_score, marker='o', label='F1-score')

    # 添加标题和标签
    plt.title('Precision, Recall, and F1-score for Each Class')
    plt.xlabel('Class')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 利用scikit-learn自带的库【计算多分类混淆矩阵】
    mcm = multilabel_confusion_matrix(Y_test, Y_pred)  # mcm即为混淆矩阵
    # 通过混淆矩阵可以得到tp,tn,fn,fp
    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    print('......................打印混淆矩阵................')
    print(mcm)
