from numpy import *
import operator
# 使用matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt


# 创建样本
def createDataSet():
    group = array([[1.0, 1.1, ], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类函数
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 行数
    # 以下三行计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 计算输入标签和其他标签的距离差
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 对于二维数组 axis=1表示按行相加 , axis=0表示按列相加。每列为不同点之间的距离cha

    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 获得排序后各元素索引
    # print(distances,sortedDistIndicies)
    classCount = {}

    # 以下两行 选取距离最近的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get 方法 如果字典里面找不到这个值，返回输入的参数0
    # print(classCount.items())
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # operator.itemgetter()函数 获取对象某些位置的数据
    return sortedClassCount[0][0]


# 将txt文件转化为可以识别的矩阵   返回值：原始数据列表  对应属于的类别列表
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)  # 文件行数
    returnMat = zeros((numberOfLines, 3))  # 多行三列
    classLabelVector = []

    index = 0
    # 解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # return为一个矩阵 这表示把读取到的数据赋值给矩阵的第几行
        classLabelVector.append(int(listFromLine[-1]))  # 列表中最后一列
        index += 1
    return returnMat, classLabelVector


# 归一化函数
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 参数0表示获取矩阵中每列中最小值，如说输入是三列矩阵，输出也为三列矩阵
    maxVals = dataSet.max(0)  # 参数0表示获取矩阵中每列中最大值，如果是1就是获取行的
    ranges = maxVals - minVals
    m = dataSet.shape[0]  # 获取行值
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 直接是矩阵中的数值相除，而不是矩阵除法
    return normDataSet, ranges, minVals


# 进行测试文件
def datingClassTest():
    hoRatio = 0.1
    # 获取数据
    datingDataMat, datingLabels = file2matrix(txtfile)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 获取行值
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 选取前n个样本 和之后所有的样本进行比较
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


if __name__=='__main__':
    # 显示分类数据
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
    print(classify0([0.2, 0.5], group, labels, 3))

    # 打开文件，预览前20个数据
    txtfile = 'D:\PythonCode\jupyter\Machine\machinelearninginaction\Ch02\datingTestSet2.txt'
    datingDataMat, datingLabels = file2matrix(txtfile)
    datingDataMat20 = datingDataMat[0:20]
    print(datingDataMat20)

    # 画图表示
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 25.0 * array(datingLabels), 15.0 * array(datingLabels))
    # 旅行时间（第一列） x轴玩游戏所占时间百分比（第二列）  y轴每周所消耗冰淇淋公升数（第三列）
    # datingDataMat20[1:2,1]   #这句话应该这么理解 [1:2,1] 中,为分隔符,取第二列中的元素切片，第一个，第二个做切片，
    # scatter(x, y, 点的大小, 颜色，标记)，这是最主要的几个用法  颜色可以直接使用几个不同的数字代替，比如1，2，3代表三种不同颜色，
    # 也可以使用字母表示
    ax.set_title('plot graph')
    # plt.legend('不喜欢','魅力一般','极具魅力')
    plt.xlabel('tour time')
    plt.ylabel('ice cream')
    plt.show()

    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat, ranges, minVals)

    # 测试函数
    datingClassTest()
