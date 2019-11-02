import numpy as np
import matplotlib.pyplot as plt

n = 100
countNumber = 500000
orig_data = np.zeros((3, 100))
aData = np.zeros((3, 50))
aData = (np.random.normal(0, 1, 50), np.random.normal(0, 1, 50), np.zeros(50))
bData = np.zeros((3, 50))
bData = (np.random.normal(2, 1, 50), np.random.normal(2, 1, 50), np.ones(50))
orig_data = np.concatenate((aData, bData), axis=1)
t_data = (np.transpose(orig_data))
np.random.shuffle(t_data)
# print(t_data[:, :])
# np.random.shuffle(orig_data)

t_data = np.c_[np.ones((100, 1)), t_data]  # loc:插入列的位置 column:列名 value:列值
X = t_data[:, 0:3]
Y = t_data[:, 3:4]

filename='test.txt'
'''
with open(filename,'w') as fileobject: #使用‘w’来提醒python用写入的方式打开
    for i in range(100):
        fileobject.write(str(t_data[i, 1]))
        fileobject.write(' ')
        fileobject.write(str(t_data[i, 2]))
        fileobject.write(' ')
        fileobject.write(str(t_data[i, 3]))
        fileobject.write('\n')

'''


def loadDataSet():   #读取数据（这里只有两个特征）
    data = np.zeros((100, 4))
    fr = open(filename)
    i = 0
    for line in fr.readlines():
        lineArr = line.strip().split()
        data[i, 0] = 1
        data[i, 1] = lineArr[0]
        data[i, 2] = lineArr[1]
        data[i, 3] = lineArr[2]
        i +=1
    return data

t_data = loadDataSet()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(x, thetas):
    return sigmoid(np.dot(x, thetas.T))


def cost(x, y, thetas):
    a = np.multiply(-y, np.log(model(x, thetas)))
    b = np.multiply(1 - y, np.log(1 - model(x, thetas)))
    return np.sum(a - b) / (len(x))


def gradient(x, y, thetas):
    grad = np.zeros(thetas.shape)
    error = (model(x, thetas) - y).ravel()
    # print(error.shape)
    # print(x[:, 0].shape)
    for j in range(len(thetas.ravel())):
        # for each parmeter
        term = np.multiply(error, x[:, j])
        grad[0, j] = np.sum(term) / len(x)
    return grad


def descent(data, alpha):
    # 梯度下降求解
    i = 0
    cols = data.shape[1]
    x = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    theta = np.zeros([1, 3])  # 构造3个参数
    grad = np.zeros(theta.shape)
    # 计算的梯度
    costs = [cost(x, y, theta)]
    # 损失值

    while i < countNumber:
        grad = gradient(x[:, :], y[:, :], theta)

        theta = theta - alpha*grad
        # 参数更新
        costs.append(cost(x, y, theta))
        # 计算新的损失
        i += 1
    return theta, costs


theta, costs = descent(t_data, 0.0001)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(aData[0], aData[1], s=30, c='b', marker='o', label='A')
ax.scatter(bData[0], bData[1],  s=30, c='r', marker='x', label='B')
ax.legend()
x1 = 0
x2 = 3
y1 = -(theta[0, 0] + theta[0, 1] * x1)/theta[0, 2]
y2 = -(theta[0, 0] + theta[0, 1] * x2)/theta[0, 2]
plt.plot([x1, y1], [x2, y2], 'b')
plt.show()

d_countNumber = 800


def d_descent(data, alpha):
    # 梯度下降求解+正则项
    i = 0
    cols = data.shape[1]
    x = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    theta = np.ones([1, 3])  # 构造3个参数
    grad = np.zeros(theta.shape)
    # 计算的梯度
    costs = [cost(x, y, theta)]
    # 损失值
    limda = 0.01
    while i < d_countNumber:
        grad = gradient(x[:, :], y[:, :], theta)
        # print(theta)
        theta = theta - alpha*grad - limda * theta
        # 参数更新
        costs.append(cost(x, y, theta))
        # 计算新的损失
        i += 1
    return theta, costs


d_theta, d_costs = d_descent(t_data, 0.0001)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(aData[0], aData[1], s=30, c='b', marker='o', label='A')
ax.scatter(bData[0], bData[1],  s=30, c='r', marker='x', label='B')
ax.legend()
y0 = -(d_theta[0, 0] + d_theta[0, 1] * x1)/d_theta[0, 2]
y00 = -(d_theta[0, 0] + d_theta[0, 1] * x2)/d_theta[0, 2]
plt.plot([x1, y0], [x2, y00], 'g')
plt.show()

n_countNumber = 5
def newton(data):

    cols = data.shape[1]
    x = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    n_theta = np.ones((1, 3))
    m = 0
    while m < n_countNumber:
        a = np.eye(100)
        for i in range(100):
            a[i, i] = sigmoid(x[i] * n_theta)[0, 0] * (1 - sigmoid(x[i] * n_theta)[0, 0])
        print(a)
        h = x.T.dot(a).dot(x)
        print(h)

        error = y - sigmoid(x .dot(n_theta.T))
        # weights = weights + H ** -1 * dataMat.transpose() * error
        # print((h ** -1 ).shape)
        # print(x.T.size)
        # print((h ** -1).size)
        n_theta = n_theta + h ** -1 * x.T .dot(error)
        # print (h ** -1)
        # n_theta = n_theta + (h ** -1) .dot(gradient(x[:, :], y[:, :], n_theta).T).reshape(1,3)
        # 参数更新
        m += 1
        print(m)
    return n_theta


def n_newton(numIter):
    dataMat = np.mat(X)
    labelMat = np.mat(Y).transpose()
    m, n = np.shape(dataMat)
    # 对于牛顿法，如果权重初始值设定为1，会出现Hessian矩阵奇异的情况.
    # 原因未知，谁能告诉我
    # 所以这里初始化为0.01
    weights = np.mat(np.ones((n, 1))) - 0.99
    weightsHis = [np.mat(np.ones((n, 1)) - 0.99)]  # 权重的记录，主要用于画图
    for _ in range(numIter):
        A = np.eye(m)
        for i in range(m):
            h = sigmoid(dataMat[i] * weights)
            hh = h[0, 0]
            A[i, i] = hh * (1 - hh)
        error = labelMat - sigmoid(dataMat * weights)
        H = dataMat.transpose() * A * dataMat  # Hessian矩阵
        print(H)
        weights = weights + H ** -1 * dataMat.transpose() * error

        weightsHis.append(weights)

    return weights, weightsHis


# t_theta, tt = n_newton(n_countNumber)
t_theta = newton(t_data)

print(t_theta)

print(cost(X,Y,theta))
# theta[0] + theta[1] * lx + theta[2] *ly = 0
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(aData[0], aData[1], s=30, c='b', marker='o', label='A')
ax.scatter(bData[0], bData[1],  s=30, c='r', marker='x', label='B')
ax.legend()
y11 = -(t_theta[0, 0] + t_theta[0, 1] * x1)/t_theta[0, 2]
y22 = -(t_theta[0, 0] + t_theta[0, 1] * x2)/t_theta[0, 2]

plt.plot([x1, y11], [x2, y22], 'k')
plt.show()