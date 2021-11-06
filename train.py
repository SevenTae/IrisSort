import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

#数据预处理：获取数据集，划分
def getdata():
    from sklearn.datasets import load_iris
    import numpy as np
    train_data = load_iris()
    data = train_data['data']
    labels = train_data['target'].reshape(-1, 1)
    total_data = np.hstack((data, labels))
    np.random.shuffle(total_data)
    train = total_data[0:110, :-1]
    test = total_data[110:, :-1]
    train_label = total_data[0:110, -1].reshape(-1, 1)

    test_label = total_data[110:, -1].reshape(-1, 1)
    print("------统计----------")
    print("统计train")
    t0 = 0
    t1 = 0
    t2 = 0
    for i in range(train_label.shape[0]):
        if train_label[i][0] == 0:
            t0 +=1
        elif train_label[i][0] ==1:
            t1+=1
        elif train_label[i][0] ==2:
            t2+=1
    print("train统计结果：",t0,t1,t2)
    print("-------统计test------")
    tt0 = 0
    tt1 = 0
    tt2 = 0
    for i in range(test_label.shape[0]):
        if test_label[i][0] == 0:
            tt0 += 1
        elif test_label[i][0] == 1:
            tt1 += 1
        elif test_label[i][0] == 2:
            tt2 += 1
    print("test统计结果：", tt0, tt1, tt2)
    print("------------------")

    return data, labels, train, test, train_label, test_label

#网络结构的设计
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.fc1=nn.Linear(4,8)
        self.act1 = nn.Sigmoid()
        self.fc2=nn.Linear(8,3)
        self.act2=nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.softmax(out)
        return out
#训练以及验证的模块函数
def train_model(model,train_loder,optimizer,epoch,test_loder):
    # 模型训练
    writer1 =SummaryWriter("./log")
    writer2 = SummaryWriter("./log")
    train_loss = 0.0
    model.train()
    for i in range(epoch):
        print("----------------第{}轮训练----------".format(i + 1))
        for batch_index, (data, target) in enumerate(train_loder):

            y = torch.reshape(target, [BATCH_SIZE])

            # 梯度清零
            optimizer.zero_grad()
            # 训练后的结果
            output = model(data)
            # 计算损失
            loss = F.cross_entropy(output, y)
            train_loss = loss.item() + train_loss
            # #找到概率值最大的下标 这个主要是后边用来计算准确率的
            # pred = output.argmax(dim=1)
            # 反向传播
            loss.backward()
            optimizer.step()
        print("epoch{}的损失值为{}".format(i + 1, train_loss))
        writer1.add_scalar("loss",train_loss,i)
        train_loss=0.0
         #测试

        print("----------------第{}轮测试----------".format(i + 1))
        # 模型验证
        model.eval()
        # 统计正确率
        correct = 0.0
        # 测试损失
        test_loss = 0.0
        # 进行测试
        with torch.no_grad():  # 不会计算梯度也不会进行反向传播
            for data, target in test_loder:
                y = torch.reshape(target, [BATCH_SIZE])
                # 测试数据

                output = model(data)

                # 计算测试损失
                test_loss += F.cross_entropy(output, y).item()
                # 找到概率值最大的下标
                pred = output.argmax(dim=1)  # 值，索引。dim是维度1代表按照行找
                # 累计正确的数目
                correct += pred.eq(target.view_as(pred)).sum().item()
            # 计算test的loss
            # test_loss /= len(test_loder.dataset)
            print("测试损失值为{}".format(test_loss))
            print("epoch{}的测试的精度为{}".format(i+1,correct / len(test_loder.dataset)))
            accu = correct / len(test_loder.dataset)
            writer2.add_scalar("Accuracy", accu, i)
        torch.save(model.state_dict(), "./model_data/net_{}.pth".format(i + 1))
#使用训练好的模型进行预测
def model_predic(model,data,model_data_path):
    model.load_state_dict(torch.load(model_data_path))  #加载模型参数
    model.eval()
    with torch.no_grad():
        output = model(data)
        # 找到概率值最大的下标
        pred = output.argmax(dim=1)  # 值，索引。dim是维度
        pred = np.array(pred)
        predict = int(pred[0])
        if predict == 0:
            print("预测结果为：", "0山鸢尾")
            return "山鸢尾"
        elif predict == 1:
            print("预测结果为：", "1变色鸢尾")
            return "变色鸢尾"
        elif predict == 2:
            print("预测结果为：","2维吉尼亚鸢尾")
            return "维吉尼亚鸢尾"



def  qtgui(sele,sewd,pale,pawd):
    seleq = sele
    sewdq = sewd
    paleq = pale
    pawdq = pawd
    predictData = np.array([  # 测试样本
        [seleq, sewdq, paleq, paleq]  # 2:5.9 ,3.0,  5.1 ,1.8   1:5.9 3.2 4.8 1.8  0:5.  3.4 1.5 0.2
    ])
    print(predictData)
    predict_data = torch.from_numpy(predictData).float()  # 转化为tensor
    model_data_path = "D:/PythonStudy/MLZY/network/fortensorboard/model_data/net_777_01.pth"  # 选择网络模型的权重
    model = Mynet()
    output = model_predic(model, predict_data, model_data_path)  # 调用预测函数
    return output


if __name__ == '__main__':
    epoch=1000 #迭代次数
    BATCH_SIZE = 10 #batch size
    data, labels, train, test, train_label, test_label = getdata()
    mynet = Mynet() #定义网路
    #对数据进行预处理：转化成pytorch所使用的tensor类型
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(), torch.from_numpy(train_label).long())

    optimizer = optim.Adam(mynet.parameters(), lr=0.001)#定义一个优化器
    #训练集和测试集的加载
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset=Data.TensorDataset(torch.from_numpy(test).float(), torch.from_numpy(test_label).long())
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train = False
    predict = False
    '''模型的训练'''
    if train:
        train_model(mynet, train_loader, optimizer, epoch, test_loader)
    '''模型的测试'''




