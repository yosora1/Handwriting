# This is a sample Python script.
#图像预处理
from torchvision import transforms
import numpy as np
import torch
import torchvision
#激活函数
import torch.nn.functional as F
from torch import nn#model
from torch import optim#优化器

train_batch_size=64#分批训练大小
test_batch_size=1000#分批测试大小
img_size=28
#加载数据集
def get_dataloader(train=True):
    assert isinstance(train,bool)#train必须是bool类型
    #准备数据集，其中0.1307，0.3081为mnist数据的均值和标准差，这样操作能够对其进行标准化
    #因为mnist只有一个通道，所以元组中只有一个值
    dataset=torchvision.datasets.MNIST('E:\\Py\\PycharmProjects\\MinisterHandwritten\\data',train=train,download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                                       ]))
    #准备数据迭代器
    batch_size=train_batch_size if train else test_batch_size#分批训练
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1=torch.nn.Linear(28*28*1,28)
        self.fc2=torch.nn.Linear(28,10)

    def forward(self,x):
        x=x.view(-1,28*28*1)#[batch_size,28*28]（输入层）
        x=self.fc1(x)#[batch_size,28]将28*28压缩成28（隐藏层）
        x=F.relu(x)#[batch_size,28]（隐藏层激活函数）relu:线性修正函数
        x=self.fc2(x)#[batch_size,10]（输出层）
        return F.log_softmax(x,dim=-1)#（输出层激活函数）

#pytorch的两种实现交叉熵损失方法
'''
1/ 
    criterion=nn.CrossEntropyLoss()
    loss=criterion(input,target)
2/
对输出值计算softmax和取对数
output=F.log_softmax(x,dim=-1)
使用torch中带权损失
loss=F.nll_loss(output,target)
'''

mnist_net=MnistNet()
optimizer=torch.optim.Adam(mnist_net.parameters(),lr=0.001)
train_loss_list=[]
train_count_list=[]

'''
训练模型过程
1.实例化模型，设置模型为训练模式
2.实例化优化器类，实例化损失函数
3.获取，遍历dataloader
4.梯度置为0
5.进行向前计算
6.计算损失
7.反向传播
8.更新参数
'''

def train(epoch):#epoch训练次数
    mnist_net.train(True)#True训练模式，False评估模式
    train_dataloader=get_dataloader(True)#得到训练集
    for idx,(data,target) in enumerate(train_dataloader):#data：tensor, target:标签
        optimizer.zero_grad()#梯度置为0
        output=mnist_net(data)#向前计算
        loss=F.nll_loss(output,target)#对数似然损失
        loss.backward()#反向传播,产生梯度
        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch,idx*len(data),
                                                                       len(train_dataloader.dataset),
                                                                       100. * idx /len(train_dataloader),
                                                                       loss.item()))
            train_loss_list.append(loss.item())
            train_count_list.append(idx*train_batch_size+(epoch-1)*len(train_dataloader))
    print("训练结束")

'''
评估过程
1、不需要计算梯度
2、需要收集损失和准确率，用来计算平均损失和平均准确率
3、损失的计算和训练时候损失的计算方法相同
4、准确率的计算：
    模型的输出为[batch_size,10]的形状
    其中最大值的位置就是其预测的目标值（预测值进行过softmax后为概率，softmax中分母都是相同的，分子越大，概率越大）
    最大值的位置获取方法可以使用torch.max，返回最大值和最大值的位置
    返回最大值的位置后，和真实值（[batch_size]）进行对比，相同表示预测成功
'''
def test():
    test_loss=0
    correct=0
    mnist_net.eval()
    test_dataloader=get_dataloader(train=False)
    with torch.no_grad():
        for data,target in test_dataloader:
            output=mnist_net(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.data.max(1,keepdim=True)[1]#获取最大值的位置，[batch_size]
            correct+=pred.eq(target.data.view_as(pred)).sum()
    test_loss/=len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f},Accuracy: {}/{}({:.2f}%)\n'.format(
        test_loss,correct,len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)
    ))



epoch=1
for i in range(epoch):
    train(i)

test()