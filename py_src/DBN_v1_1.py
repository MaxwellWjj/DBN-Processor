import numpy as np
import copy
import struct
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DBN:

    def __init__(self):
        self.size_h1 = 200  # 隐层1节点数
        self.size_h2 = 100  # 隐层2节点数
        self.size_v1 = 784  # 显层1节点数
        self.size_v2 = self.size_h1
        self.size_x = self.size_h2 # neuron network input
        self.size_softmax_layer = 10
        self.size_y = self.size_softmax_layer   ## neuron network output (10 modes: 0--9)
        self.n = 50  # 循环重复训练次数
        self.alpha1 = 0.002  # RBM
        self.alpha2 = 0.002 #### bp_net ####
        self.lam = 0.002 ####bp_net ####
        self.batch_size = 6000  # 样本数
        self.test_size = 1000 # 测试集数
        self.reset()  # 重置参数
        # self.plot_num = 16  # 画图数
        # self.k = 1  # CD-k  k次采样
        # self.bound = 0.5
        # self.plot_length = 10  # 每个图片的像素长宽，防止不是正方形
        # self.plot_width = 10
    def reset(self):  # 用来重置w,a,b
        np.random.seed(0)
        self.w1 = 1 / np.sqrt(self.size_v1 * self.size_h1) * np.random.rand(self.size_h1, self.size_v1)  # 权重矩阵
        self.w2 = 1 / np.sqrt(self.size_v2 * self.size_h2) * np.random.rand(self.size_h2, self.size_v2)  # 权重矩阵
        self.a1 = np.zeros((self.size_v1, 1))  # 显层偏置
        self.b1 = np.zeros((self.size_h1, 1))  # 隐层偏置
        self.a2 = np.zeros((self.size_v2, 1))  # 显层偏置
        self.b2 = np.zeros((self.size_h2, 1))  # 隐层偏置
        self.delta_w1 = np.zeros((self.size_h1, self.size_v1))
        self.delta_a1 = np.zeros((self.size_v1, 1))
        self.delta_b1 = np.zeros((self.size_h1, 1))
        self.delta_w2 = np.zeros((self.size_h2, self.size_v2))
        self.delta_a2 = np.zeros((self.size_v2, 1))
        self.delta_b2 = np.zeros((self.size_h2, 1))
        self.h1 = np.zeros((self.batch_size,self.size_h1,1))
        self.h2 = np.zeros((self.batch_size,self.size_h2,1))
        # self.bias = np.zeros((self.size_h2 + self.size_v1, 1))
        # self.delta_bias = np.zeros((self.size_h2 + self.size_v1, 1))
        # self.sparse_matrix = np.ones((self.size_h, self.size_v1))
        ########## fully-connected neuron network ############## (BPnet)
        self.w_bp = np.zeros((self.size_y,self.size_x))
        ################  predict ####################
        self.h1_pre = np.zeros((self.test_size,self.size_h1,1))
        self.h2_pre = np.zeros((self.test_size,self.size_h2,1))
        self.z_out = np.zeros((self.test_size,self.size_softmax_layer,1))
        self.labels_pre = np.zeros((self.test_size,1),dtype='uint8')
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def softmax(self,y):
        s_y = np.exp(y)/np.sum(np.exp(y))
        return s_y
    def forward(self, v1, w, b):
        P_h1 = self.sigmoid(b + np.dot(w, v1))  # 直接带入矩阵运算   w是 size(h)*size(v) x0是列向量size(v)*1
        temp = np.random.rand(P_h1.size, 1)
        h1 = (temp < P_h1).astype(int)  ##小于可以认为取1  size(h)*1
        # h1[P_h1>=self.bound] = 1
        # h1[P_h1<self.bound] = 0
        return P_h1, h1
    def backward(self, h1, w, a):
        P_v2 = self.sigmoid(a + np.dot(w.T, h1))  # 注意转置
        temp = np.random.rand(P_v2.size, 1)
        v2 = (temp < P_v2).astype(int)
        # v2[P_v2 >= self.bound] = 1
        # v2[P_v2 < self.bound] = 0
        return P_v2, v2
    def get_data(self):

        ###################  get img   ######################
        self.train_data = np.load('MNIST_28_28_train.npy')[0:self.batch_size]
        ################## get labels #######################
        self.train_labels = np.load('MNIST_train_labels.npy')[0:self.batch_size]
        ################## get test data  ###################
        self.test_data = np.load('MNIST_28_28_test.npy')[0:self.test_size]
        self.test_labels = np.load('MNIST_test_labels.npy')[0:self.test_size]

    def train(self):
        self.train_data[self.train_data >= 0.5] = 1
        self.train_data[self.train_data < 0.5] = 0
        rbm1 = RBM(self.size_h1,self.size_v1,self.n,self.batch_size,self.alpha1,self.train_data,False)
        rbm1.train(rbm1.CD_k)
        self.w1 = rbm1.w
        self.a1 = rbm1.a
        self.b1 = rbm1.b
        self.h1 = rbm1.h
        # print(self.h1)
        rbm2 = RBM(self.size_h2,self.size_v2,self.n,self.batch_size,self.alpha1,self.h1,False)
        rbm2.train(rbm2.CD_k)
        self.w2 = rbm2.w
        self.a2 = rbm2.a
        self.b2 = rbm2.b
        self.h2 = rbm2.h
        # print(self.h2)
        bp_net = bp_network(self.h2,self.train_labels,self.size_x,self.size_y,self.batch_size,self.alpha2,self.lam)
        bp_net.train()
        self.w_bp = bp_net.w
        # print(self.w_bp)
        bp_net = bp_implement(self.size_v1, self.size_h1, self.size_h2, self.size_softmax_layer, self.train_data,
                              self.train_labels,self.alpha2,int(self.n),self.test_data,self.test_labels,self.batch_size,self.test_size)
        bp_net.init_weight(self.w1,self.w2,self.w_bp)
        bp_net.train()
        bp_net.predict('dbn')

    def predict(self):
        precision = 0
        self.test_data[self.test_data>=0.5] = 1
        self.test_data[self.test_data<0.5] = 0
        for i in range(self.test_size):
            #########  rbm1  ###########
            P_h1, h1 = self.forward(self.test_data[i], self.w1, self.b1)
            P_v2, v2 = self.backward(h1, self.w1, self.a1)
            P_h2, h2 = self.forward(v2, self.w1, self.b1)
            #########  rbm2  ##########
            P_h3, h3 = self.forward(h2, self.w2, self.b2)
            P_v4, v4 = self.backward(h3, self.w2, self.a2)
            P_h4, h4 = self.forward(v4, self.w2, self.b2)
            self.h1_pre[i] = h2
            self.h2_pre[i] = h4
            z_out = self.softmax(np.dot(self.w_bp,h4))  #size  y*1
            self.z_out[i] = z_out
            self.labels_pre[i] = np.argmax(z_out)
            # print(self.labels_pre[i])
            if(np.argmax(self.test_labels[i])==np.argmax(self.z_out[i])):
                precision = precision + 1
        precision = precision/self.test_size
        print('dbn',precision)


##############     RBM    ##############
class RBM :
    def __init__(self,size_h,size_v,n,batch_size,alpha,input_data,enable_sparsity):
        self.size_h = size_h
        self.size_v = size_v
        self.n = n
        self.batch_size = batch_size
        self.alpha = alpha
        self.train_data = input_data
        self.enable_sparsity = enable_sparsity
        self.reset()
    def reset(self):
        self.w = 1 / np.sqrt(self.size_v * self.size_h) * np.random.rand(self.size_h, self.size_v)  # 权重矩阵
        # self.w = 0.1 * np.random.rand(self.size_h, self.size_v)  # 权重矩阵
        self.a = np.zeros((self.size_v, 1))  # 显层偏置
        self.b = np.zeros((self.size_h, 1))  # 隐层偏置
        self.h = np.zeros((self.batch_size,self.size_h,1))
        self.delta_w = np.zeros((self.size_h, self.size_v))
        self.delta_a = np.zeros((self.size_v, 1))
        self.delta_b = np.zeros((self.size_h, 1))
        self.bias = np.zeros((self.size_h + self.size_v, 1))
        self.delta_bias = np.zeros((self.size_h + self.size_v, 1))
        # self.sparse_matrix = np.ones((self.size_h, self.size_v))
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def forward(self, v1, w, b):
        P_h1 = self.sigmoid(b + np.dot(w, v1))  # 直接带入矩阵运算   w是 size(h)*size(v) x0是列向量size(v)*1
        temp = np.random.rand(P_h1.size, 1)
        h1 = (temp < P_h1).astype(int)  ##小于可以认为取1  size(h)*1
        # h1[P_h1>=self.bound] = 1
        # h1[P_h1<self.bound] = 0
        return P_h1, h1
    def backward(self, h1, w, a):
        P_v2 = self.sigmoid(a + np.dot(w.T, h1))  # 注意转置
        temp = np.random.rand(P_v2.size, 1)
        v2 = (temp < P_v2).astype(int)
        # v2[P_v2 >= self.bound] = 1
        # v2[P_v2 < self.bound] = 0
        return P_v2, v2
    def train(self,Algorithm):
        for i in range(self.n):
            # self.penalty = (1 - 0.9 * i / self.n) * self.penalty
            for j in range(self.batch_size):
                Algorithm(self.train_data[j], j , self.alpha, self.enable_sparsity)
    def CD_k(self, x0, j, alpha,enable_sparsity):  # x0训练样本 n算法的迭代次数 alpha学习率  w权重矩阵 可见层偏置a 隐层偏置b  矩阵均采用(numpy)array
        # momentum = 0.3
        P_v1 = x0
        v1 = copy.deepcopy(P_v1)
        v1[v1 >= 0.5] = 1
        v1[v1 < 0.5] = 0
        v1_tmp = copy.deepcopy(v1)
        # print(P_v1)
        # 训练传递阶段 ， 隐藏参数k k次训练
        P_h1, h1 = self.forward(v1_tmp, self.w, self.b)
        P_v2, v2 = self.backward(h1, self.w, self.a)
        P_h2, h2 = self.forward(v2, self.w, self.b)
        # v1_tmp = v2  # 重置参数

        ##误差计算,更新阶段
        self.delta_w = alpha * (np.dot(h1, v1.T) - np.dot(h2, v2.T))
        # self.delta_a = alpha*(P_v1 - P_v2)
        # self.delta_b = alpha*(P_h1 - P_h2)
        self.delta_a = np.zeros((self.size_v, 1))
        self.delta_b = np.zeros((self.size_h, 1))
        # return w,a,b
        self.w = self.w + self.delta_w
        # self.a = self.a + self.delta_a
        # self.b = self.b + self.delta_b
        self.a = np.zeros((self.size_v, 1))  # 显层偏置
        self.b = np.zeros((self.size_h, 1))  # 隐层偏置
        self.h[j] = h2
        # sparsity
        # if enable_sparsity is True:
        #     sparse_tmp = self.sparsity(self.sparse_matrix)
        #     self.sparse_matrix = sparse_tmp
        #     self.w = self.w * sparse_tmp
        #     self.delta_w = self.delta_w * sparse_tmp
    def VPF(self, x0, j, learning_rate, enable_sparsity):  # bias为size(v)+size(h) , w为size(h)*size(v)
        # i is in range(size(h))  j is in range(size(v))
        # z is in range(szie(h)+size(v))
        # decay = 1. - 0.9 * n / self.n
        z = np.zeros((self.size_v + self.size_h, 1))
        bias = self.bias
        weight = self.w
        P_v1 = x0
        v1 = copy.deepcopy(P_v1)
        v1[v1 >= 0.5] = 1
        v1[v1 < 0.5] = 0
        P_h1, h1 = self.forward(v1, weight, bias[-self.size_h:])
        # print(v1.shape,h1.shape)
        y = np.concatenate((v1, h1), axis=0)  # y  01_state  y = (v,h)
        alpha = 1 / 2 - y
        # print(weight)
        z1 = np.dot(weight.T, y[-self.size_h:]) + bias[0:self.size_v]  # z1 represent v  --- 0-size(v)
        # print(weight[30,77])
        z2 = np.dot(weight, y[0:self.size_v]) + bias[-self.size_h:]  # z2 represent h  -- size(v)--size(v)+size(h)
        z = np.concatenate((z1, z2), axis=0)  # z -- z（v,h）
        # print(min(z),max(z))
        delta = np.exp(alpha * z)
        # print(delta)
        temp_basis = alpha * delta
        # print(temp_basis)
        self.delta_bias = -learning_rate * temp_basis
        delta_w = -learning_rate * (np.dot(temp_basis, y.T) + np.dot(y, temp_basis.T))
        # print(delta_w)
        # self.delta_w = delta_w[-self.size_h:,:self.size_v]*decay
        self.delta_w = delta_w[-self.size_h:, :self.size_v]
        self.w = self.w + self.delta_w
        # self.w[self.w>1] = 1
        # self.w[self.w<-1] = -1
        self.bias = self.bias + self.delta_bias
        # self.a = self.bias[0:self.size_v]
        # self.b = self.bias[-self.size_h:]
        self.a = np.zeros((self.size_v, 1))  # 显层偏置
        self.b = np.zeros((self.size_h, 1))  # 隐层偏置
        self.h[j] = h1
        # sparsity
        # if enable_sparsity is True:
        #     sparse_tmp = self.sparsity(self.sparse_matrix)
        #     self.sparse_matrix = sparse_tmp
        #     self.w = self.w * sparse_tmp
        #     self.delta_w = self.delta_w * sparse_tmp

############  bp_network  ################ (softmax regression)
################################################
#y_i = softmax(wij*xij.sum)
#yi = softmax(y0i)
#delta_w = -(y_-y)*x + lambda * w
#w = w - alpha * delta_w
################################################
class bp_network :
    def __init__(self,x0,y0,size_x,size_y,batch_size,alpha = 0.05,lam = 0.01):
        self.size_x = size_x
        self.size_y = size_y
        self.x0 = x0
        self.y0 = y0
        self.batch_size = batch_size
        self.alpha = alpha
        self.lam = lam
        self.w = 0.1*np.random.rand(self.size_y, self.size_x)
        self.delta_w = np.zeros((self.size_y, self.size_x))
        self.n = 10
        # self.y_ = np.zeros((self.n,self.y,1))
        # self.y = np.zeros((self.n, self.y, 1))
    def softmax(self,y):
        s_y = np.exp(y)/np.sum(np.exp(y))
        return s_y
    def train(self):
        y_ = np.zeros((self.batch_size, self.size_y, 1))
        y = np.zeros((self.batch_size, self.size_y, 1))
        for j in range(self.n):
         for i in range(self.batch_size) :
            y_[i] = self.softmax(np.dot(self.w,self.x0[i]))  ## y*x  *  x*1
            y[i] = self.y0[i]
            self.delta_w = np.dot(y_[i]-y[i],self.x0[i].T)
            # self.delta_w = np.dot(y_[i] - y[i], self.x0[i].T) + lam* weight
            self.w = self.w - self.alpha*self.delta_w
    ##############  only fully-connected layer #################
    def predict(self,test_data,test_labels,test_size):
        y_ = np.zeros((test_size, self.size_y, 1))
        presicion = 0
        for i in range(test_size):
            y_[i] = self.softmax(np.dot(self.w,test_data[i]))
            if(np.argmax(test_labels[i])==np.argmax(y_[i])):
                presicion = presicion +1
        presicion = presicion/test_size
        print('only-bp_net:',presicion)

class multiLayer_bp(nn.Module) :
    def __init__(self,size_v1,size_h1,size_h2,size_softmax):
        super(multiLayer_bp, self).__init__()
        self.layer1 = torch.nn.Linear(size_v1, size_h1)
        self.layer2 = torch.nn.Linear(size_h1, size_h2)
        self.layer3 = torch.nn.Linear(size_h2, size_softmax)
        self.net = torch.nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
        )

    def return_net(self):
        return self.net
    def init_weight(self,w1,w2,w_bp):
        param = getattr(self.layer1,'weight')
        param.data = w1
        param = getattr(self.layer2, 'weight')
        param.data = w2
        param = getattr(self.layer3, 'weight')
        param.data = w_bp
class bp_implement:
    def __init__(self,size_v1,size_h1,size_h2,size_softmax,x0,train_labels,alpha,epochs,test_data,test_labels,batch_size,test_size):
        self.x0 = x0
        self.train_labels = train_labels
        self.alpha = alpha
        self.epochs = epochs
        self.bp_net_frame = multiLayer_bp(size_v1,size_h1,size_h2,size_softmax)
        self.bp_net = self.bp_net_frame.return_net()
        self.optimizer = torch.optim.SGD(self.bp_net.parameters(), lr=dbn.alpha2)
        self.loss_func = torch.nn.CrossEntropyLoss()  ##交叉熵损失函数包含softmax
        self.test_data = test_data
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.test_size = test_size
    def init_weight(self,w1,w2,w_bp):
        w1 = torch.from_numpy(w1)
        w1 = w1.type(torch.float32)
        w2 = torch.from_numpy(w2)
        w2 = w2.type(torch.float32)
        w_bp = torch.from_numpy(w_bp)
        w_bp = w_bp.type(torch.float32)
        self.bp_net_frame.init_weight(w1,w2,w_bp)

    def train(self):
        for epoch in range(self.epochs) :
            eval_loss = 0
            for i in range(self.batch_size):
                x0 = torch.from_numpy(self.x0[i].reshape(1,-1))
                # x0 = torch.tensor(x0,dtype=torch.float32)
                x0 = x0.type(torch.float32)
                label = torch.from_numpy(self.train_labels[i].reshape(1,-1))
                # label = torch.tensor(label, dtype=torch.float32)
                label = label.type(torch.float32)
                label = torch.max(label,1)[1]
                out = self.bp_net(x0)
                loss = self.loss_func(out,label)
                eval_loss = eval_loss +loss.data.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # print(eval_loss)

    def predict(self,string1):
        precision = 0
        for i in range(self.test_size):
            test_data = torch.from_numpy(self.test_data[i].reshape(1,-1))
            # test_data = torch.tensor(test_data,dtype=torch.float32)
            test_data = test_data.type(torch.float32)
            test_label = torch.from_numpy(self.test_labels[i].reshape(1,-1))
            # test_label = torch.tensor(test_label, dtype=torch.float32)
            test_label = test_label.type(torch.int)
            out = self.bp_net(test_data)
            prediction = torch.max(out,1)[1].type(torch.int)
            pre_labels = prediction.data.numpy()
            y_label = torch.max(test_label,1)[1]
            y_label = y_label.data.numpy()
            if(pre_labels==y_label):
                precision = precision +1
        precision = precision/self.test_size
        print(string1,precision)

if __name__ == '__main__':
    dbn = DBN()
    dbn.get_data()
    dbn.train()
    # dbn.predict()
    # bp2 = bp_network(dbn.train_data,dbn.train_labels,dbn.size_v1,dbn.size_softmax_layer,dbn.batch_size)
    # bp2.train()
    # bp2.predict(dbn.test_data,dbn.test_labels,dbn.test_size)


    bp_net = bp_implement(dbn.size_v1,dbn.size_h1,dbn.size_h2,dbn.size_softmax_layer,dbn.train_data,
                          dbn.train_labels,dbn.alpha2,dbn.n,dbn.test_data,dbn.test_labels,dbn.batch_size,dbn.test_size)
    bp_net.train()
    bp_net.predict('bp')

