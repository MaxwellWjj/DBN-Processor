import numpy as np
import copy
import struct
import matplotlib.pyplot as plt
import math

class QDBN:

    def __init__(self):
        self.size_h1 = 500  # 隐层1节点数
        self.size_h2 = 100  # 隐层2节点数
        self.size_v1 = 784  # 显层1节点数
        self.size_v2 = self.size_h1
        self.size_x = self.size_h2 # neuron network input
        self.size_softmax_layer = 10
        self.size_y = self.size_softmax_layer   ## neuron netword output (10 modes: 0--9)
        self.n = 10  # 循环重复训练次数
        self.alpha1 = 0.002  # RBM
        self.alpha2 = 0.05 #### bp_net ####
        self.lam = 0.01 ####bp_net ####
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
        #################  get img  ##################
        # with open(r'.\minist\MNIST\raw\train-images-idx3-ubyte','rb') as f:
        #     img_file = f.read()
        # train_data_bytes = img_file[16:len(img_file)]   ##60000*784(28*28)  47040000
        # fmt_img = '>'+str(784)+'B'
        # imgs_data = np.empty((self.batch_size,784))
        # offset = 0
        # for i in range(self.batch_size):
        #     imgs_data[i] = np.array(struct.unpack_from(fmt_img, train_data_bytes, offset))
        #     offset = offset +784
        # imgs_data = imgs_data/256
        # imgs_data = imgs_data.reshape(self.batch_size,784,1)
        # self.train_data = imgs_data
        # # self.test_data = self.train_data
        # # self.plot_data = self.test_data
        # #################  get label ##################
        # with open(r".\minist\MNIST\raw\train-labels-idx1-ubyte", "rb") as f:
        #     label_file = f.read()
        # train_data_bytes = label_file[8:len(label_file)]  ##60000*784(28*28)  47040000
        # fmt_img = '>' + str(1) + 'B'
        # labels = np.empty((self.batch_size, 1), dtype='uint8')
        # offset = 0
        # labels_array = np.zeros((self.batch_size, 10, 1), dtype='uint8')
        # for i in range(self.batch_size):
        #     labels[i] = np.array(struct.unpack_from(fmt_img, train_data_bytes, offset))
        #     offset = offset + 1
        # labels = labels.reshape(self.batch_size, 1)
        # for i in range(self.batch_size):
        #     labels_array[i, labels[i]] = 1
        # self.labels_array = labels_array
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
        print("RBM_1 training")
        rbm1 = RBM(self.size_h1,self.size_v1,self.n,self.batch_size,self.alpha1,self.train_data,False)
        rbm1.train(rbm1.CD_k)
        self.w1 = rbm1.w
        self.a1 = rbm1.a
        self.b1 = rbm1.b
        self.h1 = rbm1.h
        # print(self.h1)
        print("RBM_2 training")
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
        print('rbms&bp_net',precision)


##############     RBM    ##############
class QRBM :
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
            print("epoch ", i)
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
if __name__ == '__main__':
    dbn = QDBN()
    dbn.get_data()
    dbn.train()
    dbn.predict()
    # bp2 = bp_network(dbn.train_data,dbn.train_labels,dbn.size_v1,dbn.size_softmax_layer,dbn.batch_size)
    # bp2.train()
    # bp2.predict(dbn.test_data,dbn.test_labels,dbn.test_size)

