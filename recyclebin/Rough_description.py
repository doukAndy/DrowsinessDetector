#!/usr/bin/env python
# coding: utf-8

# ### EEG 数据的简要说明

# 大致上可以看成是多维度的时间序列，其中每个维度对应于从特定接收位点采集到的数据。不同的数据集采用的位点有不同，可参见具体数据集的说明。
# 比如下图上所标识的位置。

# In[3]:


import matplotlib.pyplot as plt
a = plt.imread('2d_sensors_Loc.png')
b = plt.imread('sensor_loc.png')
c = plt.imread('sensors_on_scalp.png')
plt.figure(figsize=(24,16))
plt.subplot(1,3,1)
plt.imshow(a)
plt.subplot(1,3,2)
plt.imshow(b)
plt.subplot(1,3,3)
plt.imshow(c)


# 通常情况下，实验室里对应于某个特定刺激会搜集长达一段时间的数据。比如让被测试者观看一段影片，阶段结束后被测试者反馈自身的情绪状态等。
# 由于这段时间会比较长，大部分的论文在做分类实验设计时会把原数据分割成小段，比如持续时间为1s、2s、5s等等. 这些小段可以有在时间上的
# 重合部分或者彼此之间不重合。下面以SEED数据集的一位受测者数据为例。(其他数据集会有不同的存储格式，但概念是一样的)

# In[4]:


from scipy.io import loadmat
import numpy as np
raw = loadmat('/mnt/HDD/Datasets/SEED/Preprocessed_EEG/1_20131027.mat')


# In[5]:


print(raw.keys()) #这个受测者进行了15个实验阶段 （15 sessions）


# In[6]:


session_1 = raw['djc_eeg1']
print(session_1.shape)  # 62 代表前面示意图中采用的通道（位点）数目，哪些通道被采用会在数据集中说明， 47001 为在时间上的采样点


# In[7]:


print(47001//200) # 这个数据集的采样频率是200hz， 所以上述数据大概长度为 235s


# In[8]:


def make_segs(data, seg_len, stride):
    '''
    epoching: 即将数据分割成合适长度的小段
    '''
    t_len = data.shape[1]
    segs = np.stack([data[:,i*stride:i*stride+seg_len] for i in range(t_len//stride) if i*stride+seg_len<=t_len], axis= 1)
    # print(segs.shape)
    return segs


# In[9]:


cc = make_segs(session_1, seg_len = 200, stride=200) #分割为长度为1s,互不重叠的小段
print(cc.shape)  # 62 个通道， 235 个小段样本， 每个样本含200个时间点， 所有这些小段会共享一个分类指标


# In[10]:


dd = cc.transpose((1,2,0)) # 可以排列为一般神经网络所接收的输入格式 （样本数在第一维）
print(dd.shape)


# 从大的方向说，一般有两种分类任务。一种是subject dependent，即训练集和测试集的数据都来自同一个人。
# 另一种是subject independent, 即训练集使用其他人的数据，目标对象的数据为测试集。对subject dependent 的实验来说，
# 针对训练、验证及测试的划分不同的论文通常有不同的方法。
# 
# 最简单的方法比如堆叠所有session分割后的小段（这时会有（N, 200, 62）大小的数据， N为堆叠后的总样本数目）再利用比如scikit-learn里的
# train_test_split等函数分割出训练、测试及验证集合。
# 
# 
# 还有的方法比如：
# * A) 采用不同的sessions。比如共15个sesssion，其中9个做训练，剩下6个作测试。 
# * B) 采用不同的时间段。比如 $0-t_1$ 分段后的结果作训练， $t_1-t_2$ 分段后的结果作验证， $>t_2$分段后的结果作测试。 $0 < t_1 < t_2$
# 
# 注意，不管选用那种方法准备训练集，要保证不同类下的样本在每次给模型提供的训练batch中所占的比例大致相同。

# 对SEED数据集，我这边已经有用简单堆叠办法得到并保存的数据，如有需要可以选用，也可以自己照着上面或相关论文的方案设计

# In[11]:


X = loadmat('/mnt/HDD/Datasets/SEED/S07_E01.mat')['segs'].transpose([2,1,0])# 该样本堆叠后为 (3394,200,62)
X.shape


# 下面是一个基于 B)的 data generator的例子，用于向模型提供batch data， 如有需要可以参考改动用于不同数据集合。

# In[18]:


def sample_generator(XData, YLabel, Y_transform=None, seg_len=128, window=(0,2000), batchsize=128):
    '''
    XData: data to slice from  (session, time, channel) or a list of arrays with shape (time, channel)
    YLabel: label score (not encoded yet)
    '''
    X = np.zeros((batchsize, seg_len, XData[0].shape[-1]))
    # Y = np.zeros(batchsize)
    
    label_pos = np.unique(YLabel)
    
    Type_idx = [np.where(YLabel == i)[0] for i in label_pos]

    while True:
        Y = np.zeros(batchsize)  #for the problem Y‘dimension is increasing every time after transformation per interation
        start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
        for i, idx in enumerate(start_idx):
            dice = np.random.randint(0, len(label_pos))  #randomly select a category
            trial_num = np.random.randint(0, len(Type_idx[dice])) # randomly select a session/trial in the category
            selected = Type_idx[dice][trial_num]     
#             print(selected)

            X[i] = XData[selected][idx:idx+seg_len, :]
            Y[i] = YLabel[selected]
        
        if Y_transform != None:
            YY = Y_transform(Y)
            yield X, YY
        else:
            yield X, Y


# In[13]:


temp = [] #这里只用了最后1min钟的数据（即200*60个点）为例，可以延长或使用全部数据。考虑到情绪是个时间累计的过程，如舍弃数据一般会舍弃前面的数据
for i in range(15):
    print(raw['djc_eeg{}'.format(i+1)].shape[1]) # 各session的长度是不一样的
#     temp.append(raw['djc_eeg{}'.format(i+1)][:, ])


# In[14]:


for i in range(15):
    temp.append(raw['djc_eeg{}'.format(i+1)][:, -200*60:].T)


# In[15]:


YLabel = np.array([1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]) # 这个数据集里 15 个session/trial 所对应的情绪类别。1：positive, 0: neutral, -1:negative


# In[20]:


train_generator = sample_generator(temp, YLabel, Y_transform=None, 
                                   seg_len=200, window=(0, 200*30), 
                                   batchsize=128) #在整理后的数据中使用前半分钟来生成训练集合


# In[21]:


Xtrain, Ytrain = train_generator.__next__()


# In[22]:


Xtrain.shape


# In[23]:


Ytrain.shape  #这里的Y未经过变换。 在generator里可以传入函数给变量transorm，用其实现onehot encoding


# 一个轻量化的模型是EEGNet, 如需参考，其基本架构大致如下(已keras下为例）：

# In[ ]:


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
           dropoutRate = 0.5, kernLength = 64, F1 = 8, use_STN = True,
           D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', **kwargs):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1       = layers.Input(shape = (Samples, Chans, 1))

    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)
    
    block1       = layers.BatchNormalization(axis = -1, name='BN-1')(block1)  # normalization on channels or time
    block1       = layers.DepthwiseConv2D((1, Chans), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv')(block1)
    block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    block1       = layers.Activation('elu')(block1)

    block1       = layers.AveragePooling2D((2, 1))(block1)
    block1       = dropoutType(dropoutRate)(block1) 
    
    block2       = layers.SeparableConv2D(F2, (5, 1),
                                   use_bias = False, padding = 'same',
                                    name = 'SepConv-1')(block1)
    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
       
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(nb_classes, name = 'last_dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    Mymodel      = Model(inputs=input1, outputs=softmax)
        
    return Mymodel

