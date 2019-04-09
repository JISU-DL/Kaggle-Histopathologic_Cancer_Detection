import numpy as np
import tensorflow as tf
from keras.layers import ZeroPadding2D, Conv2D,BatchNormalization,MaxPool2D, Dense, ReLU,ReLU, Add,\
GlobalAveragePooling2D,Softmax,DepthwiseConv2D,Reshape,Lambda,Concatenate,Input
from keras import backend as K

def InputBlock(input_tensor):
    conv1_padding = ZeroPadding2D((3,3))(input_tensor)
    conv1_conv = Conv2D(64,7,strides=(2,2))(conv1_padding)
    conv1_bn = BatchNormalization()(conv1_conv) 
    conv1_relu = ReLU()(conv1_bn) 
    return conv1_relu

def PoolingLayer(x):
    pool1_padding = ZeroPadding2D()(x)
    pool1_pool = MaxPool2D((3,3),2)(pool1_padding)    
    return pool1_pool

def DepthwiseConv(x,stride=1, cardinality= 32):
    x_filter = int(x.shape[-1])

    C = x_filter // cardinality

    x = DepthwiseConv2D(1,strides=stride, depth_multiplier=C)(x)
    x_shape = K.int_shape(x)[1:-1]  
    x = Reshape(x_shape + (cardinality, C, C))(x)  
    x = Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(C)]))(x)

    x = Reshape((x_shape[0],x_shape[1],x_filter))(x) 

    return x

# x:  입력 레이어를 입력합니다..
# filters: ConvNet의 필터 갯수를 리스트로 입력합니다
#           example : 64,64,256 => [ 64,64,256 ]
# kernel_size: 커널 크기를 리스트 형태로 입력합니다 
#           example 1x1, 3x3, 1x1 => [ 1,3,1]
# block_count: 반복할 횟수를 지정합니다 1번 반복 후  Residual을 합니다.
# Grouped_conv_strid (defult: True) : Grouped ConvNet의 stride를 2x2 크기로 주어 입력 크기를 감소시킵니다 
def ResidualBlock(x, filters, kernel_size, block_count,group=32,Grouped_conv_strid = True):
    if(len(filters) != len(kernel_size)) :
        raise ValueError("filters count and kernel_size count must be the same(Filter:%d, Kernal: %d)"%(len(filters), len(kernel_size)))
    if (len(kernel_size)==0 or len(filters)==0):
        raise ValueError("Have not entered value")
    
    shortcut = x
    residual_conv= True #첫번째 residual에 ConvNet을 추가합니다
    for i in range(block_count):
        if residual_conv == True:            
            if Grouped_conv_strid:
                shortcut = Conv2D(filters[-1],kernel_size[-1],strides=2,padding='same')(shortcut)
            else:
                shortcut = Conv2D(filters[-1],kernel_size[-1])(shortcut)
            shortcut = BatchNormalization()(shortcut)
            residual_conv = False
               
        x = Conv2D(filters[0],kernel_size[0])(x)            
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        if Grouped_conv_strid:
            x = DepthwiseConv(x,stride=2)
            Grouped_conv_strid = False
        else:
            x = DepthwiseConv(x)
                
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        if len(filters)==3:
            x = Conv2D(filters[2],kernel_size[2])(x)
            x = BatchNormalization()(x)
        x = Add()([x,shortcut])
        x = ReLU()(x)
        shortcut = x
    return x


def ResNeXt50(inputs=None):
    conv1 = InputBlock(inputs)
    pooling = PoolingLayer(conv1)
    conv2 = ResidualBlock(pooling,filters=[128,128,256], kernel_size=[1,3,1], block_count=3,Grouped_conv_strid=False)
    conv3 = ResidualBlock(conv2,filters=[256,256,512], kernel_size=[1,3,1], block_count=4)
    conv4 = ResidualBlock(conv3,filters=[512,512,1024], kernel_size=[1,3,1], block_count=6)
    conv5 = ResidualBlock(conv4,filters=[1024,1024,2048], kernel_size=[1,3,1], block_count=3)
    return conv5

    