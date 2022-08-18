import numpy as np
import scipy.constants
import h5py
'''
数据包包括两个部分，第一部分是一些用到的函数如求两个矢量的夹角、后续
也可以将需要的函数添加到该数据包之中，
第二部分：
本试验使用到的各种参数，包括内外半径，PMT的直径，以及折射率和光速
'''

def vector_angle (array_of_vector1,array_of_vector2):
    '''
    两个传入的np数组的形式均为(N,3)，列是序号，行是对应的xyz坐标

    返回值为一个长度为N的数组，每一个元素都是两个传入数组对应序号的两个矢量之间
    的夹角，取值范围为（0~pi）
    '''
    norm1 = np.sqrt(np.sum(array_of_vector1*array_of_vector1,axis=1))
    norm2 = np.sqrt(np.sum(array_of_vector2*array_of_vector2,axis=1))
    dot = np.sum(array_of_vector1*array_of_vector2,axis=1)
    costheta = dot/(norm1*norm2)
    theta = np.arccos(costheta)
    return theta



# 本试验所有用到的参数
R_i = 17.71 
R_o  = 19.5
n_w =1.33
n_l =1.48
n_g =1.50
c = scipy.constants.c
r_p = 0.5 * 0.508
lambda_max = 1000/np.sqrt(3)
T=60
r = 0.5