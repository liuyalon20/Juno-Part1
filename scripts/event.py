

#  调用的各种包
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from .import utils_para 


def dis_f(x):
    return 1500*np.exp(-x/10) *(1- np.exp(-x/5))

def eventpoint_generate(number):
    '''
    输入一个number(4000)

    依照助教在readme中给出的方法,返回一个结构化数组,内部的标签包括
        EventID: 事件编号        '<i4'
        x:       顶点坐标x/mm    '<f8'
        y:       顶点坐标y/mm    '<f8'
        z:       顶点坐标z/mm    '<f8'
        p:       顶点动量/MeV    '<f8'

        为了方便计算，还会返回另外一个非结构的np数组，其具体的形式为 ：
        数组的长度为4000x4，分别为x,y,z,event id
    '''
    start_time = time.time()

    truth_arr_poi = np.zeros(number, 
      dtype=[("EventID", "<i4"), 
     ("x", "<f8"), ("y", "<f8"),
     ("z", "<f8"), ("p", "<f8")])
    event_id = np.array(range(number)) 
    truth_arr_poi["EventID"] = event_id
    
    rng = np.random.default_rng()
    r_distribution = rng.power(3,number)
    theta = rng.random((1,number))
    theta_distribution =(np.arcsin(2*theta-1)*2
                        +np.pi)/2
    phi_distribution = rng.random((1,number))

    x = ( utils_para.R_i * r_distribution 
         * np.sin(theta_distribution)
         * np.cos(np.pi * phi_distribution *2) ) *1000

    y = ( utils_para.R_i * r_distribution 
         * np.sin(theta_distribution)
         * np.sin(np.pi * phi_distribution *2)) *1000

    z = ( utils_para.R_i * r_distribution 
         * np.cos(theta_distribution)) *1000
    # 摘取自help文档：Draws samples in [0, 1] from a power distribution with positive exponent a - 1.power(a, size=None)
    p = np.ones(number)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x,y,z,',',alpha=0.1)
    plt.savefig('docs/distribution_1.png')

    

    vertical_array_tran= np.vstack((x,y,z,event_id))
    vertical_array=np.transpose(vertical_array_tran)

    truth_arr_poi["x"] = x
    truth_arr_poi["y"] = y
    truth_arr_poi["z"] = z
    truth_arr_poi["p"] = p

    
    end_time = time.time() 
    print(f'生成{number}个顶点位置共用时{end_time-start_time}s') 
    
    return truth_arr_poi,vertical_array



def photon_generate(number):
    '''
     函数生成的思路：
     按照给定的分布产生光子对于给定的分布,我们可以直接计算得出来其
     归一化系数为1500,我们做出图像(见doc中)发现大于60ms之后生成的
     光子数几乎可以忽略,因此我们模拟就截止到60ms内产生的光子.

     然后在这个区间里我们采用蒙特卡洛方法来生成光子，来保证最终生产的
     光子数目的期望为1000具体方法如下:
     
     选定一个给定的在(0,60)区间内的均匀随机分布(a first course in probability  sheldon.M,Ross)
     我们采用舍去法，先计算出来分布函数的最大值，记为lambda_max（1000/\sqrt(3)）,然后用蒙特卡
     洛方法产生齐次lambda_max的分布，之后再生成一个随机数s，如果s <lambda(t)/lambda_max,保留这一数据。

     接下来我们就使用这一方法来生成一个顶点按照给定分布的光子，返回一个结构化数组，其具体的结构如下
    ________________________________________________
            photonID: 光子编号          '<i8'
            t:        时间坐标t/ns      '<f8'
            v_x:      归一化的x方向速度  '<f8' 
            v_y:      归一化的y方向速度  '<f8' 
            v_z:      归一化的z方向速度  '<f8' 
    __________________________________________________
    '''
    # 现在确定了lambda_max，我们先确定下来lambda取最大值的可能光子数，作为蒙特卡洛的撒点数

    rng = np.random.default_rng()

    n = rng.poisson(utils_para.T *utils_para.lambda_max,1000)
    N = round(np.mean(n))
    # 这是1000次按照lambda_max * T模拟总时间次数的平均值，包括了1个顶点按照
    # 最大值齐次分布应当产生的光子
    ph_t_origin = rng.random((1, N)) * utils_para.T
    index = rng.random((1, N))
    a = dis_f(ph_t_origin)/utils_para.lambda_max
    ratio = np.where(a >index)[1]   #使用了numpy的where检索功能
    t_array = ph_t_origin[:,ratio]
    le_p = len(t_array[0])
    truth_arr_ph = np.zeros(le_p,
            dtype=[('photonID', '<i8'),
                   ('t','<f8'),
                   ('v_x','<f8'),
                   ('v_y','<f8'),
                   ('v_z','<f8')])


    theta = rng.random((1,le_p))
    theta_distribution =(np.arcsin(2*theta-1)*2
                        +np.pi)/2
    phi_distribution = rng.random((1,le_p)) *2*np.pi 

    vx = np.sin(theta_distribution)*np.cos(phi_distribution)
    vy = np.sin(theta_distribution)*np.sin(phi_distribution)
    vz = np.cos(theta_distribution)

    truth_arr_ph['photonID'] = np.array(range(le_p))
    truth_arr_ph['t'] = t_array[0]
    truth_arr_ph['v_x'] = vx
    truth_arr_ph['v_y'] = vy
    truth_arr_ph['v_z'] = vz


    return truth_arr_ph



def eventpoint_array(ParticleTruth):
    '''
    输入一个顶点事件的结构化数组

    返回一个数组，数组的长度为4000x4

    分别为x,y,z,event id

    '''
    start_time = time.time()
    event_id = np.array(range(number)) 
    
    rng = np.random.default_rng()
    r_distribution = rng.power(3,4000)
    theta = rng.random((1,4000))
    theta_distribution =(np.arcsin(2*theta-1)*2
                        +np.pi)/2
    phi_distribution = rng.random((1,4000))

    x = ( utils_para.R_i * r_distribution 
         * np.sin(theta_distribution)
         * np.cos(np.pi * phi_distribution *2) ) *1000

    y = ( utils_para.R_i * r_distribution 
         * np.sin(theta_distribution)
         * np.sin(np.pi * phi_distribution *2)) *1000

    z = ( utils_para.R_i * r_distribution 
         * np.cos(theta_distribution)) *1000
    # 摘取自help文档：Draws samples in [0, 1] from a power distribution with positive exponent a - 1.power(a, size=None)
    

    vertical_array_tran= np.vstack((x,y,z,event_id))
    vertical_array=np.transpose(vertical_array_tran)

    
    end_time = time.time() 
    print(f'生成4000个顶点的数组用时{end_time-start_time}s') 
    
    return vertical_array


def photon_array(number):
    '''
     函数生成的思路：
     按照给定的分布产生光子对于给定的分布,我们可以直接计算得出来其
     归一化系数为1500,我们做出图像(见doc中)发现大于60ms之后生成的
     光子数几乎可以忽略,因此我们模拟就截止到60ms内产生的光子.

     然后在这个区间里我们采用蒙特卡洛方法来生成光子，来保证最终生产的
     光子数目的期望为1000具体方法如下:
     
     选定一个给定的在(0,60)区间内的均匀随机分布(a first course in probability  sheldon.M,Ross)
     我们采用舍去法，先计算出来分布函数的最大值，记为lambda_max（1000/\sqrt(3)）,然后用蒙特卡
     洛方法产生齐次lambda_max的分布，之后再生成一个随机数s，如果s <lambda(t)/lambda_max,保留这一数据。

     最终结果是一个约10000x4的数组，4分别为t，vx,vy,yz
    '''
    # 现在确定了lambda_max，我们先确定下来lambda取最大值的可能光子数，作为蒙特卡洛的撒点数
 
    rng = np.random.default_rng()

    n = rng.poisson(utils_para.T *utils_para.lambda_max,1000)
    N = round(np.mean(n))
    # 这是1000次按照lambda_max * T模拟总时间次数的平均值，包括了1个顶点按照
    # 最大值齐次分布应当产生的光子
    ph_t_origin = rng.random((1, N)) * utils_para.T
    index = rng.random((1, N))
    a = dis_f(ph_t_origin)/utils_para.lambda_max
    ratio = np.where(a >index)[1]   #使用了numpy的where检索功能
    t_array = ph_t_origin[:,ratio]
    le_p = len(t_array[0])

    theta = rng.random((1,le_p))
    theta_distribution =(np.arcsin(2*theta-1)*2
                        +np.pi)/2
    phi_distribution = rng.random((1,le_p)) *2*np.pi 

    vx = np.sin(theta_distribution)*np.cos(phi_distribution)
    vy = np.sin(theta_distribution)*np.sin(phi_distribution)
    vz = np.cos(theta_distribution)


    pho_array_tran= np.vstack((t_array,vx,vy,vz))
    pho_array=np.transpose(pho_array_tran)


    return pho_array


