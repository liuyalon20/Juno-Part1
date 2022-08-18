# 光学过程
import time
import numpy as np
from tqdm import tqdm
from .event import photon_array
from scipy.spatial import KDTree
from .utils_para import R_i, R_o, n_w, n_l, c, r


velocity_1 = c / n_l / (10**9)
velocity_2 = c / n_w / (10**9)
n = n_l / n_w
total_reflection_cos = np.sqrt(1 - 1 / (n**2))


def product(x, y):
    '''
        x和y是两个N*3的二维数组，他们的每一行对应一个三维矢量。
    本函数用于计算x,y对应行的矢量点积之后的结果，返回一个N*1的
    二维数组，该数组每行为相应的点积结果。
    '''
    r = np.einsum('ij, ij->i', x, y)
    r = np.array([r]).T
    return r

def line(position, direction, R, time1, velocity):
    '''
        本函数用于模拟光子在球内的运动。参数position为光子出发点，direction为
    初速度方向，R为球半径，time1为光子运动起始时刻，velocity为光在介质中的传播
    速度。返回值point为光子与球面的交点，即将要处理的反射折射点，time2为光子运
    动到球面上的时刻，cos_in_angle是入射角的余弦值。在本函数模拟的过程中，认为
    光子做匀速直线运动。
        position，direction， point都是N*3的二维数组；R和velocity为常数；time1，
    time2，cos_in_angle都是N*1的二维数组。
    '''
    projection = product(position, direction)
    run = np.sqrt(abs(R**2 - (product(position,position) - projection**2))) - projection
    point = position + run * direction
    time2 = time1 + run / velocity
    cos_in_angle = product(direction, point) / R
    return point, time2, cos_in_angle

def total_reflection(cos_in, point, event, direction, time):
    '''
        本函数用于筛选出在界面处不发生全反射的光子，以便继续追踪它们的轨迹。cos_in
    为入射角的余弦，用于构造判准judge。point为光子击中液闪球面上的点，event为事件编
    号，direction为光子初始运动方向，time为光子起始运动时刻，point_,event_,direction_
    和time_为对应的筛选结果。
        point和direction是N*3的二维数组，cos_in、event和time是N*1的二维数组，point_
    和direction_是M*3的二维数组，event_和time_是M*1的二维数组。由于经过了筛选，因而有
    M<=N。
    '''
    judge = cos_in > total_reflection_cos
    point_ = point[judge]
    event_ = event[judge]
    direction_ = direction[judge]
    time_ = time[judge]
    return point_, event_, direction_, time_

def reflection(point, in_vector, cos_in):
    '''
        本函数用于处理在界面处发生反射的情况。in_vector和out_vector分别为入射方向和反射
    方向的单位矢量，cos_in为入射角余弦值，point为反射点。基本思路是算出法向量，用入射向量
    和法向量叠加出反射向量。
        point、in_vector和out_vector都是N*3的二维数组，cos_in是N*1的二维数组。
    '''
    normal_vector = point / R_i
    out_vector = in_vector - 2 * cos_in * normal_vector
    return out_vector

def refraction(point, in_vector, cos_in, cos_out):
    '''
        本函数用于处理在界面处发生折射的情况。in_vector和out_vector分别为入射方向和出射
    方向的单位矢量，cos_in和cos_out分别为入射角和折射角的余弦值，point为折射点。基本思路
    是算出法向量，并利用折射定律，由入射向量和法向量叠加出出射向量。
        point、in_vector和out_vector都是N*3的二维数组，cos_in和cos_out都是N*1的二维数组。
    '''
    normal_vector = point / R_i
    out_vector = n * in_vector + (cos_out - n*cos_in) * normal_vector
    return out_vector

def petruthfunction(ParticleTruth_array,PMT_loc):
    '''
        本函数是光学过程的主函数，用于生成PETruth。ParticleTruth_array是4000*4的二维数组，
    包含了顶点位置以及事件编号的信息。PMT_loc是从geo.h5中读取的PMT的信息。最终生成的PETruth
    是题目要求的结构化数组，将保存在data.h5中。
    '''
    start_time = time.time()
    # 先读入顶点数据
    eventpoint_position = np.delete(ParticleTruth_array, -1, axis = 1)
    eventpoint_id = np.delete(ParticleTruth_array, [0,1,2], axis = 1)
    ChannelID_end = [0]
    endtime_end = [[0]]
    event_id_end = [[0]]
    
    
    print("正在读入PMT几何信息")
    # 读入PMT的几何信息
    PMT_theta = PMT_loc['theta']*np.pi/180
    PMT_phi = PMT_loc['phi']*np.pi/180
    
    PMT_x1 = (R_o - r * np.cos(np.pi/18)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y1 = (R_o - r * np.cos(np.pi/18)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z1 = (R_o - r * np.cos(np.pi/18)) * np.cos(PMT_theta) 
    PMT_position1 = np.vstack((PMT_x1, PMT_y1, PMT_z1)).T
    
    PMT_x2 = (R_o - r * np.cos(np.pi/9)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y2 = (R_o - r * np.cos(np.pi/9)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z2 = (R_o - r * np.cos(np.pi/9)) * np.cos(PMT_theta) 
    PMT_position2 = np.vstack((PMT_x2, PMT_y2, PMT_z2)).T
    
    PMT_x3 = (R_o - r * np.cos(np.pi/6)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y3 = (R_o - r * np.cos(np.pi/6)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z3 = (R_o - r * np.cos(np.pi/6)) * np.cos(PMT_theta) 
    PMT_position3 = np.vstack((PMT_x3, PMT_y3, PMT_z3)).T

    PMT_x4 = (R_o - r * np.cos(np.pi/9 * 2)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y4 = (R_o - r * np.cos(np.pi/9 * 2)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z4 = (R_o - r * np.cos(np.pi/9 * 2)) * np.cos(PMT_theta) 
    PMT_position4 = np.vstack((PMT_x4, PMT_y4, PMT_z4)).T

    PMT_x5 = (R_o - r * np.cos(np.pi/18 * 5)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y5 = (R_o - r * np.cos(np.pi/18 * 5)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z5 = (R_o - r * np.cos(np.pi/18 * 5)) * np.cos(PMT_theta) 
    PMT_position5 = np.vstack((PMT_x5, PMT_y5, PMT_z5)).T

    PMT_x6 = (R_o - r * np.cos(np.pi/3)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y6 = (R_o - r * np.cos(np.pi/3)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z6 = (R_o - r * np.cos(np.pi/3)) * np.cos(PMT_theta) 
    PMT_position6 = np.vstack((PMT_x6, PMT_y6, PMT_z6)).T

    PMT_x7 = (R_o - r * np.cos(np.pi/18 * 7)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y7 = (R_o - r * np.cos(np.pi/18 * 7)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z7 = (R_o - r * np.cos(np.pi/18 * 7)) * np.cos(PMT_theta) 
    PMT_position7 = np.vstack((PMT_x7, PMT_y7, PMT_z7)).T

    PMT_x8 = (R_o - r * np.cos(np.pi/18 * 8)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y8 = (R_o - r * np.cos(np.pi/18 * 8)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z8 = (R_o - r * np.cos(np.pi/18 * 8)) * np.cos(PMT_theta) 
    PMT_position8 = np.vstack((PMT_x8, PMT_y8, PMT_z8)).T

    PMT_x9 = R_o * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y9 = R_o * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z9 = R_o * np.cos(PMT_theta) 
    PMT_position9 = np.vstack((PMT_x9, PMT_y9, PMT_z9)).T
    
    print('正在为每一个顶点按照非齐次泊松采样生成光子')
    for j in tqdm(range(10)):      # 分循环防止内存不足。
        print(f"正在进行第{j+1}次循环")
        startpoint = [[0, 0, 0]]
        v0 = [[0, 0, 0]]
        starttime = [[0]]
        event_id = [[0]]

        for i in tqdm(range(400)):
            photon = photon_array(10000)
            photon_v0 = np.delete(photon, 0, axis = 1)
            photon_time = np.delete(photon, [1,2,3], axis =1)
            photon_number = photon.shape[0]
            
            startpoint = np.vstack((startpoint, [eventpoint_position[i + 400*j]]*photon_number))
            event_id = np.vstack((event_id, [eventpoint_id[i + 400*j]]*photon_number))
            v0 = np.vstack((v0, photon_v0))
            starttime = np.vstack((starttime, photon_time))

        startpoint = np.delete(startpoint, 0, axis = 0)
        startpoint = startpoint / 1000
        event_id = np.delete(event_id, 0, axis = 0)
        v0 = np.delete(v0, 0, axis = 0)
        starttime = np.delete(starttime, 0, axis = 0)

        print("光子生成完毕")
        print("正在筛选第一次不发生全反射的光子")

        
        # 根据入射角的余弦值筛选出不发生全反射的光子。
        cos_in_angle_judge_1 = (line(startpoint, v0, R_i, starttime, velocity_1)[2].T)[0]
        total_reflection_1 = total_reflection(cos_in_angle_judge_1, startpoint, event_id, v0, starttime)
        # 编号1代表第一次不发生全反射的光子
        startpoint_1 = total_reflection_1[0]
        event_id_1 = total_reflection_1[1]
        v0_1 = total_reflection_1[2]
        starttime_1 = total_reflection_1[3]
        
        
        # 模拟第一次折射过程，计算透射系数，筛选出发生折射和反射的光子。
        print("正在筛选需要发生折射和反射的光子")
        first_progress = line(startpoint_1, v0_1, R_i, starttime_1, velocity_1)
        endpoint_1 =first_progress[0]
        endtime_1 = first_progress[1]
        cos_in_1 = first_progress[2]
        # 计算好角度备用
        sin_in_1 = np.sqrt(abs(1 - (cos_in_1)**2))
        sin_out_1 = n * sin_in_1
        cos_out_1 = np.sqrt(abs(1 - (sin_out_1)**2))
        sin_sum = sin_in_1 * cos_out_1 + cos_in_1 * sin_out_1
        cos_dif = cos_in_1 * cos_out_1 + sin_in_1 * sin_out_1
        # 计算透射系数
        tp = (2 * sin_out_1 * cos_in_1) / (sin_sum * cos_dif)
        ts = (2 * sin_out_1 * cos_in_1) / (sin_sum)
        t = (cos_out_1 * (tp**2 + ts**2)) / (2 * n * cos_in_1)
        t = (t.T)[0]
        # 进行筛选
        number = startpoint_1.shape[0]
        probability = np.random.rand(number)
        judge_refraction = probability < t
        judge_reflection = probability >= t
        # 编号2代表第一次发生反射的光子
        startpoint_2 = endpoint_1[judge_reflection]
        event_id_2 = event_id_1[judge_reflection]
        in_vector_2 = v0_1[judge_reflection]
        starttime_2 = endtime_1[judge_reflection]
        cos_in_2 = cos_in_1[judge_reflection]
        # 编号3代表第一次就发生折射的光子
        startpoint_3 = endpoint_1[judge_refraction]
        event_id_3 = event_id_1[judge_refraction]
        in_vector_3 = v0_1[judge_refraction]
        starttime_3 = endtime_1[judge_refraction]
        cos_in_3 = cos_in_1[judge_refraction]
        cos_out_3 = cos_out_1[judge_refraction]
        
        
        # 对于第一次发生反射的光子，模拟他们第二次达到边界的过程
        print("正在追踪第一次发生反射的光子在液闪内的后续运动过程")
        v0_2 = reflection(startpoint_2, in_vector_2, cos_in_2)
        # 继续筛选不发生全反射的光子。
        cos_in_angle_judge_2 = (line(startpoint_2, v0_2, R_i, starttime_2, velocity_1)[2].T)[0]
        total_reflection_2 = total_reflection(cos_in_angle_judge_2, startpoint_2, event_id_2, v0_2, starttime_2)
        # 编号4代表第二次不发生全反射的光子，认为它们全部出射
        startpoint_4 = total_reflection_2[0]
        event_id_4 = total_reflection_2[1]
        v0_4 = total_reflection_2[2]
        starttime_4 = total_reflection_2[3]
        second_progress = line(startpoint_4, v0_4, R_i, starttime_4, velocity_1)
        endpoint_4 = second_progress[0]
        endtime_4 = second_progress[1]
        cos_in_4 = second_progress[2]
        sin_in_4 = np.sqrt(abs(1 - (cos_in_4)**2))
        sin_out_4 = n * sin_in_4
        cos_out_4 = np.sqrt(abs(1 - (sin_out_4)**2))


        # 模拟所有出射光子的折射行为。编号5代表所有出射光子。
        print("正在模拟折射过程")
        startpoint_5 = np.vstack((startpoint_3, endpoint_4))
        starttime_5 = np.vstack((starttime_3, endtime_4))
        event_id_5 = np.vstack((event_id_3, event_id_4))
        in_vector_5 = np.vstack((in_vector_3, v0_4))
        cos_in_5 = np.vstack((cos_in_3, cos_in_4))
        cos_out_5 = np.vstack((cos_out_3, cos_out_4))
        v0_5 = refraction(startpoint_5, in_vector_5, cos_in_5, cos_out_5)
        print(f"经统计，共有{startpoint.shape[0] - startpoint_5.shape[0]}个光子发生了全反射")

        # 追踪出射光子，判断它们能否打到PMT上以及打在哪个PMT上，对于打中PMT的光子，还要记录它们打中的时刻。
        # 本部分使用若干次KDTree判断光子打中的PMT的编号。因此third_progress要做多次，为了和之前的变量作区分，
        # 分别用不加下划线的数来编号。
        print("正在计算光子的最终落点以及终点时刻，并判断光子打中的PMT编号")
        third_progress1 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/18), starttime_5,velocity_2)
        endpoint1 = third_progress1[0]
        endtime1 = third_progress1[1]
        kdtree1 = KDTree(PMT_position1)
        judge1 = kdtree1.query(endpoint1)
        endpoint1 = endpoint1[judge1[0] <= r * np.sin(np.pi/18)]
        endtime1 = endtime1[judge1[0] <= r * np.sin(np.pi/18)]
        event_id_5_1 = event_id_5[judge1[0] <= r * np.sin(np.pi/18)]
        ChannelID1 = kdtree1.query(endpoint1)[1]

        startpoint_5 = startpoint_5[judge1[0] > r * np.sin(np.pi/18)]
        v0_5 = v0_5[judge1[0] > r * np.sin(np.pi/18)]
        starttime_5 = starttime_5[judge1[0] > r * np.sin(np.pi/18)]
        event_id_5 = event_id_5[judge1[0] > r * np.sin(np.pi/18)]
        third_progress2 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/9), starttime_5,velocity_2)
        endpoint2 = third_progress2[0]
        endtime2 = third_progress2[1]
        kdtree2 = KDTree(PMT_position2)
        judge2 = kdtree2.query(endpoint2)
        endpoint2 = endpoint2[judge2[0] <= r * np.sin(np.pi/9)]
        endtime2 = endtime2[judge2[0] <= r * np.sin(np.pi/9)]
        event_id_5_2 = event_id_5[judge2[0] <= r * np.sin(np.pi/9)]
        ChannelID2 = kdtree2.query(endpoint2)[1]

        startpoint_5 = startpoint_5[judge2[0] > r * np.sin(np.pi/9)]
        v0_5 = v0_5[judge2[0] > r * np.sin(np.pi/9)]
        starttime_5 = starttime_5[judge2[0] > r * np.sin(np.pi/9)]
        event_id_5 = event_id_5[judge2[0] > r * np.sin(np.pi/9)]
        third_progress3 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/6), starttime_5,velocity_2)
        endpoint3 = third_progress3[0]
        endtime3 = third_progress3[1]
        kdtree3 = KDTree(PMT_position3)
        judge3 = kdtree3.query(endpoint3)
        endpoint3 = endpoint3[judge3[0] <= r * np.sin(np.pi/6)]
        endtime3 = endtime3[judge3[0] <= r * np.sin(np.pi/6)]
        event_id_5_3 = event_id_5[judge3[0] <= r * np.sin(np.pi/6)]
        ChannelID3 = kdtree3.query(endpoint3)[1]

        startpoint_5 = startpoint_5[judge3[0] > r * np.sin(np.pi/6)]
        v0_5 = v0_5[judge3[0] > r * np.sin(np.pi/6)]
        starttime_5 = starttime_5[judge3[0] > r * np.sin(np.pi/6)]
        event_id_5 = event_id_5[judge3[0] > r * np.sin(np.pi/6)]
        third_progress4 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/18 * 4), starttime_5,velocity_2)
        endpoint4 = third_progress4[0]
        endtime4 = third_progress4[1]
        kdtree4 = KDTree(PMT_position4)
        judge4 = kdtree4.query(endpoint4)
        endpoint4 = endpoint4[judge4[0] <= r * np.sin(np.pi/18 * 4)]
        endtime4 = endtime4[judge4[0] <= r * np.sin(np.pi/18 * 4)]
        event_id_5_4 = event_id_5[judge4[0] <= r * np.sin(np.pi/18 * 4)]
        ChannelID4 = kdtree4.query(endpoint4)[1]

        startpoint_5 = startpoint_5[judge4[0] > r * np.sin(np.pi/18 * 4)]
        v0_5 = v0_5[judge4[0] > r * np.sin(np.pi/18 * 4)]
        starttime_5 = starttime_5[judge4[0] > r * np.sin(np.pi/18 * 4)]
        event_id_5 = event_id_5[judge4[0] > r * np.sin(np.pi/18 * 4)]
        third_progress5 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/18 * 5), starttime_5,velocity_2)
        endpoint5 = third_progress5[0]
        endtime5 = third_progress5[1]
        kdtree5 = KDTree(PMT_position5)
        judge5 = kdtree5.query(endpoint5)
        endpoint5 = endpoint5[judge5[0] <= r * np.sin(np.pi/18 * 5)]
        endtime5 = endtime5[judge5[0] <= r * np.sin(np.pi/18 * 5)]
        event_id_5_5 = event_id_5[judge5[0] <= r * np.sin(np.pi/18 * 5)]
        ChannelID5 = kdtree5.query(endpoint5)[1]


        startpoint_5 = startpoint_5[judge5[0] > r * np.sin(np.pi/18 * 5)]
        v0_5 = v0_5[judge5[0] > r * np.sin(np.pi/18 * 5)]
        starttime_5 = starttime_5[judge5[0] > r * np.sin(np.pi/18 * 5)]
        event_id_5 = event_id_5[judge5[0] > r * np.sin(np.pi/18 * 5)]
        third_progress6 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/3), starttime_5,velocity_2)
        endpoint6 = third_progress6[0]
        endtime6 = third_progress6[1]
        kdtree6 = KDTree(PMT_position6)
        judge6 = kdtree6.query(endpoint6)
        endpoint6 = endpoint6[judge6[0] <= r * np.sin(np.pi/3)]
        endtime6 = endtime6[judge6[0] <= r * np.sin(np.pi/3)]
        event_id_5_6 = event_id_5[judge6[0] <= r * np.sin(np.pi/3)]
        ChannelID6 = kdtree6.query(endpoint6)[1]


        startpoint_5 = startpoint_5[judge6[0] > r * np.sin(np.pi/3)]
        v0_5 = v0_5[judge6[0] > r * np.sin(np.pi/3)]
        starttime_5 = starttime_5[judge6[0] > r * np.sin(np.pi/3)]
        event_id_5 = event_id_5[judge6[0] > r * np.sin(np.pi/3)]
        third_progress7 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/18 * 7), starttime_5,velocity_2)
        endpoint7 = third_progress7[0]
        endtime7 = third_progress7[1]
        kdtree7 = KDTree(PMT_position7)
        judge7 = kdtree7.query(endpoint7)
        endpoint7 = endpoint7[judge7[0] <= r * np.sin(np.pi/18 * 7)]
        endtime7 = endtime7[judge7[0] <= r * np.sin(np.pi/18 * 7)]
        event_id_5_7 = event_id_5[judge7[0] <= r * np.sin(np.pi/18 * 7)]
        ChannelID7 = kdtree7.query(endpoint7)[1]


        startpoint_5 = startpoint_5[judge7[0] > r * np.sin(np.pi/18 * 7)]
        v0_5 = v0_5[judge7[0] > r * np.sin(np.pi/18 * 7)]
        starttime_5 = starttime_5[judge7[0] > r * np.sin(np.pi/18 * 7)]
        event_id_5 = event_id_5[judge7[0] > r * np.sin(np.pi/18 * 7)]
        third_progress8 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/18 * 8), starttime_5,velocity_2)
        endpoint8 = third_progress8[0]
        endtime8 = third_progress8[1]
        kdtree8 = KDTree(PMT_position8)
        judge8 = kdtree8.query(endpoint8)
        endpoint8 = endpoint8[judge8[0] <= r * np.sin(np.pi/18 * 8)]
        endtime8 = endtime8[judge8[0] <= r * np.sin(np.pi/18 * 8)]
        event_id_5_8 = event_id_5[judge8[0] <= r * np.sin(np.pi/18 * 8)]
        ChannelID8 = kdtree8.query(endpoint8)[1]


        startpoint_5 = startpoint_5[judge8[0] > r * np.sin(np.pi/18 * 8)]
        v0_5 = v0_5[judge8[0] > r * np.sin(np.pi/18 * 8)]
        starttime_5 = starttime_5[judge8[0] > r * np.sin(np.pi/18 * 8)]
        event_id_5 = event_id_5[judge8[0] > r * np.sin(np.pi/18 * 8)]
        third_progress9 = line(startpoint_5, v0_5, R_o, starttime_5,velocity_2)
        endpoint9 = third_progress9[0]
        endtime9 = third_progress9[1]
        kdtree9 = KDTree(PMT_position9)
        judge9 = kdtree9.query(endpoint9)
        endpoint9 = endpoint9[judge9[0] <= r]
        endtime9 = endtime9[judge9[0] <= r]
        event_id_5_9 = event_id_5[judge9[0] <= r]
        ChannelID9 = kdtree9.query(endpoint9)[1]


        
        ChannelID = np.hstack((ChannelID1, ChannelID2, ChannelID3,ChannelID4, ChannelID5, ChannelID6,ChannelID7, ChannelID8, ChannelID9))
        endtime = np.vstack((endtime1, endtime2, endtime3,endtime4, endtime5, endtime6,endtime7, endtime8, endtime9))
        event_id_5 = np.vstack((event_id_5_1, event_id_5_2, event_id_5_3,event_id_5_4, event_id_5_5, event_id_5_6,event_id_5_7, event_id_5_8, event_id_5_9))


        # 汇总数据
        print(f"正在汇总第{j+1}次循环生成的数据")
        ChannelID_end = np.hstack((ChannelID_end, ChannelID))
        endtime_end = np.vstack((endtime_end, endtime))
        event_id_end = np.vstack((event_id_end, event_id_5))


    ChannelID_end = ChannelID_end[1:]
    endtime_end = np.delete(endtime_end, 0, axis = 0)
    event_id_end = np.delete(event_id_end, 0, axis = 0)
    
    print("正在生成PETruth")
    # 完成PETruth
    PETruth = np.empty(ChannelID_end.shape[0],
    dtype=[("EventID", "<i4"),("ChannelID", "<i4"),
           ("PETime", "<f8")])
    PETruth["EventID"] = event_id_end.T
    PETruth["ChannelID"] = ChannelID_end
    PETruth["PETime"] = endtime_end.T
    print(f"据统计，共有{event_id_end.shape[0]}个光子成功打到了PMT上")
    
    end_time = time.time()
    print(f'光学过程用时{end_time-start_time}s')
    
    return PETruth
 
    