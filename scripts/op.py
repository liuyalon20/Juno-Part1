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
    '''
    judge = cos_in > total_reflection_cos
    point_ = point[judge]
    event_ = event[judge]
    direction_ = direction[judge]
    time_ = time[judge]
    return point_, event_, direction_, time_

def reflection(point, in_vector, cos_in):
    ''''''
    normal_vector = point / R_i
    out_vector = in_vector - 2 * cos_in * normal_vector
    return out_vector

def refraction(point, in_vector, cos_in, cos_out):
    ''''''
    normal_vector = point / R_i
    out_vector = n * in_vector + (cos_out - n*cos_in) * normal_vector
    return out_vector

def petruthfunction(ParticleTruth_array,PMT_loc):
    ''''''
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
    
    PMT_x1 = (R_o - r * np.cos(np.pi/6)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y1 = (R_o - r * np.cos(np.pi/6)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z1 = (R_o - r * np.cos(np.pi/6)) * np.cos(PMT_theta) 
    PMT_position1 = np.vstack((PMT_x1, PMT_y1, PMT_z1)).T
    
    PMT_x2 = (R_o - r * np.cos(np.pi/3)) * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y2 = (R_o - r * np.cos(np.pi/3)) * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z2 = (R_o - r * np.cos(np.pi/3)) * np.cos(PMT_theta) 
    PMT_position2 = np.vstack((PMT_x2, PMT_y2, PMT_z2)).T
    
    PMT_x3 = R_o * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y3 = R_o * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z3 = R_o * np.cos(PMT_theta) 
    PMT_position3 = np.vstack((PMT_x3, PMT_y3, PMT_z3)).T
    
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
        third_progress1 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/6), starttime_5,velocity_2)
        endpoint1 = third_progress1[0]
        endtime1 = third_progress1[1]
        kdtree1 = KDTree(PMT_position1)
        judge1 = kdtree1.query(endpoint1)
        endpoint1 = endpoint1[judge1[0] <= r * np.sin(np.pi/6)]
        endtime1 = endtime1[judge1[0] <= r * np.sin(np.pi/6)]
        event_id_5_1 = event_id_5[judge1[0] <= r * np.sin(np.pi/6)]
        ChannelID1 = kdtree1.query(endpoint1)[1]

        startpoint_5 = startpoint_5[judge1[0] > r * np.sin(np.pi/6)]
        v0_5 = v0_5[judge1[0] > r * np.sin(np.pi/6)]
        starttime_5 = starttime_5[judge1[0] > r * np.sin(np.pi/6)]
        event_id_5 = event_id_5[judge1[0] > r * np.sin(np.pi/6)]
        third_progress2 = line(startpoint_5, v0_5, R_o - r * np.cos(np.pi/3), starttime_5,velocity_2)
        endpoint2 = third_progress2[0]
        endtime2 = third_progress2[1]
        kdtree2 = KDTree(PMT_position2)
        judge2 = kdtree2.query(endpoint2)
        endpoint2 = endpoint2[judge2[0] <= r * np.sin(np.pi/3)]
        endtime2 = endtime2[judge2[0] <= r * np.sin(np.pi/3)]
        event_id_5_2 = event_id_5[judge2[0] <= r * np.sin(np.pi/3)]
        ChannelID2 = kdtree2.query(endpoint2)[1]

        startpoint_5 = startpoint_5[judge2[0] > r * np.sin(np.pi/3)]
        v0_5 = v0_5[judge2[0] > r * np.sin(np.pi/3)]
        starttime_5 = starttime_5[judge2[0] > r * np.sin(np.pi/3)]
        event_id_5 = event_id_5[judge2[0] > r * np.sin(np.pi/3)]
        third_progress3 = line(startpoint_5, v0_5, R_o, starttime_5,velocity_2)
        endpoint3 = third_progress3[0]
        endtime3 = third_progress3[1]
        kdtree3 = KDTree(PMT_position3)
        judge3 = kdtree3.query(endpoint3)
        endpoint3 = endpoint3[judge3[0] <= r]
        endtime3 = endtime3[judge3[0] <= r]
        event_id_5_3 = event_id_5[judge3[0] <= r]
        ChannelID3 = kdtree3.query(endpoint3)[1]

        ChannelID = np.hstack((ChannelID1, ChannelID2, ChannelID3))
        endtime = np.vstack((endtime1, endtime2, endtime3))
        event_id_5 = np.vstack((event_id_5_1, event_id_5_2, event_id_5_3))


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
 
    