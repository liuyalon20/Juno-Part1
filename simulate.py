import argparse

# 处理命令行
parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="n", type=int, help="Number of events")
parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
args = parser.parse_args()

import h5py as h5
from tqdm import tqdm
import time
from scripts.op import petruthfunction
from scripts.event import eventpoint_generate
from scripts.event import  eventpoint_array

'''
本程序的作用是 生成模拟数据, 保存在hdf5文件中

参数:由上述的命令行工具传入
-n: Number of events
-g, --geo: Geometry file
-o, --output: Output file

输出格式:
文件名opt，格式为hdf5

ParticleTruth 表:
EventID 事件编号     '<i4'
x       顶点坐标x/mm '<f8'
y       顶点坐标y/mm '<f8'
z       顶点坐标z/mm '<f8'
p       顶点动量/MeV '<f8'

PETruth 表:
EventID   事件编号      '<i4'
ChannelID PMT 编号      '<i4'
PETime    PE击中时间/ns '<f8'
'''
print("正在运行simulate.py")
start_time = time.time()
# 读入几何文件
with h5.File(args.geo, "r") as geo:
    PMT_loc = geo['Geometry'][:17612]

ParticleTruth ,ParticleTruth_array = eventpoint_generate(4000)

PETruth = petruthfunction(ParticleTruth_array,PMT_loc)

# 输出
with h5.File(args.opt, "w") as opt:
    opt['ParticleTruth'] = ParticleTruth
    opt['PETruth'] = PETruth    

end_time = time.time()

print(f"simmulate 运行完毕，用时{end_time - start_time}")
