import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from matplotlib import cm
from scripts.utils_para import R_i
from scripts.utils_para import vector_angle



# drawing prameters
 
# utils_para 内记录了内外半径

n1_bins = 20
pe_bins = 300

r_bins = np.linspace(0,1000 * R_i,400)
theta_bins = np.linspace(0,2 * np.pi,360)


# 该类在测试时会用到，请不要私自修改函数签名，后果自负
class Drawer:
    def __init__(self, data, geo):
        self.simtruth = data["ParticleTruth"]
        self.petruth = data["PETruth"]
        self.geo = geo["Geometry"]



    def draw_vertices_density(self, fig, ax):

        '''
        本函数画出给定顶点的分布概率密度随着r的关系
        在画图时使用了一个bins的修正，对于每一个区间、
        进行一个加权和修正
        '''
        x = np.array(self.simtruth['x'])
        y = np.array(self.simtruth['y'])
        z = np.array(self.simtruth['z'])
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2 )/ 1000
        n, bins, patches = ax.hist(r, bins=n1_bins, density = False )
        ax.cla()
        # 参考了https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html?highlight=ax%20hist#matplotlib.axes.Axes.hist
        # 上述返回的数组中 bins 是一个长度为46的数组，代表着每一个bins的边界位置
        ax.set_title("Draw vertices density by radius")
        ax.set_xlabel("ratio of r and LS radius R")
        ax.set_ylabel("Volume Density of Vertices ")
        ax.set_xlim(0, 1)
        ####


        ####
        ax.set_ylim(0,2)
        rho_0 =  len(self.simtruth)/(4 / 3 * np.pi * (R_i ** 3)) 
        #ax.set_ylim(0, 2)
        
        V = (4 / 3 )* np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        
        # 这里bin是一个25+1的数组，从第一个之后向后取和从第一个取到最后一个即可
        bin_center = (bins[:-1] + bins[1:]) / 2 / R_i
        ax.scatter(bin_center, n/(V*rho_0) , color='blue')



    def draw_pe_hit_time(self, fig, ax):
        '''
        本函数做出所有的pe时间的直方图，是对所有pmt上的打击时间进行一个
        直方图的分布作图，最大数据量约为10000 *4000 但是考虑到并非所有
        光子都能够命中pmt，因此真实的数据要小于这个数
        '''
        t = np.array(self.petruth['PETime'])
        max_t = np.max(t)
 
        #直方图的区间数目

        # plot
        ax.set_title('Histogram of Hit Time')
        ax.set_ylabel('Number of PE Hit')
        ax.set_xlabel('Hit Time /ns')
        ax.set_xlim(0, max_t)
        ax.hist(t,bins=pe_bins)




    def draw_probe(self, fig, ax):

        pmt_number = len(self.geo[:17612])
        sim_number = len(self.simtruth)
        pe_number = len(self.petruth)
        

        # 初始化对应的数组
        pmt_loc_carti = np.zeros((pmt_number,3))  #pmt位置
        vertices = np.zeros((sim_number,3))  
        pmt_loc = np.zeros((pe_number,3)) 
        vertices_pe_loc = np.zeros((pe_number,3)) 

        #构建一个pmt几何位置的数组,[:,1\2\3]分别表示x\y\z
        # 这一部分之和后边的位置点乘，不需要乘以utils之中的外球半径
        pmt_loc_carti[:,0] = np.sin(self.geo["theta"][:17612]*np.pi/180) *np.cos(self.geo["phi"][:17612] * np.pi/180)
        pmt_loc_carti[:,1] = np.sin(self.geo["theta"][:17612]*np.pi/180) *np.sin(self.geo["phi"][:17612] * np.pi/180)
        pmt_loc_carti[:,2] = np.cos(self.geo["theta"][:17612]*np.pi/180) 

        #生成PMT、光子、顶点对应的序列
        #填充长度和pethuth 相同的序列，每一个序列之中都是对应每一个event的pmt坐标

        pmt_loc [:,0] =pmt_loc_carti[:,0][self.petruth["ChannelID"]]
        pmt_loc [:,1] =pmt_loc_carti[:,1][self.petruth["ChannelID"]]
        pmt_loc [:,2] =pmt_loc_carti[:,2][self.petruth["ChannelID"]]

        #填充长度和pethuth 相同的序列序列对应petruth中每一个event对应的顶点坐标######
        #从长度为4000的数组之中使用np任意索引返回长度为4000*10000X1的数组，每个都记录了来自哪一个顶点
        
        vertices_pe_loc[:,0] = self.simtruth["x"][[self.petruth["EventID"]]]
        vertices_pe_loc[:,1] = self.simtruth["y"][[self.petruth["EventID"]]]
        vertices_pe_loc[:,2] = self.simtruth["z"][[self.petruth["EventID"]]]
        

        vertices[:,0] = self.simtruth["x"]
        vertices[:,1] = self.simtruth["y"]
        vertices[:,2] = self.simtruth["z"]


        #光子密度的数组计算
        #4000*10000
        s1 = np.tile(pmt_loc_carti,(sim_number,1))
        s2 = vertices.repeat(17612,axis=0)
        theta = vector_angle(pmt_loc,vertices_pe_loc)
        theta_0 = vector_angle(s1,s2)

        theta = np.hstack([theta,2*np.pi-theta])
        theta_0 = np.hstack([theta_0,2*np.pi-theta_0])

        r = np.sqrt(np.sum(vertices_pe_loc * vertices_pe_loc,axis=1))
        r_0 = np.sqrt(np.sum(s2 * s2,axis=1))

        r = np.hstack([r,r])
        r_00 = np.hstack([r_0,r_0])

        dis , bin_r, bin_theta = np.histogram2d(r,theta,bins=[r_bins,theta_bins])
        dis_0 , bin_r_0, bin_theta_0 = np.histogram2d(r_00,theta_0,bins=[r_bins,theta_bins])
        
        X,Y =np.meshgrid(bin_theta,bin_r)
        print('正在绘制热力图')
        pic = ax.pcolormesh(X,Y,dis/dis_0,
            shading='auto',
            norm=colors.LogNorm(vmin=1e-1, vmax=1e2),
            cmap=cm.get_cmap('jet'))
        ax.set_title("Heatmap of the Probe Function ")
        fig.colorbar(pic, label='Expected Number of PE per Vertex')





        
if __name__ == "__main__":
    import argparse

    # 处理命令行
    parser = argparse.ArgumentParser()
    parser.add_argument("ipt", type=str, help="Input simulation data")
    parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
    parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
    args = parser.parse_args()

    # 读入模拟数据
    data = h5.File(args.ipt, "r")
    geo = h5.File(args.geo, "r")
    drawer = Drawer(data, geo)

    # 画出分页的 PDF
    with PdfPages(args.opt) as pp:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_vertices_density(fig, ax)
        pp.savefig(figure=fig)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_pe_hit_time(fig, ax)
        pp.savefig(figure=fig)

        # Probe 函数图像使用极坐标绘制，注意 x 轴是 theta，y 轴是 r
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
        drawer.draw_probe(fig, ax)
        pp.savefig(figure=fig)
