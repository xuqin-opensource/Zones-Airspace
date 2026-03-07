from air_corridor.geometry.geom3d import Cylinder, newTorus
from air_corridor.tools.util import *

'''
corridors={'A':{'object':Torus,'connections':{'B':LineSegment,'D':LineSegment }},
					'B':{},
					'C':{}}
corridors=[{'object':Torus,'connections':{'B':LineSegment,'D':LineSegment }},
		   { },
		   { }]
* corridors
    - object: 
        -- Torus_1
    - connections: 
        -- LineSegment_1
        -- LineSegment_2
					'''


class Corridor:
    # all_corridors = []
    graph = None    # 类属性，存储所有走廊的连接关系图
    # consider_next_corridor = False
    num_corridor_in_state = 1   # 状态中考虑的走廊数量

    def __init__(self, name, connections, reduce_space):
        # Initialize corridor properties
        self.name = name    # 走廊名称
        self.connections = connections  # 与其他走廊的连接关系
        self.reduce_space = reduce_space    # 是否减少状态空间

    """
    @classmethod 装饰器定义。
    它绑定到类而非实例，允许在不创建类实例的情况下直接调用。
    类方法的第一个参数通常命名为 cls，表示类本身，可以通过它访问类的属性和方法。
    """
    @classmethod
    def convert2graph(cls, corridors):
        # 将走廊字典转换为图结构
        cls.graph = {}
        for name, one_corridor in corridors.items():
            cls.graph[name] = one_corridor.connections
        return cls.graph

    def evaluate_action(self, a_uav, alignment=1, crossed=False):
        '''
        评估飞行器动作的奖励
        1.检查无人机是否在走廊内
        2.如果穿越边界，更新无人机状态
        3.根据情况给予奖励或惩罚
        alignement:衡量无人机飞行方向与走廊期望方向的对齐程度，取值范围[-1,1]其中 1 (完全对齐) -1 (完全相反)
        corssed [False, True]
        '''
        reward = PENALTY_TIME   # 基本生存时间奖励

        # all specified corridors follow dual inheritance, is_inside is from geom
        # 所有指定的corridors都遵循双重继承，is_inside继承自geom
        # a=self.is_inside(a_uav.next_position)
        flag, status = self.is_inside(a_uav.next_position)  # 判断是否发生碰撞
        if flag:
            # reward += aligned * REWARD_POSITIVE_STEP
            pass
        # 未发生碰撞
        elif crossed:
            '''
            whether consider the next corridor, considering two corridors' reward as one episode
            reward only make sense during training, here is trained with considering two corridors.
            是否考虑下一条走廊，考虑两条走廊的奖励作为一个episode奖励在训练中才有意义，这里是考虑两条走廊的训练。
            '''
            path = a_uav.enroute['path']
            
            # 到达最终管道
            if path[-1] == self.name:
                a_uav.status = 'won'
                reward += REWARD_REACH
            # 成功通过一个管道
            else:
                reward += REWARD_INTERMEDIA
                path_index = path.index(self.name)
                a_uav.enroute['current'] = path[path_index + 1]
                if path_index + 2 < len(path):
                    a_uav.enroute['next'] = path[path_index + 2]
                else:
                    a_uav.enroute['next'] = None
            # reward += alignment * REACH_ALIGNMENT
        # 发生碰撞
        else:
            # breach boundary
            # a_uav.outside_counter += 1
            # reward += PENALTY_BREACH
            # if a_uav.outside_counter > BREACH_TOLERANCE:
            #     a_uav.status = 'breached'
            reward += PENALTY_BREACH
            a_uav.status = status
        return reward

# 圆柱形走廊，继承自Corridor和Cylinder几何体。
class CylinderCorridor(Corridor, Cylinder):
    """
    anchor_point: 锚点(起点)
    length: 长度
    width: 宽度(直径)
    orientation_rad/vec: 方向(弧度或向量)
    """
    def __init__(self,
                 anchor_point,
                 length,
                 width,
                 name,
                 connections,
                 reduce_space,
                 orientation_rad=None,
                 orientation_vec=None, ):
        Corridor.__init__(self, name, connections, reduce_space)
        self.radius = width / 2
        Cylinder.__init__(self,
                          anchor_point=anchor_point,
                          orientation_vec=orientation_vec,
                          orientation_rad=orientation_rad,
                          length=length,
                          radius=self.radius)
        self.shapeType = [1, 0]
        self.reduce_space = reduce_space

    def evaluate_action(self, a_uav):
        # 重写父类方法:评估动作，检查是否穿越末端圆面
        alignment = 0
        # 通过检查从position到next_position的线段，可以判断UAV是否在当前时间步内穿过了走廊的终点
        cross_flag = self.endCirclePlane.cross_circle_plane(line_start=a_uav.position,
                                                            line_end=a_uav.next_position)
        if cross_flag:
            alignment = align_measure(end=a_uav.next_position, start=a_uav.position, direction=self.orientation_vec)
        # 继承Corridor类中的evaluate_action函数，用于评估飞行器动作的奖励
        reward = super().evaluate_action(a_uav, alignment=alignment, crossed=cross_flag)

        return reward

    def render_self(self, ax):
        # 渲染圆柱形3D可视化
        def cylinder(r, h, theta_res=100, z_res=100):
            theta = np.linspace(0, 2 * np.pi, theta_res)
            z = np.linspace(0, h, z_res)
            theta, z = np.meshgrid(theta, z)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return x, y, z

        Xc, Yc, Zc = cylinder(self.radius, self.length)
        x_rot, y_rot, z_rot = [], [], []
        for a, b, c in zip(Xc, Yc, Zc):
            # x_p, y_p, z_p = np.dot(self.rotation_matrix, np.array([a, b, c]))
            # 旋转
            x_p, y_p, z_p = np.dot(self.rotation_matrix_to_base, np.array([a, b, c]))

            x_rot.append(x_p)
            y_rot.append(y_p)
            z_rot.append(z_p)
        ax.plot_surface(np.array(x_rot), np.array(y_rot), np.array(z_rot), edgecolor='royalblue', lw=0.1, rstride=20,
                        cstride=8, alpha=0.3)

    @lru_cache(maxsize=8)
    def report(self, base=None):
        """
        目的：为强化学习agent提供走廊的状态表示
        使用LRU缓存，最多缓存8个结果，提高性能
        返回：固定长度的状态向量（形状类型、尺寸等参数）
        """
        # 7+2+4=13,
        # last two 0 are padding, keeping the format the same as

        # 简化模式
        if self.reduce_space:
            # base: 参考走廊对象，用于相对状态计算，当base相同时，缓存会生效，避免重复计算
            common_part = super().report(base)
            # common_part: 从父类继承的基础状态信息(8元素)
            # 位置差值（3个元素）：base.anchor_point - self.anchor_point，表示当前走廊锚点相对于基准走廊的位置偏移
            # 方向向量（3个元素）：base.rotate_to_base(self.orientation_vec)，将当前走廊的方向向量转换到基准走廊的坐标系中
            # 角度信息（2个元素）
            # shapeType: 形状标识 [1, 0] 表示圆柱形
            # length: 圆柱长度
            # radius: 圆柱半径
            # padding: 填充0，保持固定长度
            corridor_status = common_part + self.shapeType + [self.length, self.radius] + [0] * 4

            # if self == base:
            #     # 8 elements
            #     # 4 for the rest
            #     # 4 for padding
            #     # 16 intotal
            #     common_part = super().report(base)
            #     corridor_status = common_part + self.shapeType + [self.length, self.radius] + [0] * 4
            #
            # else:
            #     corridor_status = self.shapeType + list(Z_UNIT) + [self.length, self.radius] + [0] * 6

        else:
            # shapeType: [1, 0] 圆柱形标识
            # orientation_vec: 方向向量(x,y,z)
            # length, radius: 几何尺寸
            # padding: 填充0
            corridor_status = (self.shapeType + list(self.orientation_vec) +
                               [self.length, self.radius] +
                               [0] * 6)
        if any(np.isnan(corridor_status)):
            print('nan in cylinder')
            input("Press Enter to continue...")
        return corridor_status

    def release_uav(self, plane_offset_assigned):
        """
        负责在走廊的起始位置生成无人机的初始坐标，通过平面偏移和方向偏移的组合来确定精确位置。
       
        plane_offset_assigned: 一个二元组 [offset_x, offset_y]，表示在走廊横截面上的偏移量
        self.x 和 self.y: 走廊横截面的局部坐标系基向量
        将输入的2D偏移量转换为3D空间向量
        允许无人机在走廊横截面的任意位置生成

        self.orientation_vec: 走廊的主方向单位向量
        self.length / 2: 从锚点到走廊中心的距离
        0.2: 从走廊起点向内的一个小偏移（可能是安全距离）
        最终效果：在走廊起点向内0.2单位的位置

        self.anchor_point: 走廊的基准锚点

        最终位置 = 基准点 + 横截面偏移 + 纵向偏移
        """
        plane_offset = self.x * plane_offset_assigned[0] + self.y * plane_offset_assigned[1]
        direction_offset = (0.2 - self.length / 2) * self.orientation_vec
        return self.anchor_point + plane_offset + direction_offset

# 部分环面(弯曲)走廊，继承自Corridor和newTorus。
class DirectionalPartialTorusCorridor(Corridor, newTorus):
    """
    major_radius: 主半径(环半径)
    minor_radius: 次半径(管半径)
    begin_rad/end_rad: 起始和结束弧度
    """
    def __init__(self,
                 anchor_point: np.ndarray,
                 major_radius: float,
                 minor_radius: float,
                 begin_rad: float,
                 end_rad: float,
                 orientation_rad=None,
                 orientation_vec=None,
                 name=None,
                 connections=None,
                 reduce_space=True):
        Corridor.__init__(self, name, connections, reduce_space)
        newTorus.__init__(self,
                          anchor_point=anchor_point,
                          orientation_vec=orientation_vec,
                          orientation_rad=orientation_rad,
                          major_radius=major_radius,
                          minor_radius=minor_radius,
                          begin_rad=begin_rad,
                          end_rad=end_rad)
        # assert -np.pi <= self.begin_rad <= np.pi, "Error, begin radian needs to be in [-pi,pi]"
        self.shapeType = [0, 1]

    def evaluate_action(self, a_uav):
        alignment = 0
        cross_flag = self.endCirclePlane.cross_circle_plane(line_start=a_uav.position,
                                                            line_end=a_uav.next_position)
        if cross_flag:
            # positive_diretion是环面法向量与锚点指向飞行器位置叉乘所得到的向量
            positive_direction = self.determine_positive_direction(a_uav.position)
            alignment = align_measure(end=a_uav.next_position, start=a_uav.position, direction=positive_direction)
        reward = super().evaluate_action(a_uav, alignment=alignment, crossed=cross_flag)
        return reward

    @lru_cache(maxsize=8)
    def report(self, base=None):
        if self.reduce_space:
            # 8 elements for common
            # 8 for the rest
            # 16 intotal
            common_part = super().report(base)
            # major_radius：圆环主半径
            # minor_radius：圆环管半径
            # np.pi / 2 - (self.end_rad - self.begin_rad)：相减结果：表示从90度中减去实际角度范围后的剩余空间
            # np.pi / 2：固定角度值
            # self.major_radius + self.minor_radius：外径
            # self.major_radius - self.minor_radius：内径
            corridor_status = common_part + self.shapeType + \
                              [self.major_radius, self.minor_radius, np.pi / 2 - (self.end_rad - self.begin_rad),
                               np.pi / 2, self.major_radius + self.minor_radius,
                               self.major_radius - self.minor_radius]
        else:
            corridor_status = self.shapeType + list(self.directionRad) + list(self.orientation_vec) + \
                              [self.major_radius, self.minor_radius, self.begin_rad, self.end_rad,
                               self.major_radius + self.minor_radius, self.major_radius - self.minor_radius]
        if any(np.isnan(corridor_status)):
            print('nan in torus')
            input("Press Enter to continue...")
        return corridor_status

    def release_uav(self, plane_offset_assigned):
        plane_offset = self.beginCirclePlane.x * plane_offset_assigned[0] + \
                       self.beginCirclePlane.y * plane_offset_assigned[1]
        direction_offset = 0.2 * self.beginCirclePlane.orientation_vec

        return self.beginCirclePlane.anchor_point + plane_offset + direction_offset

    def render_self(self, ax):
        def torus(R, r, R_res=100, r_res=100):
            u = np.linspace(0, 1.5 * np.pi, R_res)
            v = np.linspace(0, 2 * np.pi, r_res)
            u, v = np.meshgrid(u, v)
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            return x, y, z

        Xc, Yc, Zc = torus(self.major_radius, self.minor_radius)
        x_rot, y_rot, z_rot = [], [], []
        for a, b, c in zip(Xc, Yc, Zc):
            # x_p, y_p, z_p = np.dot(self.rotation_matrix, np.array([a, b, c]))
            x_p, y_p, z_p = np.dot(self.rotation_matrix_to_base, np.array([a, b, c]))
            x_rot.append(x_p)
            y_rot.append(y_p)
            z_rot.append(z_p)
        ax.plot_surface(np.array(x_rot), np.array(y_rot), np.array(z_rot), edgecolor='royalblue', lw=0.1, rstride=20,
                        cstride=8, alpha=0.3)
