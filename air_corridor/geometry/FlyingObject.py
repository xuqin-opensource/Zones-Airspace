from pygame import gfxdraw

from air_corridor.tools._descriptor import Position, PositiveNumber
from air_corridor.tools.util import *

# from utils.memory import DebugTracking

'''
training and testing are very different
train: UAVs are only trained for a single corridor, 
        either DirectionalPartialAnnulusCorridor or RectangleCorridor
test: UAV only use well-trained model for testing in a multi-corridor environment, 
        with all positions reformed with relative positions to the centric of the current corridor.
So, accumulated reward in only calculated within one corridor, not across corridors.
'''


class FlyingObject: # 所有飞行物体的基类
    flying_list = []    # 存储所有飞行物体的类变量列表
    position = Position(3)  # 3D位置描述符，确保位置是3维数组

    '''
    corridors={'A':Corridor{'name','anchor'},
               'B':Corridor{'name','anchor'}}
    '''
    safe_distance = 1   # 安全距离
    events = ['won', 'collided', 'breached', 'half', 'breached1', 'breached2', 'collided1', 'collided2']    # 可能的事件类型
    GAMMA = 0.99
    capacity = 6
    beta_adaptor_coefficient = 1.0  # 动作适配系数
    num_corridor_in_state = 1   # 状态中包含的走廊数量

    # Flag indicating if the current environment is in the final corridor.
    # Essential for training in multi-corridor environments and
    # applicable to scenarios with a single-segment corridor in the state.
    corridor_index_awareness = False    # 走廊索引感知标志
    corridor_state_length = 16  # 走廊状态长度
    uniform_state = False   # 统一状态标志

    def __int__(self,
                name,   # 飞行对象名称
                position=np.array([0, 0, 0]),   # 当前位置
                position_delta=np.array([0, 0, 0]), # 位置变化量
                next_position=np.array([0, 0, 0]),  # 下一位置
                velocity=np.array([0, 0, 0]),   # 当前速度
                next_velocity=np.array([0, 0, 0]),  # 下一速度
                discrete=False, # 是否离散动作空间
                reduce_space=True,  
                ):
        self.discrete = discrete
        self.name = name
        self.terminated = False
        self.truncated = False

        self.globalPosition = None

        self.position = position
        self.position_delta = position_delta
        self.next_position = next_position

        self.velocity = velocity
        self.next_velocity = next_velocity
        self.position_delta = None

        self.status = 'Normal'
        self.reduce_space = reduce_space

    def apply_acceleration(self, acc, dt):
        # 调用apply_acceleration工具函数计算下一速度和位置变化
        self.next_velocity, self.position_delta, reward_illegal_acc = apply_acceleration(self.velocity,
                                                                                         self.velocity_max,
                                                                                         acc,
                                                                                         dt)
        self.next_position = self.position + self.position_delta    # 下一时刻位置
        if np.linalg.norm(self.next_position) > 500:    # 检查位置是否异常（超过500）
            input('abnormal')
            apply_acceleration(self.velocity, self.velocity_max, acc, dt)
        return reward_illegal_acc

    def get_distance_to(self, other_flying_object):
        distance = distance_point_point(self.position, other_flying_object.position)    # 计算到另一个飞行物体的欧式距离
        return distance

    def render_self(self):
        """ render itself """
        pass

    @classmethod
    def action_adapter(cls, action):
        '''
        将标准化动作转换为球坐标
        r, theta, phi  = action
        r     = [0, 1] -> [0,1]
        theta = [0, 1] -> [0, np.pi]
        phi   = [0, 1] -> [-np.pi, np.pi]*1.1, with beta base of 1, the selection concentrate on [2pi,0] is truncated.
        
        action[0] - 0.5：将[0,1]映射到[-0.5, 0.5]
        * 2 * np.pi：映射到[-π, π]的全方位角范围
        * cls.beta_adaptor_coefficient：调整方位角φ的敏感度，在训练中可以限制或扩展水平转向范围
        '''
        return [action[0], action[1] * np.pi, (action[2] - 0.5) * 2 * np.pi * cls.beta_adaptor_coefficient]
        # return [action[0], action[1] * np.pi, (action[2] - 0.5) * 2 * np.pi * cls.beta_adaptor_coefficient]
        # return [(action[0] - 0.5) * 2 * np.pi, action[1] * np.pi, action[2]]


class UAV(FlyingObject):
    '''unmanned aerial vehicle'''
    corridor_graph = None   # 走廊图
    corridors = None    # 走廊字典

    reduce_space = True

    def __init__(self,
                 init_corridor, # 起始走廊
                 des_corridor=None, # 目标走廊
                 discrete=False,
                 name=None,
                 # velocity_max=0.6,
                 # acceleration_max=0.6,
                 velocity_max=1.5,  # 最大速度
                 acceleration_max=0.3,  # 最大加速度
                 plane_offset_assigned=None,    # 平面偏移分配
                 reduce_space=True):

        # if self.corridor_graph is None:
        #     print("Error: Have not graph the corridors.")
        #     sys.exit()
        super().__int__(name, discrete=discrete, reduce_space=reduce_space)

        self.plane_offset_assigned = plane_offset_assigned

        if discrete:
            self.discrete_action_space = 8

        self.velocity_max = velocity_max
        self.acceleration_max = acceleration_max

        self.init_corridor = init_corridor
        self.des_corridor = des_corridor
        self.enroute = None

        self.instant_reward = 0


        self.neighbors = []

        self.steps = 0
        self.outside_counter = None
        self.flying_list.append(self)

        self.accumulated_reward = 0

        self.trajectory = []
        self.reward = 0

    def update_position(self):
        self.position = self.next_position
        self.velocity = self.next_velocity
        # print(self.position)

    def decompose_target(self):
        """
        验证起始和目标走廊在图中
        调用bfs_find_path查找路径
        设置路径和下一个走廊
        """
        assert (self.enroute['init'] in self.corridor_graph.keys() and
                self.enroute['des'] in self.corridor_graph.keys()), \
            "Error, the initial or the last corridor is not specified."
        path = bfs_find_path(self.corridor_graph, self.enroute['init'], self.enroute['des'])
        if path is None:
            self.enroute['path'] = None
            self.terminated = True
        else:
            self.enroute['path'] = path
            if len(path) > 1:
                self.enroute['next'] = path[1]

    #
    def take(self, action, dt):
        '''
        in take action on the base with reduced space, while output the "actual" values
        '''
        action = self.action_adapter(action)    # action是一个3维连续向量，每个分量的取值范围是[-1, 1]
        # r, theta, phi = action
        r = action[0]
        if self.reduce_space:
            # action is generated on based shape with direction of [0,0,1]
            heading_vector_on_base = polar_to_unit_normal(action[1:])   # 将球坐标方向转换为三维单位向量
            # 旋转到实际坐标系
            heading_vector = self.corridors[self.enroute['current']].rotate_to_remote(heading_vector_on_base)
            # heading_vector = np.dot(self.corridors[self.enroute['current']].rotation_matrix_to_remote,
            #                         heading_vector_on_base)
        else:
            # action is generated on shape with different direction
            heading_vector = polar_to_unit_normal(action[1:])

        # self.acceleration_max：最大允许加速度标量
        # r：加速度大小比例系数 ∈[-1, 1]
        # heading_vector：单位方向向量（三维）
        # 结果：三维加速度向量 = 最大加速度 × 比例系数 × 方向
        acc = self.acceleration_max * r * heading_vector
        reward_illegal_acc = self.apply_acceleration(acc, dt)
        self.steps += 1
        # here penalize with illegal actions in two parts,
        # 1) action range beyond pre-determined range
        # 2) action within range but enforce uav goes beyond velocity max
        # print(f"acc: {np.round(acc,3)},last vel: {np.round(self.velocity,3)}, "
        #       f"next vel:{np.round(self.next_velocity,3)}, position_delta:{np.round(self.position_delta,3)}")
        return 0  # reward_illegal_acc

    def reset(self):
        self.terminated = False
        self.truncated = False
        # self.enroute是一个字典，记录UAV从起始走廊到目标走廊的完整路径信息
        self.enroute = {'init': self.init_corridor,
                        'des': self.des_corridor,
                        'current': self.init_corridor,  # UAV当前所在的走廊
                        'next': None,
                        'path': None}   # 从起始到目标的完整走廊序列
        # 使用BFS算法在走廊图中查找从起始到目标的最短路径，并将结果存储在self.enroute['path']
        self.decompose_target()

        # 初始化飞行器在走廊中的位置
        self.position = UAV.corridors[self.enroute['current']].release_uav(self.plane_offset_assigned)  # self.position是以当前走廊为参考坐标系的位置（局部坐标系）
        self.next_position = None
        self.velocity = np.array([0, 0, 0])
        self.next_velocity = None
        self.outside_counter = 0
        self.status = 'Normal'

    def update_accumulated_reward(self):
        self.accumulated_reward = self.accumulated_reward * UAV.GAMMA + self.instant_reward

    def _report_self(self):
        # 4+3*4=16
        cur = self.corridors[self.enroute['current']]   # 获取当前走廊对象
        if self.reduce_space:
            base_position = cur.project_to_base(self.position)  # 将局部（走廊）坐标系中飞行器位置转换到基坐标系中
            # 1+1+1+1+3(base_position)+1+1+1=10
            first = [self.velocity_max, self.acceleration_max,
                     cur.distance_object_to_point(self.position),   # 当前位置到走廊中心的距离
                     np.linalg.norm(self.velocity)] + \
                    list(base_position) + \
                    list(cur.rotate_to_base(self.velocity)) # 当前速度大小 + 基坐标系中的位置 + 基坐标系中的速度
            # if torus
            if cur.shapeType == [0, 1]:
                second = list(cur.convert_2_polar(self.position, self.reduce_space)) + \
                         list(cur.convert_vec_2_polar(self.position, self.velocity, self.reduce_space)) # 将走廊坐标系中的位置转换为球坐标 + 速度向量在球坐标系下的变化率
            # if cylinder
            elif cur.shapeType == [1, 0]:
                second = [0] * 6    # 用零填充
            # indicate whether being in the last corridor
        else:
            first = [self.velocity_max, self.acceleration_max,
                     cur.distance_object_to_point(self.position),
                     np.linalg.norm(self.velocity)] + \
                    list(cur.point_relative_center_position(self.position)) + \
                    list(self.velocity)
            if cur.shapeType == [0, 1]:
                second = [0] * 6
            elif cur.shapeType == [1, 0]:
                second = [0] * 6
        if any(np.isnan(first + second)):
            print('nan in self')
            input("Press Enter to continue...")
        agent_status = first + second   # 10+6=16
        # if UAV.corridor_index_awareness:
        #     third = [1] if self.enroute['current'] == self.enroute['path'][-1] else [0]
        #     agent_status += third
        corridor_status = self._report_corridor()   # 获取走廊信息
        return agent_status + corridor_status   # 16+16=32，返回自身状态和走廊状态

    def _report_other(self):
        # 用于报告其他UAV状态的方法
        other_uavs_status = []
        for agent in self.flying_list:
            # 遍历所有飞行物体，并跳过自身和不在同一走廊的UAV
            if agent is self or agent.enroute['current'] != self.enroute['current']:
                continue

            cur = self.corridors[agent.enroute['current']]  # 获取其他UAV所在当前走廊对象
            if self.reduce_space:
                if UAV.uniform_state:
                    
                    first = [float(not agent.terminated), agent.velocity_max, agent.acceleration_max,
                             np.linalg.norm(agent.position - self.position)] + list(
                        cur.project_to_base(agent.position)) + list(
                        cur.rotate_to_base(agent.velocity)) + list(
                        cur.rotate_to_base(agent.position - self.position)) + list(
                        cur.rotate_to_base(agent.velocity - self.velocity))
                    if cur.shapeType == [0, 1]: # if torus
                        second = list(cur.convert_2_polar(agent.position, self.reduce_space)) + \
                                 list(cur.convert_vec_2_polar(agent.position, agent.velocity, self.reduce_space))
                    elif cur.shapeType == [1, 0]:
                        second = [0] * 6
                else:
                    # 4+3*4=16
                    # 终止状态(True=0.0, False=1.0)、最大速度、最大加速度、相对距离、基础坐标系中的位置（3）、速度（3）、相对位置（3）、相对速度（3）=16
                    first = [float(not agent.terminated), agent.velocity_max, agent.acceleration_max,
                             np.linalg.norm(agent.position - self.position)] + list(
                        cur.project_to_base(agent.position)) + list(
                        cur.rotate_to_base(agent.velocity)) + list(
                        cur.rotate_to_base(agent.position - self.position)) + list(
                        cur.rotate_to_base(agent.velocity - self.velocity))
                    if cur.shapeType == [0, 1]:
                        second = list(cur.convert_2_polar(agent.position, self.reduce_space)) + \
                                 list(cur.convert_vec_2_polar(agent.position, agent.velocity, self.reduce_space))
                    elif cur.shapeType == [1, 0]:
                        second = [0] * 6

            else:
                first = ([float(not agent.terminated), agent.velocity_max, agent.acceleration_max,
                          np.linalg.norm(agent.position - self.position)] +
                         list(agent.position) +
                         list(agent.velocity) +
                         list(agent.position - self.position) +
                         list(agent.velocity - self.velocity))
                if cur.shapeType == [0, 1]:
                    second = [0] * 6
                elif cur.shapeType == [1, 0]:
                    second = [0] * 6

            # if np.any(np.isnan(one_agent_status)):
            #     print('nan in neighbor')
            #     input("Press Enter to continue...")
            agent_status = first + second   # 16+6=22
            # if UAV.corridor_index_awareness:
            #     third = [1] if agent.enroute['current'] == agent.enroute['path'][-1] else [0]
            #     agent_status += third
            corridor_status = agent._report_corridor()  # 获取走廊信息
            other_uavs_status.append(agent_status + corridor_status)    # 维度为(other_agent数量，22+16=38)
        while len(other_uavs_status) < self.capacity - 1:
            # base_elements = 23 if UAV.corridor_index_awareness else 22
            # other_uavs_status.append([0] * (base_elements + 17 * self.num_corridor_in_state))
            # UAV.corridor_state_length=16, self.num_corridor_in_state=1
            other_uavs_status.append([0] * (22 + UAV.corridor_state_length * self.num_corridor_in_state))

        return other_uavs_status

    def _report_corridor(self):
        # 16 elements
        cur = self.corridors[self.enroute['current']]
        corridor_status = []
        # 确定当前走廊在完整路径中的位置
        # self.enroute['path']：完整的走廊路径列表，如 ['A', 'B', 'C', 'D'].index(值)：返回该值在列表中的索引位置
        # 示例：如果当前在'B'，路径是['A','B','C','D']，则cur_index = 1
        cur_index = self.enroute['path'].index(self.enroute['current'])
        res_path = self.enroute['path'][cur_index:] # 从当前走廊开始，获取剩余的路径
        for i, key_corridor in enumerate(res_path):
            if i + 1 > self.num_corridor_in_state:  # 遍历剩余路径，但限制数量只处理前self.num_corridor_in_state个走廊
                break
            # 16维状态向量（位置差3维 + 方向向量3维 + 球坐标角度2维 + 走廊类型标识2维 + 圆柱半径和长度2维 + 4位填充位）
            single_c_status = self.corridors[key_corridor].report(base=cur) # 将不同方向和位置的走廊统一到当前走廊的坐标系中，使得后续所有走廊状态都相对于当前走廊，具有旋转和平移不变性。

            # UAV.corridor_index_awareness = [1,1,0,1]
            # 含义：感知[起点, 当前走廊, 不感知下一个走廊, 感知终点]
            if UAV.corridor_index_awareness:
                corridor_index_state = [0, 0, 0, 0]
                if UAV.corridor_index_awareness[-1] and res_path[-1] == key_corridor:
                    corridor_index_state[3] = 1
                elif UAV.corridor_index_awareness[0] and (
                        self.enroute['path'][0] == key_corridor or sum(UAV.corridor_index_awareness) == 2):
                    corridor_index_state[0] = 1
                elif UAV.corridor_index_awareness[1] and (
                        self.enroute['path'][1] == key_corridor or sum(UAV.corridor_index_awareness) == 3):
                    corridor_index_state[1] = 1
                elif UAV.corridor_index_awareness[2] and (
                        self.enroute['path'][2] == key_corridor or sum(UAV.corridor_index_awareness) == 4):
                    corridor_index_state[2] = 1
                
                # 确保只有一个位置标志被激活（one-hot编码），如果激活了多个或没有激活，显示当前的感知配置
                assert sum(corridor_index_state) == 1, f"{UAV.corridor_index_awareness}"

            # indicating_being_the_last_corridor = [0] if res_path[-1] == key_corridor else [1]
            # if len(self.enroute['path'])==1:
            #     indicating_being_the_first_corridor=[0]
            # indicating_being_the_first_corridor = [1] if self.enroute['path'][0] == key_corridor else [0]
            # single_c_status += indicating_being_the_last_corridor
                single_c_status = single_c_status + corridor_index_state

            corridor_status += single_c_status  # 累积所有走廊状态
        corridor_status += [0] * (UAV.corridor_state_length * (self.num_corridor_in_state - len(res_path))) # 如果剩余路径中的走廊数少于num_corridor_in_state，用零填充
        return corridor_status

    def report(self):
        '''
        corridor_status: 16*n, single is 16
        self= 16+16*n
        other_uav: 22+16*n
        :param padding:
        :param reduce_space:
        :return:
        '''

        uav_status = self._report_self()
        other_uavs_status = self._report_other()
        # print(f" corridor, {len(corridor_status)}")
        # print(f" uav_status, {len(uav_status)}")
        return {'self': uav_status, 'other': other_uavs_status}

        # 8
        # corridor_status = self._report_corridor()
        # uav_status = self._report_self()
        # other_uavs_status = self._report_other()
        # # print(f" corridor, {len(corridor_status)}")
        # # print(f" uav_status, {len(uav_status)}")
        # return {'self': uav_status + corridor_status, 'other': other_uavs_status}

    def render_self(self, surf):
        """
        gfxdraw.filled_circle：Pygame的绘图函数，绘制填充圆
        参数1：surf - 目标画布

        参数2：int(OFFSET_x + self.position[0] * SCALE) - 圆心X坐标
        OFFSET_x：屏幕偏移量（可能是居中或边界偏移）
        self.position[0]：物体在世界坐标中的X位置
        SCALE：缩放因子（世界坐标→像素坐标）

        参数3：int(OFFSET_y + self.position[1] * SCALE) - 圆心Y坐标
        参数4：FLYOBJECT_SIZE - 1 - 圆的半径（比正常稍小）
        参数5：GREEN - 颜色常量（绿色）
        """
        if self.status == 'won':
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE - 1,
                GREEN,
            )
        elif self.terminated:
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE - 1,
                RED,
            )
        else:
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE,
                PURPLE,
            )

    class NCFO(FlyingObject):
        """
        non-cooperative flying objects（可能是障碍物或中立物体）
        """
        boundary = [None] * 3   # 初始化边界
        velocity = PositiveNumber() # 速度必须为正数

        def __init__(self, position, velocity):
            super().__init__(position)
            self.velocity = velocity
            self.flying_object_list.append(self)

        def setup_boundary(self, boundary):
            self.boundary = boundary / 2

        def is_boundary_breach(self, tentative_next_position):
            return True if any(tentative_next_position > self.boundary) or any(
                tentative_next_position < -self.boundary) else False

    class Baloon(NCFO):

        def __init__(self, position, speed, velocity):
            super().__init__(position, velocity)
            self.flying_object_list.append(self)

        def update_position(self):
            while True:
                tentative_next_position = self.position + self.direction * self.speed
                if not self.is_boundary_breach(tentative_next_position):
                    self.position = tentative_next_position
                    break
                v = np.random.randn(3)

    class Flight(NCFO):
        pass
