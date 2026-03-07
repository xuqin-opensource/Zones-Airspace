import functools
from functools import reduce

import gymnasium as gym
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from air_corridor.d3.corridor.corridor import CylinderCorridor, DirectionalPartialTorusCorridor
from air_corridor.d3.geometry.FlyingObject import UAV
from air_corridor.tools.util import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "rps_v2"}

    def __init__(self,
                 render_mode=None,
                 reduce_space=True):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.state = None
        self.env_moves = None
        self.corridors = None
        self.render_mode = render_mode
        self.isopen = True
        self.distance_map = None
        self.reduce_space = True
        self.liability = False
        self.collision_free = False
        self.dt = 1
        self.consider_boid = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Dict(
            {'self': spaces.Box(low=-100, high=100, shape=(16 + 10,), dtype=np.float32),
             'other': spaces.Box(low=-100, high=100, shape=(22, (self.num_agents - 1)), dtype=np.float32)})

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # hasattr(object, name) 接受两个参数：
        # object：要检查的对象。
        # name：属性名称（字符串形式）。
        # 如果对象具有指定的属性，返回 True；否则返回 False。
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )   # 创建surface窗口
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)) # 创建离屏Surface（不显示窗口）
        if not hasattr(self, 'clock') or self.clock is None:
            self.clock = pygame.time.Clock()    # 时钟初始化，用于控制帧率

        # self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        # self.surf.fill(WHITE)
        fig = plt.figure()
        # 创建3d图形的两种方式
        # 将figure变为3d
        self.surf = Axes3D(fig)

        # 遍历所有走廊和智能体，调用它们的render_self方法，self.surf为三维画布
        for _, one_corridor in self.corridors.items():
            one_corridor.render_self(self.surf)
        for agent in self.agents:
            agent.render_self(self.surf)

        # pygame.transform.flip(surface, flip_x, flip_y)函数用于对图像进行水平和垂直翻转
        # surface: 需要翻转的图像。flip_x: 布尔值，表示是否进行水平翻转。flip_y: 布尔值，表示是否进行垂直翻转。
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0)) # 将创建好的图像填充到窗口中
        if self.render_mode == "human":
            # 更新pygame显示
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()   # 更新整个屏幕的显示

        elif self.render_mode == "rgb_array":
            # 返回numpy数组（形状：H×W×3）
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        try:
            if self.screen is not None:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
        except:
            pass

    def update_distance_map(self):
        # 计算得到每个智能体的距离，并存储在二维数组中
        count = len(self.agents)
        self.distance_map = np.ones([count, count]) / 1e-5

        for i in range(count):
            if self.agents[i].terminated:
                continue
            for j in range(i + 1, count):
                if self.agents[j].terminated:
                    continue
                dis = self.agents[i].get_distance_to(self.agents[j])
                self.distance_map[i, j] = dis

    def collision_detection(self, collisiion_distance=0.4):
        index = np.where(self.distance_map < collisiion_distance)   # 获取距离小于阈值的数组索引：(array(行索引列表), array(列索引列表))
        # 设原始数据: (array([0,1,2]), array([2,3,0]))
        # [list(i) for i in index] 
        # 将numpy数组转换为列表列表
        # 结果是：[[0, 1, 2], [2, 3, 0]]

        # reduce((lambda x, y: x + y), ...)
        # 这等价于：[[0,1,2], [2,3,0]] -> [0,1,2,2,3,0]
        # 将两个列表合并为一个

        # set(...)
        # 转换为集合去重：[0,1,2,2,3,0] -> {0,1,2,3}
        collide_set = set(reduce((lambda x, y: x + y), [list(i) for i in index]))
        for i in collide_set:
            if not self.agents[i].terminated:
                self.agents[i].status = 'collided'

    def access_neighbor_info(self):
        info = []
        for agent_i in self.agents:
            single_info = []
            for agent_j in self.agents:
                if agent_i is agent_j:
                    continue
                else:
                    # 包含相对位置向量和速度向量的完整信息
                    single_info.append(list(agent_j.position - agent_i.position) + list(agent_j.velocity))
            info.append(single_info)
        return info

    def random_combination(self, ratio, num):
        # ratio参数控制生成环形走廊（torus）和圆柱形走廊（cylinder）的比例
        seq = []
        for i in range(num):
            if random.random() < ratio:
                seq.append('t')
            else:
                seq.append('c')
        return tuple(seq)

    def generate_structure(self, difficulty=1, seq=None, minor_radius=2.0, test=False):
        '''
        :param connect_plane_anchor: in base,
        :param connect_plane_orientation: in base,
        :param rotation_matrix: base to remote,
        :param anchor_point: base to remote,
        :return:
        1e-3
        '''
        if seq is not None:
            num = len(seq)
        for i in range(num):
            non_last_flag = num > i + 1 # 确定是否是最后一个走廊
            name = chr(65 + i) # 返回asciII对应字母作为名称
            minor_radius = minor_radius
            if i == 0:
                intial_anchor = np.random.rand(3) * 2   # 随机起点坐标
                # initial_orientation_rad初始化theta和phi，根据三位球坐标使用polar_to_unit_normal函数转化为方向向量
                initial_orientation_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta,phi
                if seq[i] == 'c': # cylinder（圆柱形走廊，即水平通道）
                    cor = CylinderCorridor(anchor_point=intial_anchor, # 水平圆柱中心点
                                           orientation_rad=initial_orientation_rad, # 球坐标系theta,phi，决定圆柱方向
                                           length=random_(difficulty, epsilon=self.epsilon, segment=True) * 18 + 2, # 圆柱长度
                                           width=minor_radius * 2, # 圆柱直径
                                           name=name,
                                           connections=[], reduce_space=self.reduce_space) # reduce: Share feature extraction layers?
                    if non_last_flag:
                        connect_plane_orientation = cor.orientation_vec # 空域走廊方向即为下一空域走廊的链接方向
                        # cor.endCirclePlane.anchor_point = cor.anchor_point + cor.orientation_vec * cor.length / 2
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                else:   
                    begin_rad = np.pi * (2 * random.random() - 1)
                    if test:
                        end_rad = begin_rad + np.pi / 2
                    else:
                        end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)
                    major_radius = 5 * (random.random() + 1)
                    # Partial Torus 部分环形坡道走廊
                    cor = DirectionalPartialTorusCorridor(name=name,
                                                          anchor_point=intial_anchor,
                                                          orientation_rad=initial_orientation_rad,
                                                          major_radius=major_radius,
                                                          minor_radius=minor_radius,
                                                          begin_rad=begin_rad,
                                                          end_rad=end_rad,
                                                          connections=[],
                                                          reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.endCirclePlane.orientation_vec  # 终点圆截面法线方向为下一空域走廊的链接方向
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                if non_last_flag:   # 为下一个走廊准备连接信息
                    cor.connections = ['B']
                    rotate_to_end_plane = cor.endCirclePlane.rotate_to_remote   # 坐标变换矩阵，用于将点从当前坐标系转换到末端平面坐标系
                    # rotate_to_end_plane1 = cor.endCirclePlane.rotate_to_base
            else:
                if seq[i] == 'c':
                    length = random_(difficulty, epsilon=self.epsilon, segment=True) * 18 + 2
                    cor = CylinderCorridor(anchor_point=connect_plane_anchor + connect_plane_orientation * length / 2,
                                           orientation_vec=connect_plane_orientation,
                                           length=length,
                                           width=minor_radius * 2,
                                           name=name,
                                           connections=[], reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                else:
                    if seq[i - 1] == 't':
                        major_radius = self.corridors[chr(65 + i - 1)].major_radius
                    else:
                        major_radius = 5 * (random.random() + 1)
                    connect_plane_x = rotate_to_end_plane(X_UNIT)   # 将基向量[1,0,0]旋转到前一个走廊末端平面的局部坐标系，其中z轴为末端平面法向量
                    connect_plane_y = rotate_to_end_plane(Y_UNIT)   # 同上
                    random_rad = (random.random() * 2 - 1) * np.pi
                    unit_vec_connect_point_to_new_obj_anchor = (connect_plane_y * np.sin(random_rad) +
                                                                connect_plane_x * np.cos(random_rad))   # 从末端圆截面中心点指向环面坡道锚点
                    new_obj_anchor = connect_plane_anchor + unit_vec_connect_point_to_new_obj_anchor * major_radius # 环面坡道锚点

                    orientation_vec = np.cross(-unit_vec_connect_point_to_new_obj_anchor, connect_plane_orientation)    # 环面坡道法向量

                    # 计算旋转矩阵，将点从环形坡道局部坐标系旋转到世界坐标系
                    new_obj_to_base_matrix = vec2vec_rotation(orientation_vec, Z_UNIT)  # 将环形坡道的法向量（局部z轴）对齐到世界z轴
                    # 应用旋转矩阵，将从环面坡道锚点指向末端圆截面中心点的向量转换到世界坐标系
                    vec_on_base = np.dot(new_obj_to_base_matrix, -unit_vec_connect_point_to_new_obj_anchor)

                    begin_rad = np.arctan2(vec_on_base[1], vec_on_base[0])  # 计算向量在xy平面上的角度，即为起始弧度
                    if test:
                        end_rad = begin_rad + np.pi / 2
                    else:
                        end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)
                    
                    # 部分环面，允许无人机上升或下降，连接到不同层的圆柱形走廊，形成进出坡道。
                    cor = DirectionalPartialTorusCorridor(name=name,
                                                          anchor_point=new_obj_anchor,
                                                          orientation_vec=orientation_vec,
                                                          major_radius=major_radius,
                                                          minor_radius=minor_radius,
                                                          begin_rad=begin_rad,
                                                          end_rad=end_rad,
                                                          connections=[],
                                                          reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.endCirclePlane.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                if non_last_flag:
                    rotate_to_end_plane = cor.endCirclePlane.rotate_to_remote
                    # rotate_to_end_plane1 = cor.endCirclePlane.rotate_to_base
                    cor.connections = [chr(65 + i + 1)]

            self.corridors[name] = cor

    def reset(self,
              seed=None,
              options=None, # 训练的课程难度
              num_agents=3, # 无人机数量
              reduce_space=True,
              level=10,
              ratio=1,
              liability=True,
              collision_free=False, # 是否启用无碰撞模式
              beta_adaptor_coefficient=1.0,
              num_corridor_in_state=1,
              dt=1.0,
              consider_boid=False,  # 是否考虑群体行为
              corridor_index_awareness=False,
              velocity_max=1.5, # 最大速度
              acceleration_max=0.3, # 最大加速度
              uniform_state=False,
              minor_radius_test=2.0,
              dynamic_minor_radius=False,
              epsilon=0.1,
              test=False):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `env_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.epsilon = epsilon
        self.dt = dt
        self.consider_boid = consider_boid
        self.liability = liability
        self.collision_free = collision_free
        self.reduce_space = reduce_space
        # setup corridors
        # level参数控制走廊的结构复杂度：决定走廊的数量和类型组合
        # difficulty参数控制走廊的参数复杂度：影响走廊长度、角度跨度等具体数值
        # 效果->渐进训练：可以先训练简单结构+简单参数->简单结构+复杂参数，逐步增加复杂度
        difficulty = 1 if options is None else options['difficulty']

        self.corridors = {}
        # the following 4 parameters used for generating training env only

        if level == 0:  # 最简单的测试环境
            # # 固定部分环形坡道
            begin_rad = -np.pi
            end_rad = begin_rad + np.pi / 2
            major_radius = 10
            # orientation_rad = [0, 0]  #  theta,phi
        elif level == 1:    # 单个走廊（随机类型）
            begin_rad = np.pi * (2 * random.random() - 1)
            if difficulty <= 1:
                end_rad = begin_rad + np.pi / 2 * (difficulty + random.uniform(-0.1, 0.1))
            else:
                end_rad = begin_rad + np.pi / 2 * random.uniform(0.9, difficulty + 0.1)
            major_radius = 10
            orientation_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta,
            if random.random() > ratio:
                self.corridors['A'] = CylinderCorridor(anchor_point=np.array([0, 0, 0]),
                                                       orientation_rad=orientation_rad,
                                                       length=random.random() * difficulty * 15 + 5,
                                                       width=4,
                                                       name='A',
                                                       connections=[], reduce_space=self.reduce_space)
            else:
                self.corridors['A'] = DirectionalPartialTorusCorridor(name='A',
                                                                      anchor_point=np.array([0, 0, 0]),
                                                                      orientation_rad=orientation_rad,
                                                                      major_radius=major_radius,
                                                                      minor_radius=2,
                                                                      begin_rad=begin_rad,
                                                                      end_rad=end_rad,
                                                                      connections=[],
                                                                      reduce_space=self.reduce_space)
        elif level == 2:
            # fixed ending degree and fixed radius, but gradually increase fixed ending degree
            seq = self.random_combination(ratio, num=1)
            self.generate_structure(difficulty, seq, test=test)
        elif level == 3:
            seq = random.choice([('t', 't'), ('t', 'c'), ('c', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 10:
            seq = random.choice([('t'), ('c')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 11:
            seq = random.choice([('t', 't'), ('t', 'c'), ('c', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 12:
            seq = random.choice([('t', 't', 'c'), ('c', 't', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 13:
            if dynamic_minor_radius:
                minor_radius = np.random.uniform(1.8, 2.2)
            else:
                minor_radius = 2
            seq = random.choices([('t', 't', 'c'), ('c', 't', 't'), ('t', 'c', 't'), ('c', 't', 'c')],
                                 weights=[1.0, 1.0, 0.8, 0.8])[0]   # 带权重的随机选择
            self.generate_structure(difficulty, seq=seq, test=test, minor_radius=minor_radius)
        elif level == 14:
            seq = ('c', 't', 't', 'c')
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 15:
            seq = random.choice([('c', 't', 't', 'c'), ('t', 'c', 't', 'c'), ('c', 't', 'c', 't'), ('t', 't', 'c', 't'),
                                 ('t', 'c', 't', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        if not test and corridor_index_awareness:
            assert len(seq) >= sum(corridor_index_awareness)
        corridor_graph = self.corridors['A'].convert2graph(self.corridors)  # 将走廊连接关系（字典类型）转换为图结构
        # 状态中考虑的走廊数量
        DirectionalPartialTorusCorridor.num_corridor_in_state = num_corridor_in_state
        CylinderCorridor.num_corridor_in_state = num_corridor_in_state

        # setup uavs
        # 初始位置分布，在半径为2的圆内均匀分布无人机，确保最小距离为1。
        plane_offsets = distribute_evenly_within_circle(radius=2, min_distance=1, num_points=num_agents)
        UAV.flying_list = []
        if len(self.corridors) == 1:
            # # 单个走廊：起点和终点都是'A'
            self.agents = [UAV(init_corridor='A',
                               des_corridor='A',
                               name=None,
                               plane_offset_assigned=plane_offset,
                               velocity_max=velocity_max,
                               acceleration_max=acceleration_max) for plane_offset in plane_offsets]
        else:
            # # 多个走廊：从'A'出发，到最后一个走廊
            self.agents = [UAV(init_corridor='A',
                               des_corridor=chr(64 + len(self.corridors)),
                               name=None,
                               plane_offset_assigned=plane_offset,
                               velocity_max=velocity_max,
                               acceleration_max=acceleration_max) for plane_offset in plane_offsets]
        UAV.corridors = self.corridors  # 所有无人机共享走廊信息
        UAV.reduce_space = reduce_space
        UAV.corridor_graph = corridor_graph
        UAV.beta_adaptor_coefficient = beta_adaptor_coefficient
        UAV.num_corridor_in_state = num_corridor_in_state
        UAV.capacity = max(num_agents, 2)
        UAV.corridor_index_awareness = corridor_index_awareness
        # index capability with 4 bits
        # index up to 2: [1,0,0,1]; up to 3: [1,1,0,1]; up to 4: [1,1,1,1].
        UAV.corridor_state_length = 20 if corridor_index_awareness else 16
        UAV.uniform_state = uniform_state

        [agent.reset() for agent in self.agents]
        self.env_moves = 0
        observations = {agent: agent.report() for agent in self.agents}
        self.state = observations
        if self.render_mode == "human":
            self.render()
        infos = {'corridor_seq': seq}   # 走廊类型序列
        return observations, infos

    def step(self, action_dic):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # 首先执行每个智能体的动作并获得基础奖励
        rewards = {agent: agent.take(action, self.dt) for agent, action in action_dic.items()}

        # collision detection
        if not self.collision_free:
            # 如果启用碰撞检测（非collision_free模式），系统会更新距离映射并检测碰撞
            self.update_distance_map()
            self.collision_detection()

        disaster = False
        for agent, _ in rewards.items():
            if not agent.terminated:    # 对每个未终止的智能体进行奖励评估
                if agent.status == 'collided':  # 碰撞智能体获得PENALTY_COLLISION惩罚
                    reward_from_corridor = PENALTY_COLLISION
                else:   # 正常智能体通过当前走廊的evaluate_action()方法获得基于位置和行为的奖励
                    reward_from_corridor = agent.corridors[agent.enroute['current']].evaluate_action(agent)
                # 更新智能体的即时奖励和累积奖励
                rewards[agent] += reward_from_corridor
                agent.instant_reward = rewards[agent]

        for agent in self.agents:
            if not agent.terminated:
                # if agent.status in UAV.events:
                if agent.status != 'Normal':
                    disaster = True
                    break

        for agent in self.agents:
            if not agent.terminated:
                # 如果启用责任模式（liability=True）且发生灾难事件，所有非正常状态的智能体会受到额外惩罚 
                if self.liability and disaster and agent.status != 'Normal':
                    rewards[agent] = rewards[agent] + LIABILITY_PENALITY
                agent.update_position() # 更新位置和速度为下一时刻的值
                agent.update_accumulated_reward()

        self.env_moves += 1
        env_truncation = self.env_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # terminations = {agent: agent.status in UAV.events for agent in self.agents}
        terminations = {}
        for agent in self.agents:
            # agent.terminated = agent.status in UAV.events
            agent.terminated = agent.status != 'Normal'
            terminations[agent] = agent.terminated

        observations = {agent: agent.report() for agent in self.agents}

        self.state = observations

        if self.render_mode == "human":
            self.render()
        infos = {agent: None for agent in self.agents}

        return observations, rewards, terminations, truncations, infos
