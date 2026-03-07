import functools
from functools import reduce

import gymnasium as gym
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from air_corridor.corridor.corridor import CylinderCorridor, DirectionalPartialTorusCorridor
from air_corridor.geometry.FlyingObject import UAV
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
    def reset(self,
              num_agents=3, # 无人机数量
              reduce_space=True,
              num_annulus=3,
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
              radius=2.0,
              epsilon=0.1):
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
        self.corridors = {}
        # 圆形空域
        for i in range(num_annulus * 4):
            name = chr(65 + i)
            # 设置航向：偶数环为顺时针，奇数环为逆时针
            # begin_rad = 0
            
            rad = [[-0.16, np.pi/2], [np.pi/2, np.pi-0.16], [np.pi-0.16, 3*np.pi/2], [-np.pi/2, -0.16],
               [0.09, -np.pi/2+0.09], [-np.pi/2+0.09, -np.pi+0.09], [-np.pi+0.09, -3*np.pi/2+0.09], [np.pi/2+0.09, +0.09],
               [0, np.pi/2-0.06], [np.pi/2-0.06, np.pi], [np.pi, 3*np.pi/2-0.06], [-np.pi/2-0.06, 0]]
            
            self.corridors[name] = DirectionalPartialTorusCorridor(name=name,
                                                    anchor_point=np.array([0, 0, 0]),
                                                    orientation_vec=[0, 0, 1],
                                                    major_radius=(i // 4 + 1) * 10,
                                                    minor_radius=radius,
                                                    begin_rad=rad[i][0],
                                                    end_rad=rad[i][1],
                                                    connections=[],
                                                    reduce_space=self.reduce_space)
        

        # 直线管道空域，直线的anchor_point是管道中心点
        positions = [
            np.array([15, 0, 0]), np.array([23, 0, 0]),     # 0、1
            np.array([0, -25, 0]), np.array([0, -17, 0]),   # 2, 3
            np.array([-15, 0, 0]), np.array([-23, 0, 0]),   # 4, 5
            np.array([0, 25, 0]), np.array([0, 17, 0])      # 6, 7
        ]
        # 定义每个管道的航向（正向或反向）
        # 1 表示正向，-1 表示反向
        directions = [
            np.array([1, 0, 0]),    # 0、1、2
            np.array([0, 1, 0]),    # 3, 4, 5
            np.array([-1, 0, 0]),   # 6, 7, 8
            np.array([0, -1, 0])    # 9, 10, 11
        ]  # 可以根据需要自定义每个管道的航向
        for i in range(4 * (num_annulus - 1)):
            name = str(i)
            self.corridors[name] = CylinderCorridor(anchor_point=positions[i],   # 水平圆柱中心点
                                           orientation_vec=directions[i // 2],
                                           length=6 if i % 2 == 0 else 10, # 圆柱长度
                                           width=radius * 2, # 圆柱直径
                                           name=name,
                                           connections=[], 
                                           reduce_space=self.reduce_space)
            
        # 定义连接走廊的配置参数
        torus_configs = [
            # name, anchor_point, begin_rad, end_rad
            ('7B', [-2, 12, 0], 0, -np.pi/2),
            ('B4', [-12, 2, 0], 0, -np.pi/2),
            ('3D', [2, -12, 0], np.pi, np.pi/2),
            ('D0', [12, -2, 0], np.pi, np.pi/2),
            ('0E', [18, -2, 0], np.pi/2, 0),
            ('E3', [2, -18, 0], -np.pi/2, -np.pi),
            ('2F', [-2, -22, 0], 0, np.pi/2),
            ('F5', [-22, -2, 0], 0, np.pi/2),
            ('4G', [-18, 2, 0], -np.pi/2, -np.pi),
            ('G7', [-2, 18, 0], np.pi/2, 0),
            ('6H', [2, 22, 0], np.pi, 3*np.pi/2),
            ('H1', [22, 2, 0], np.pi, 3*np.pi/2),
            ('I6', [2, 28, 0], np.pi/2, np.pi),
            ('1I', [28, 2, 0], -np.pi/2, 0),
            ('K2', [-2, -28, 0], -np.pi/2, 0),
            ('5K', [-28, -2, 0], np.pi/2, np.pi)
        ]

        # 用循环创建所有环形走廊
        for name, anchor, begin_r, end_r in torus_configs:
            self.corridors[name] = DirectionalPartialTorusCorridor(
                name=name,
                anchor_point=np.array(anchor),
                orientation_rad=[0, 0],
                major_radius=2.0,
                minor_radius=2.0,
                begin_rad=begin_r,
                end_rad=end_r,
                connections=[name[1]],
                reduce_space=True
            )
            
        # 设定管道链接
        self.corridors['A'].connections = ['B']
        self.corridors['B'].connections = ['C', 'B4']
        self.corridors['C'].connections = ['D']
        self.corridors['D'].connections = ['A', 'D0']

        self.corridors['E'].connections = ['F', 'E3']
        self.corridors['F'].connections = ['G', 'F5']
        self.corridors['G'].connections = ['H', 'G7']
        self.corridors['H'].connections = ['E', 'H1']
        
        self.corridors['I'].connections = ['J', 'I6']
        self.corridors['J'].connections = ['K']
        self.corridors['K'].connections = ['L', 'K2']
        self.corridors['L'].connections = ['I']

        self.corridors['0'].connections = ['1', '0E']
        self.corridors['1'].connections = ['1I']

        self.corridors['2'].connections = ['3', '2F']
        self.corridors['3'].connections = ['3D']

        self.corridors['4'].connections = ['5', '4G']
        self.corridors['5'].connections = ['5K']

        self.corridors['6'].connections = ['7', '6H']
        self.corridors['7'].connections = ['7B']
            
        # if not test and corridor_index_awareness:
        #     assert len(seq) >= sum(corridor_index_awareness)

        # 状态中考虑的走廊数量
        DirectionalPartialTorusCorridor.num_corridor_in_state = num_corridor_in_state
        CylinderCorridor.num_corridor_in_state = num_corridor_in_state

        # setup uavs
        self.corridors_list = list(self.corridors.keys())
        corridor_graph = self.corridors['A'].convert2graph(self.corridors)  # 将走廊连接关系（字典类型）转换为图结构

        # 初始位置分布，在半径为2的圆内均匀分布无人机，确保最小距离为1。
        plane_offsets = distribute_evenly_within_circle(radius=2, min_distance=1, num_points=num_agents)
        UAV.flying_list = []
        self.agents = [UAV(init_corridor=np.random.choice(self.corridors_list[12:20], replace=False),
                            des_corridor=np.random.choice(self.corridors_list),
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
        seq = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11')
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
    
def visualization():
    def torus(R, r, begin_rad, end_rad, R_res=100, r_res=100):
        u = np.linspace(begin_rad, end_rad, R_res)
        v = np.linspace(0, 2 * np.pi, r_res)
        u, v = np.meshgrid(u, v)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return x, y, z

    def cylinder(r, h, theta_res=100, z_res=100):
        theta = np.linspace(0, 2 * np.pi, theta_res)
        z = np.linspace(-h / 2, h / 2, z_res)
        theta, z = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z
    
    def circle(ax, r, anchor_point, orientation_vec, length, rotation_matrix, color):
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.linspace(0, r, 5)
        Theta, R = np.meshgrid(theta, r)
        x_circle = R * np.cos(Theta)
        y_circle = R * np.sin(Theta)
        z_circle = np.zeros_like(x_circle)

        # 旋转到正确的方向
        # 圆面中心在圆柱末端
        end_center = anchor_point + orientation_vec * length / 2
        x_rot, y_rot, z_rot = [], [], []
        for a, b, c in zip(x_circle, y_circle, z_circle):
            x_p, y_p, z_p = np.dot(rotation_matrix, np.array([a, b, c]))
            x_rot.append(x_p + end_center[0])
            y_rot.append(y_p + end_center[1])
            z_rot.append(z_p + end_center[2])
        ax.plot_surface(np.array(x_rot), np.array(y_rot), np.array(z_rot), color=color, alpha=1.0)

    env = parallel_env()  # 创建并行环境
    env.reset(num_agents=5)
    fig = plt.figure()
    # 创建3d图形的两种方式
    # 将figure变为3d
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    # plt.draw()
    for _, corridor in env.corridors.items():
        if isinstance(corridor, CylinderCorridor):
            Xt, Yt, Zt = cylinder(r=corridor.radius,
                                    h=corridor.length)
            rotation_matrix = vec2vec_rotation(Z_UNIT, corridor.orientation_vec)
            ax.text(corridor.anchor_point[0], corridor.anchor_point[1], corridor.anchor_point[2]+5, 
                    corridor.name, color='black', fontsize=12, ha='center')

        elif isinstance(corridor, DirectionalPartialTorusCorridor):
            Xt, Yt, Zt = torus(R=corridor.major_radius,
                                r=corridor.minor_radius,
                                begin_rad=corridor.begin_rad,
                                end_rad=corridor.end_rad)
            rotation_matrix = vec2vec_rotation(Z_UNIT, corridor.orientation_vec)
            ax.text(corridor.endCirclePlane.anchor_point[0], 
                    corridor.endCirclePlane.anchor_point[1], 
                    corridor.endCirclePlane.anchor_point[2]+5, 
                    corridor.name, color='black', fontsize=12, ha='center')

        # Apply rotation
        x_rot_torus, y_rot_torus, z_rot_torus = [], [], []
        for a, b, c in zip(Xt, Yt, Zt):
            x_p, y_p, z_p = np.dot(rotation_matrix, np.array([a, b, c]))
            x_rot_torus.append(x_p + corridor.anchor_point[0])
            y_rot_torus.append(y_p + corridor.anchor_point[1])
            z_rot_torus.append(z_p + corridor.anchor_point[2])

        ax.plot_surface(np.array(x_rot_torus), np.array(y_rot_torus),
                                    np.array(z_rot_torus),
                                    edgecolor='royalblue',
                                    lw=0.1, rstride=20, cstride=4, alpha=0.1)
        


    # for agent in env.agents:
    #     position = agent.position
    #     corridor = env.corridors[agent.enroute['des']]
    #     rotation_matrix = vec2vec_rotation(Z_UNIT, corridor.orientation_vec)
    #     circle(ax, corridor.radius, corridor.anchor_point, corridor.orientation_vec, corridor.length, rotation_matrix, agent.color)
        
    #     ax.scatter(*position, s=50, c=agent.color, alpha=1, marker='o', depthshade=True)
    #     ax.text(position[0], position[1], position[2]+5, agent.agent_id, color=agent.color, fontsize=12, ha='center')
    #     print(f"uav_id:{agent.agent_id}, position:{position}, corridor:{agent.enroute['current']}, destination:{agent.enroute['des']}, path:{agent.enroute['path']}")

    plt.show()