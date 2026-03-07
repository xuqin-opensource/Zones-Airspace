import argparse
import glob
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import reduce

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import air_corridor.scenario.D3annulus as d3
from air_corridor.tools.log_config import setup_logging
from air_corridor.tools.util import save_init_params
from ppo import PPO


def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=float, default=1000000, help='which model to load')
parser.add_argument('--LoadFolder', type=str, default=None, help='Which folder to load')
parser.add_argument('--complexity', type=str, default='simple', help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--time', type=str, default=None, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--exp_name', type=str, default="0:0", help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--EnvIdex', type=int, default=2, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--video_turns', type=int, default=50, help='which model to load')
parser.add_argument('--num_agents', type=int, default=5, help='Decay rate of entropy_coef')
parser.add_argument('--dt', type=float, default=1, help='Decay rate of entropy_coef')
parser.add_argument('--reduce_space', type=str2bool, default=True, help='Share feature extraction layers?')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--distnum', type=int, default=0, help='0:Beta ; 1:GS_ms;  2: GS_m')
parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e4, help='Model evaluating interval, in steps.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--activation', type=str, default='tanh', help='activation function')
parser.add_argument('--a_lr', type=float, default=1.5e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1.5e-5, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--share_layer_flag', type=str2bool, default=True, help='Share feature extraction layers?')
parser.add_argument('--multiply_horrizion', type=int, default=8, help='Share feature extraction layers?')
parser.add_argument('--multiply_batch', type=int, default=16, help='Share feature extraction layers?')
parser.add_argument('--reduce_epoch', type=str2bool, default=False, help='Share feature extraction layers?')
parser.add_argument('--curriculum', type=str2bool, default=True, help='gradually increase range')
parser.add_argument('--consider_boid', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--token_query', type=str2bool, default=True, help='tokenize s1 for query')
parser.add_argument('--trans_position', type=str2bool, default=False, help='token input with position')
parser.add_argument('--num_enc', type=int, default=4, help='number of encoders')
parser.add_argument('--net_model', type=str, default='fc', help='number of encoders')
parser.add_argument('--liability', type=str2bool, default=True, help='number of encoders')
parser.add_argument('--collision_free', type=str2bool, default=False, help='number of encoders')
parser.add_argument('--beta_adaptor_coefficient', type=float, default=1.1, help='number of encoders')
parser.add_argument('--beta_base', type=float, default=1.0, help='number of encoders')
parser.add_argument('--level', type=int, default=14, help='Share feature extraction layers?')
parser.add_argument('--num_corridor_in_state', type=int, default=1, help='number of encoders')
parser.add_argument('--corridor_index_awareness', type=str, default=None, help='indicate the corridor index')
parser.add_argument('--acceleration_max', type=float, default=0.3, help='Learning rate of actor')
parser.add_argument('--velocity_max', type=float, default=1.5, help='Learning rate of actor')
parser.add_argument('--base_difficulty', type=float, default=0.2, help='Learning rate of actor')
parser.add_argument('--ratio', type=float, default=0.5, help='How much percent for torus?')
parser.add_argument('--uniform_state', type=str2bool, default=False, help='number of encoders')
parser.add_argument('--dynamic_minor_radius', type=str2bool, default=False, help='Share feature extraction layers?')

opt = parser.parse_args()

if opt.corridor_index_awareness:
    opt.corridor_index_awareness = [int(i) for i in opt.corridor_index_awareness]
opt.T_horizon *= opt.multiply_horrizion
opt.a_optim_batch_size *= opt.multiply_batch
opt.c_optim_batch_size *= opt.multiply_batch
opt.save_interval = 2.5e5
opt.Max_train_steps = 1e7


def main():
    # 初始化和设置
    write = opt.write  # Use SummaryWriter to record the training.
    render = opt.render # 是否渲染环境
    # 创建运行目录和日志
    exp_name = opt.exp_name.split(':')  # 分割实验名称
    if opt.time is None:
        run_name = f"annulus_{int(time.time())}_{exp_name[0]}"
    else:
        run_name = f"annulus_{opt.time}_{exp_name[0]}"
    env = d3.parallel_env(render_mode='rgb_array')  # 创建并行环境
    exp_name = ''.join(exp_name[1:])
    if exp_name is None:
        dir = f'{run_name}'
    else:
        dir = f'{run_name}/{exp_name}'

    # 创建目录和日志
    if not os.path.exists(dir): os.makedirs(dir)
    logger = setup_logging(f"{run_name}/process_log.txt", logging.INFO)

    # 设置最大步数和轨迹长度
    max_steps = 500
    T_horizon = opt.T_horizon  # lenth of long trajectory

    # 定义概率分布类型
    Dist = ['Beta', 'GS_ms', 'GS_m']  # type of probility distribution，Beta分布，高斯混合等
    distnum = opt.distnum

    # 训练参数
    Max_train_steps = opt.Max_train_steps   # 最大训练步数
    save_interval = opt.save_interval  # 模型保存间隔
    eval_interval = opt.eval_interval  # 评估间隔

    # 设置随机数种子
    seed = opt.seed
    logger.info("Random Seed: {}".format(seed))
    torch.manual_seed(seed)
    # env.seed(seed)
    # eval_env.seed(seed)
    np.random.seed(seed)

    # TensorBoard日志记录器设置
    if write:   # 如果需要记录训练日志
        if opt.Loadmodel:   # 判断是否从预训练模型加载
            file_pattern = f"{opt.LoadFolder}/events.out.tfevents.*"    # 构建文件匹配模式
            files = glob.glob(file_pattern) # 查找符合file_pattern命名规则的文件
            summary_file = files[0]
            if os.path.exists(summary_file):    # 检查文件是否存在
                logger.info('Summary exists')   # 输出日志信息，表明找到了已有的事件文件
                src = summary_file  # 定义源文件路径
                dst = f"{run_name}/{exp_name}"  # 定义目标目录路径
                if not os.path.exists(dst): os.makedirs(dst)
                logger.info(f"src: {src} \n"
                            f"dst: {dst}")  # 显示要复制的源文件和目标位置
                # 使用shutil.copy将TensorBoard事件文件从源位置复制到目标位置，这样可以在新的训练中继续使用之前的训练历史
                shutil.copy(src=summary_file, dst=dst)  # 设置TensorBoard写入路径
                writepath = dst
        else:
            logger.info('did not find summary')
            if exp_name is None:
                writepath = f"{run_name}"
            else:
                writepath = f"{run_name}/{exp_name}"
        writer = SummaryWriter(f"{writepath}")  # 创建TensorBoard写入器，SummaryWriter会将训练指标（损失、准确率等）写入指定目录

    # PPO模型配置
    kwargs = {
        "state_dim": 26,    # 状态维度
        "s2_dim": 22,       # 第二个状态维度
        "action_dim": 3,    # 动作维度
        "env_with_Dead": True,  # 环境是否包含终止状态
        "gamma": opt.gamma,
        "lambd": opt.lambd, # GAE（广义优势估计）的λ参数
        "clip_rate": opt.clip_rate, # PPO剪裁率，通常为0.2
        "K_epochs": opt.K_epochs,   # 每次更新的epoch数    
        "net_width": opt.net_width, # 网络宽度
        "a_lr": opt.a_lr,       # Actor网络学习率
        "c_lr": opt.c_lr,       # Critic网络学习率
        "dist": Dist[distnum],  # 动作分布类型
        "l2_reg": opt.l2_reg,  # L2 regulization for Critic
        "a_optim_batch_size": opt.a_optim_batch_size,   # Actor优化批量大小
        "c_optim_batch_size": opt.c_optim_batch_size,   # Critic优化批量大小
        "entropy_coef": opt.entropy_coef,   # 熵系数，鼓励探索
        # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay": opt.entropy_coef_decay,   # 熵系数衰减
        'activation': opt.activation,       # 激活函数
        'share_layer_flag': opt.share_layer_flag,   # 是否共享网络层
        'anneal_lr': True,      # 是否退火学习率
        'totoal_steps': opt.Max_train_steps,    # 总训练步数
        'with_position': opt.trans_position,    # 是否包含位置信息
        'token_query': opt.token_query, # 是否使用token查询
        'num_enc': opt.num_enc, # 编码器数量
        'dir': dir, # 保存目录
        "writer": writer,   # TensorBoard写入器
        'logger': logger,   # 日志记录器
        'net_model': opt.net_model, # 网络模型类型
        'beta_base': opt.beta_base  # Beta分布基础参数
    }

    if opt.Loadmodel:   # 加载预训练模型
        model = PPO(**kwargs)
        model.load(folder=opt.LoadFolder,
                   global_step=opt.ModelIdex)
        total_steps = opt.ModelIdex # 从指定步数继续训练
    else:   # 创建新模型
        save_init_params(name='net_params', **kwargs)   # 保存网络参数
        opt_dict = vars(opt)
        opt_dict['dir'] = dir
        save_init_params(name='main_params', **vars(opt))
        model = PPO(**kwargs)
        total_steps = 0

    traj_lenth = 0  # 保存网络参数

    videoing = False    # 是否录制视频
    ready_for_train = False
    ready_for_log = True
    logger.info(opt)
    episodes = 0    # 当前批次episode数
    total_episode = 0   # 总episode数
    trained_times = 0   # 训练次数
    extra_save_index = 0
    start_time = time.time()
    epsilon = 0.1
    if opt.curriculum: # 是否启用课程难度
        env_options = {'difficulty': opt.base_difficulty}   # 基础难度
    else:
        env_options = {'difficulty': 1} # 固定难度

    # 训练循环
    while total_steps < Max_train_steps:
        # active_agents = [{'terminated': False, 'trajectory': []} for _ in range(num_agents)]
        steps = 0   # 当前episode步数
        # 环境重置
        s, init_info = env.reset(num_agents=opt.num_agents,
                                 reduce_space=opt.reduce_space,
                                 collision_free=opt.collision_free,
                                 liability=opt.liability,
                                 beta_adaptor_coefficient=opt.beta_adaptor_coefficient, # 调整方位角φ的敏感度，在训练中可以限制或扩展水平转向范围
                                 num_corridor_in_state=opt.num_corridor_in_state,   # 神经网络可处理的走廊数量
                                 dt=opt.dt,
                                 consider_boid=opt.consider_boid,
                                 corridor_index_awareness=opt.corridor_index_awareness,
                                 velocity_max=opt.velocity_max,
                                 acceleration_max=opt.acceleration_max,
                                 uniform_state=opt.uniform_state,
                                 epsilon=epsilon)
        # 第一次状态更新：提取所有智能体的新状态，用于存储到经验回放缓冲区（agent.trajectory.append()）
        # {'self': uav_status, 'other': other_uavs_status}
        s1 = {agent: s[agent]['self'] for agent in env.agents}  # 自身状态
        s2 = {agent: s[agent]['other'] for agent in env.agents} # 其他智能体状态
        if ready_for_log:   # 如果需要记录日志
            model.weights_track(total_steps)    # 跟踪网络权重
            ready_for_log = False
            videoing = True 
            turns = 0   # 视频录制轮次
            scores = 0  # 累计得分
            lst_std_variance = []   # Beta分布方差列表
            env.anima_recording = True  # 启用动画录制
            status_summary = defaultdict(list)  # 访问不存在的键，返回空列表
        episodes += 1
        total_episode += 1
        '''Interact & trian'''
        while env.agents:   # 当还有存活的智能体时
            traj_lenth += 1 # 轨迹长度增加
            steps += 1  # 步数增加

            s1_lst = [state for agent, state in s1.items()]
            s2_lst = [state for agent, state in s2.items()]
            if render:  # 是否渲染环境
                env.render()
                a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)   # 评估模式
            else:
                # 选择动作
                a_lst, logprob_a_lst, alpha, beta = model.select_action(s1_lst, s2_lst)
            # 格式化动作和概率
            logprob_a = {agent: logprob for agent, logprob, in zip(env.agents, logprob_a_lst)}
            a = {agent: a for agent, a in zip(env.agents, a_lst)}

            s_prime, r, terminated, truncated, info = env.step(a)   # 执行动作，获取新状态
            # 更新状态
            s1_prime = {agent: s_prime[agent]['self'] for agent in env.agents}
            s2_prime = {agent: s_prime[agent]['other'] for agent in env.agents}

            """
            terminated[agent]：正常终止标志，通常是任务成功完成（如到达目标）或者失败（如碰撞、坠毁）
            truncated[agent]：截断终止标志，通常是超时终止（达到最大步数）
            
            | 运算符：按位或（or）运算，在布尔值上相当于逻辑或
            True | False = True
            False | True = True
            False | False = False
            """
            done = {agent: terminated[agent] | truncated[agent] for agent in env.agents}    # 判断是否终止

            '''
            distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax
            dw for TD_target and Adv; done for GAE
            done[agent]：上一行计算出的终止标志，True：智能体已终止，False：智能体未终止
            steps != max_steps：当前步数不等于最大步数，max_steps = 500（代码开头定义的），steps：当前episode已进行的步数
            如果 steps != max_steps 为 True，表示不是因为超时终止的
            '''
            dw = {agent: done[agent] and steps != max_steps for agent in env.agents}
            for agent in env.agents:
                agent.trajectory.append([s1[agent],
                                         s2[agent],
                                         a[agent],
                                         r[agent],
                                         s1_prime[agent],
                                         s2_prime[agent],
                                         logprob_a[agent],
                                         done[agent],
                                         dw[agent]])
                if done[agent]: # 如果智能体终止
                    for transition in agent.trajectory:
                        model.put_data(agent, transition)   # 将数据存入经验池
                        # model.put_data( transition)
            # 视频录制和评估
            if videoing:    # 如果正在录制视频
                # ani.put_data(round=turns, agents={agent: agent.position for agent in env.agents})
                for agent in env.agents:
                    if done[agent]:
                        status_summary[init_info['corridor_seq']].append(agent.status)  # 每个走廊序列下agent状态为[['Normal'], ['collided'], ['Won']]
                        scores += agent.accumulated_reward  # 所有agent的得分总和

                ## calculate beta distribution variance
                alpha = np.array(alpha.to('cpu'))
                beta = np.array(beta.to('cpu'))
                variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                # Calculate the standard deviation for each element
                std_dev = np.sqrt(variance)
                # Calculate the average standard deviation for each dimension
                # std_dev = np.squeeze(std_dev)
                lst_std_variance.append(std_dev)

                # all() 函数用于判断给定的可迭代参数iterable 中的所有元素是否都为TRUE，如果是返回True，否则返回False
                if all(done.values()):
                    turns += 1
                if turns == opt.video_turns:
                    videoing = False    # 若录制轮次，达到设置轮次则停止录制
                    average_score = scores / opt.video_turns / opt.num_agents   # 计算平均分
                    status_lst = reduce(lambda x, y: x + y, status_summary.values())
                    counter = Counter(status_lst)   # 用于获取元素出现的次数。它返回一个按照出现次数从高到低排序的元素列表，其中每个元素都是一个(key, count)的元组。
                    # 输出日志信息
                    logger.info(
                        f"seed: {seed}, steps: {int(total_steps / 1000)}k, diffic: {env_options['difficulty']}, score: {round(average_score, 2)}, status: {counter}, {opt.exp_name}")

                    ## average beta distribution variance
                    stacked_std = np.vstack(lst_std_variance)
                    mean_array = np.mean(stacked_std, axis=0)

                    if write:   # 若启用TensorBoard
                        for key, values in status_summary.items():
                            writer.add_scalar(f"scenario/{key}", Counter(values)['won'] / len(values), total_steps) # 记录不同场景的胜率
                        won_percent = counter['won'] / sum(counter.values())    # 总胜率
                        # 记录各种指标到TensorBoard
                        # 'charts/reward_steps'：标量名称（图表标题）。
                        # average_score：标量值（需为 float，若是 PyTorch 张量需 .item() 转换）。
                        # total_steps：横轴步数（通常是迭代次数或 epoch 数）。
                        writer.add_scalar('charts/reward_steps', average_score, total_steps)
                        writer.add_scalar('charts/reward_episodes', average_score, total_episode)
                        writer.add_scalar("charts/won_percent", counter['won'] / sum(counter.values()), total_steps)
                        writer.add_scalar("charts/collide_percent", counter['collided'] / sum(counter.values()),
                                          total_steps)
                        writer.add_scalar('charts/difficulty', env_options['difficulty'], total_steps)
                        writer.add_scalar('charts/uni_r_steps', average_score * min(1, env_options['difficulty']),
                                          total_steps)
                        writer.add_scalar("charts/uni_won_percent", won_percent * min(1, env_options['difficulty']),
                                          total_steps)
                        writer.add_scalar("dist/beta_std_phi", mean_array[2], total_steps)
                        writer.add_scalar("dist/beta_std_theta", mean_array[1], total_steps)
                        writer.add_scalar("dist/beta_std_r", mean_array[0], total_steps)

                    if opt.curriculum:  # 课程学习难度调整
                        # difficulty 1 is np.pi/2;   1.2 corresponds to slightly larger than
                        maxDiff = 1.0   # 最大难度
                        diffThreshold = 0.8 # 胜率阈值
                        if env_options['difficulty'] < maxDiff and won_percent >= diffThreshold:
                            if env_options['difficulty'] == 1 and epsilon > 1e-5:
                                epsilon /= 2    # 减小探索率
                            else:
                                epsilon = 0
                                # epsilon = epsilon
                            env_options['difficulty'] = min(env_options['difficulty'] + 0.1, maxDiff)   # 增加难度
                            # model.save(total_steps, env_options['difficulty'])
                        # elif env_options['difficulty'] > base_difficulty and won_percent < 0.9 / 2:
                        #     env_options['difficulty'] = min(env_options['difficulty'] - 0.05, maxDiff)
            # 第二次状态更新：筛选存活智能体的状态，生命周期终止的角色移出队伍，只带存活角色进入下一关
            env.agents = [agent for agent in env.agents if not done[agent]]
            s1 = {agent: s1_prime[agent] for agent in env.agents}
            s2 = {agent: s2_prime[agent] for agent in env.agents}

            '''update if its time'''
            if traj_lenth % T_horizon == 0: # 达到轨迹长度阈值，准备训练
                ready_for_train = True
            if ready_for_train and not env.agents:
                ready_for_train = False
                # trained_between_evaluations = True
                if opt.reduce_epoch:
                    epoches = int(np.power(episodes, 1 / 2.3))
                else:
                    epoches = opt.K_epochs

                model.train(total_steps, epoches)   # 执行PPO训练
                # model.save(total_steps, extra_save_index)
                extra_save_index += 1
                trained_times += 1
                # 记录训练信息
                logger.info(f"{episodes}, {epoches}, {opt.exp_name}")
                writer.add_scalar('weights/episodes', episodes, total_steps)
                writer.add_scalar('weights/epoches', epoches, total_steps)
                traj_lenth = 0  # 重置轨迹长度
                episodes = 0    # 重置episode计数
            total_steps += 1

            '''record & log'''
            if total_steps % save_interval == 0:    # 定期保存模型
                model.save(total_steps)
            if trained_times == 2:  # 每训练2次记录一次日志
                ready_for_log = True
                trained_times = 0

            # if total_steps % 10 == 0:
            #     logger.info(total_steps, exp_name)
    env.close() # 关闭环境


if __name__ == '__main__':
    log_file = "error_log.txt"
    log_format = "%(asctime)s [%(levelname)s]: %(message)s"

    main()

    # d3.visualization()