# from pettingzoo.mpe import simple_adversary_v3
import collections
import json

import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import air_corridor.scenario.D3annulus as d3
from air_corridor.tools.util import load_init_params
from air_corridor.tools.visualization import Visualization as vl
from ppo import PPO

env = d3.parallel_env(render_mode="")

loadModel = True
if loadModel:
    init = False
    loadFolder = 'annulus_success/0'
    modelINdex = '6.5m'
    net_model = 'fc'

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)

    with open(f"{loadFolder}/net_params.json", 'r') as json_file:
        kwargs1 = json.load(json_file)
        # opt=json.load

    kwargs['net_model'] = net_model
    model = PPO(**kwargs)

    model.load(folder=loadFolder, global_step=modelINdex)

max_round = 5
ani = vl(max_round, to_base=False)

status = {}
for i in range(max_round):
    print(f"{i}/{max_round}")
    # 训练场景可能与测试场景不同，因此需要根据无人机训练时的设置匹配处于状态的走廊数和走廊指数感知。
    s, infos = env.reset(num_agents=5,
                         dt=1,
                         radius=2,
                         num_corridor_in_state=opt['num_corridor_in_state'],
                         corridor_index_awareness=opt['corridor_index_awareness'])
    current_actions = {}
    step = 0
    agents = env.agents
    ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=i)
    while env.agents:
        if loadModel:
            s1 = {agent: s[agent]['self'] for agent in env.agents}
            s2 = {agent: s[agent]['other'] for agent in env.agents}
            # s1_lst：(agent数量, 16+16=32)，返回自身状态和走廊状态（FlyingObject.py中的_report_self函数）
            s1_lst = [state for agent, state in s1.items()]
            # s2_lst：(agent数量, other_agent数量，22+16=38)，用于报告其他agent状态的方法（FlyingObject.py中的_report_other函数）
            s2_lst = [state for agent, state in s2.items()]
            a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)
            actions = {agent: a for agent, a in zip(env.agents, a_lst)}
        else:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        action_updated = False
        s, rewards, terminations, truncations, infos = env.step(actions)
        ani.put_data(round=i, agents={agent: agent.position for agent in env.agents})
        # print(rewards)

        for agent in env.agents:
            if agent.status != 'Normal' and agent not in status:
                status[agent] = agent.status
        env.agents = [agent for agent in env.agents if not agent.terminated]
        step += 1
        # print(step)

state_count = collections.Counter(status.values())
print(state_count)
ani.show_animation(mp4=False, gif=False)

env.close()
