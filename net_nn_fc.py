# from modules import SAB, PMA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from air_corridor.tools.util import nan_recoding
from modules import FcModule, Tokenizer, MAB


## trans for neighbors, then no-input-query
## combined with self

class SmallSetTransformer(nn.Module):
    def __init__(self, neighbor_dimension=7, net_width=256, with_position=False, token_query=False, num_enc=4,
                 logger=None):
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, 1, net_width))
        nn.init.xavier_uniform_(self.S) # Xavier初始化权重，Var(w) = 2 / (n_in + n_out)
        """
        构建Transformer编码器，这个编码器用于处理输入的集合数据，学习元素间的关系
        d_model=net_width: 模型维度（特征维度）
        nhead=4: 注意力头数
        dim_feedforward=512: 前馈网络维度
        batch_first=True: 批次维度在前
        num_layers=num_enc: 编码器层数（默认为4）
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=net_width, nhead=4, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc)
        """
        使用MAB（Multi-head Attention Block）作为解码器
        这是一个自定义的注意力模块（不是PyTorch官方的TransformerDecoder）
        参数：输入维度、输出维度、键值维度都是net_width，num_heads=4: 注意力头数，ln=True: 使用LayerNorm
        """
        self.decoder_mab = MAB(net_width, net_width, net_width, num_heads=4, ln=True)
        # pytorch official decoder, having bugs
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=net_width, nhead=8, activation='gelu',
                                                        dim_feedforward=512,
                                                        batch_first=True)
        self.tk = Tokenizer(output_dim=net_width)
        self.fc = nn.Linear(net_width, net_width)
        self.with_position = with_position
        self.token_query = token_query
        self.fc1 = nn.Linear(2 * net_width, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.fc3 = nn.Linear(net_width, net_width)
        self.logger = logger
        self.fc_module = FcModule(net_width)    # 自定义的全连接模块

        ########################################
        # self.bn1 = nn.BatchNorm1d(net_width)
        # self.bn2 = nn.BatchNorm1d(net_width)
        # self.fc3 = nn.Linear(net_width, net_width)
        # ########################################

    def forward(self, x, state):
        """
        x: 输入集合数据，形状为(batch_size, sequence_length, net_width)
        state: 外部状态/上下文信息，形状为(batch_size, net_width)
        """
        x1 = self.encoder(x)    # 通过Transformer编码器处理集合数据
        nan_recoding(self.logger, x1, 'encoding')   # 检测并记录编码输出中是否有NaN值
        query = self.S.repeat(x.size(0), 1, 1)  # 准备查询向量将可学习的种子token（S）扩展到批次大小，即从(1, 1, net_width)扩展到(batch_size, 1, net_width)
        # 使用MAB解码器，以种子token为查询，编码特征为键值，实现注意力机制：查询向量从编码特征中提取信息，输出形状为(batch_size, 1, net_width)
        x7 = self.decoder_mab(query, x1)
        x7 = x7.view(x7.size(0), -1)    # 展平解码器输出，从(batch_size, 1, net_width)变为(batch_size, net_width)
        x7 = torch.cat([x7, state], dim=1)  # 特征融合-将解码特征与外部状态连接，将两者在特征维度上拼接，输出形状为(batch_size, 2*net_width)
        x8 = self.fc_module(x7) # 通过全连接模块处理并返回最终特征
        return x8   # 返回形状为(batch_size, net_width)


class FixedBranch(nn.Module):
    def __init__(self, input_dimension=11, net_width=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class BetaActorMulti(nn.Module):    # 连续动作空间的Beta分布策略网络
    def __init__(self, s1_dim, s2_dim, action_dim, net_width, shared_layers=None, beta_base=1.0):
        super(BetaActorMulti, self).__init__()
        self.fc1 = nn.Linear(net_width, net_width)
        self.fc2_a = nn.Linear(net_width, int(net_width / 2))
        self.bn1 = nn.BatchNorm1d(int(net_width / 2))
        self.fc2_b = nn.Linear(int(net_width / 2), net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)
        if shared_layers is None:
            self.intput_merge = MergedModel(s1_dim, s2_dim, net_width)
        else:
            self.intput_merge = shared_layers
        self.beta_base = beta_base  # Beta分布的基础参数，防止参数为0

    def forward(self, s1, s2):
        merged_input = self.intput_merge(s1, s2)    # 通过MergedModel处理得到两种状态的融合特征
        x = F.relu(self.fc1(merged_input))  # 第一层变换 + ReLU激活
        x_a = F.relu(self.fc2_a(x)) # 先通过fc2_a降维，再ReLU激活
        x_b = F.relu(self.fc2_b(x_a)) + x   # 残差连接
        alpha = F.softplus(self.alpha_head(x_b)) + self.beta_base   # softplus: 确保输出为正数 + self.beta_base: 加上基础值，防止参数过小接近0
        beta = F.softplus(self.beta_head(x_b)) + self.beta_base
        return alpha, beta  # 此时α > 0, β > 0

    def get_dist(self, s1, s2, log):
        nan_event = False
        alpha, beta = self.forward(s1, s2)

        # 检测并处理α中的NaN值
        nan_mask = torch.isnan(alpha)   # 如果有NaN，标记事件并记录日志
        if nan_mask.sum() > 0:
            nan_event = True
            log.info(f"s1: {s1}")
            log.info(f"s2: {s2}")
            log.info(f"alpha: {alpha}")
            log.info(f"alpha with shape {alpha.shape} has {nan_mask.sum()} nan")
            alpha[nan_mask] = torch.rand(nan_mask.sum()).to(alpha.device)   # 用随机数替换NaN值，避免训练崩溃

        # 同样处理β中的NaN值
        nan_mask = torch.isnan(beta)
        if nan_mask.sum() > 0:
            nan_event = True
            log.info(f"s1: {s1}")
            log.info(f"s2: {s2}")
            log.info(f"beta: {beta}")
            log.info(f"beta with shape {beta.shape} has {nan_mask.sum()} nan")
            beta[nan_mask] = torch.rand(nan_mask.sum()).to(beta.device)
        dist = Beta(alpha, beta)    # 创建Beta分布，将神经网络的输出转换为一个可以采样、可以计算概率的动作分布
        return dist, alpha, beta, nan_event

    def dist_mode(self, s1, s2):
        alpha, beta = self.forward(s1, s2)
        mode = (alpha-1+1e-5) / (alpha + beta-2+2e-5)   # 评估时使用众数而非采样
        return mode

class CriticMulti(nn.Module):
    def __init__(self, s1_dim, s2_dim, net_width, shared_layers=None):
        super(CriticMulti, self).__init__()
        self.C4 = nn.Linear(net_width, 1)
        if shared_layers is None:
            self.intput_merge = MergedModel(s1_dim, s2_dim, net_width)
        else:
            self.intput_merge = shared_layers

    def forward(self, s1, s2):
        merged_input = self.intput_merge(s1, s2)
        v = self.C4(merged_input)
        return v


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class MergedModel(nn.Module):
    def __init__(self, s1_dim, s2_dim, net_width, with_position, token_query, num_enc, logger=None):    # s1_dim, s2_dim: 两种输入状态的维度
        super(MergedModel, self).__init__()
        # self.fixed_branch = FixedBranch(s1_dim, net_width)
        self.trans = SmallSetTransformer(net_width, net_width, with_position, token_query, num_enc, logger) # 返回形状为(batch_size, net_width)
        self.net_width = net_width
        self.tk1 = Tokenizer(output_dim=net_width)  # tk1: 用于处理第一种状态s1
        self.tk2 = Tokenizer(output_dim=net_width)  # tk2: 用于处理第二种状态s2，两者输出维度都是net_width，确保维度一致

        self.with_position = with_position  # 记录是否使用位置编码
        self.logger = logger    # 保存日志记录器

    def forward(self, s1, s2=None):
        s3 = s2
        s1_p = self.tk1(s1) # 作为状态上下文
        s1_p = s1_p.squeeze(1)  # 使用squeeze(1)移除第1维（可能是batch维度外的额外维度）
        s3_p = self.tk2(s3) # 作为主要输入
        x = self.trans(s3_p, state=s1_p)    # 融合两种特征，模型基于s1的状态信息来处理s2的集合数据
        nan_recoding(self.logger, x, 'trans_output')    # 记录变换器输出中是否出现NaN值
        return x
